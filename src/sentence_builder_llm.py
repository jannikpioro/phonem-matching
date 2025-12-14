#!/usr/bin/env python3
"""
Generate bilingual sentences using LLM assistance and grammar validation.
Supports both local Ollama (including Docker containers) and cloud LLM providers.
"""

import json
import spacy
import subprocess
from pathlib import Path
from tqdm import tqdm

try:
    import requests
except ImportError:
    print("Installing requests...")
    subprocess.run(["pip", "install", "requests"], check=True)
    import requests


def call_ollama_llm(prompt, model="llama3.1", use_docker=False, container_name="open-webui"):
    """Call Ollama LLM (local API or Docker container)."""
    try:
        if use_docker:
            # Use Ollama inside Docker container
            cmd = [
                "docker", "exec", container_name, "ollama", "run", model, prompt
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout for large models
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"Docker Ollama error: {result.stderr}")
                return None
        else:
            # Use local Ollama API
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.7,
                        'top_p': 0.9,
                    }
                },
                timeout=600  # 10 minutes timeout
            )
            
            if response.status_code == 200:
                return response.json()['response'].strip()
            else:
                print(f"Ollama error: {response.status_code}")
                return None
    except Exception as e:
        print(f"Ollama connection error: {e}")
        if use_docker:
            print(f"Make sure Docker container '{container_name}' is running")
        else:
            print("Make sure Ollama is running: ollama serve")
        return None



        return None


def check_grammar(sentence, nlp, language="en"):
    """Check grammar quality using spaCy dependency parsing."""
    doc = nlp(sentence)
    
    score = 0
    issues = []
    
    # Check for basic sentence structure
    has_verb = any(token.pos_ == "VERB" for token in doc)
    has_noun = any(token.pos_ in ["NOUN", "PROPN", "PRON"] for token in doc)
    has_root = any(token.dep_ == "ROOT" for token in doc)
    
    if has_verb:
        score += 30
    else:
        issues.append("Missing verb")
    
    if has_noun:
        score += 30
    else:
        issues.append("Missing noun/pronoun")
    
    if has_root:
        score += 20
    else:
        issues.append("No root dependency")
    
    # Check for subject
    has_subject = any(token.dep_ in ["nsubj", "nsubjpass", "csubj"] for token in doc)
    if has_subject:
        score += 20
    else:
        issues.append("Missing subject")
    
    # Cap score at 100
    score = min(score, 100)
    
    return score, issues


def generate_sentence_pair_with_llm(word_pairs, llm_function):
    """
    Generate matching German and English sentences using the same word pairs in the same order.
    
    Args:
        word_pairs: List of (de_word, en_word, similarity, category) tuples - ALL available words
        llm_function: Function to call LLM
    
    Returns:
        Tuple of (german_sentence, english_sentence, words_used) or (None, None, None)
    """
    de_words = [pair[0] for pair in word_pairs]
    en_words = [pair[1] for pair in word_pairs]
    
    de_word_list = ", ".join([f'"{w}"' for w in de_words])
    en_word_list = ", ".join([f'"{w}"' for w in en_words])
    
    # Create word pair mapping for the prompt
    pair_list = "\n".join([f'  {i+1}. "{de}" → "{en}"' for i, (de, en, _, _) in enumerate(word_pairs)])
    
    prompt = f"""You have these German-English word pairs:
{pair_list}

Task: Create TWO sentences that sound phonetically similar:
1. A German sentence using ONLY German words from the pairs above
2. An English sentence using ONLY English words from the pairs above

CRITICAL RULES:
- Use the SAME word pairs in BOTH sentences
- Use them in the SAME ORDER (e.g. if "Hund" is word 2 in German, "hound" must be word 2 in English)
- Use ONLY words from the list above
- You MAY also use these grammatical words: 
  * German: der, die, das, er, ich, sie, es, wir, ihr, dein
  * English: the, I, he, her, she, it, we, you, we're
- Do NOT use any other words
- Both sentences must be grammatically correct and should make sense
- The sentences should sound similar when spoken
- IMPORTANT: Avoid Lists. Make them natural sentences that are grammatically correct (4-10 words).

Format your response EXACTLY like this:
GERMAN: [your German sentence]
ENGLISH: [your English sentence]

Example:
GERMAN: Der Hund bellt laut
ENGLISH: The hound barks loud

Your response:"""
    
    response = llm_function(prompt)
    
    if response:
        # Clean up response
        text = response.strip()
        
        # Remove thinking blocks if present
        if "...done thinking." in text:
            parts = text.split("...done thinking.")
            if len(parts) > 1:
                text = parts[-1].strip()
        
        if text.startswith("Thinking..."):
            lines = text.split("\n")
            for i, line in enumerate(lines):
                if "GERMAN:" in line.upper():
                    text = "\n".join(lines[i:])
                    break
        
        # Extract German and English sentences
        german_sentence = None
        english_sentence = None
        
        for line in text.split("\n"):
            line = line.strip()
            if line.upper().startswith("GERMAN:"):
                german_sentence = line.split(":", 1)[1].strip().strip('"\'')
            elif line.upper().startswith("ENGLISH:"):
                english_sentence = line.split(":", 1)[1].strip().strip('"\'')
        
        if german_sentence and english_sentence:
            # Validate that ALL words in both sentences come from the word pairs
            de_words_lower = set(w.lower() for w in de_words)
            en_words_lower = set(w.lower() for w in en_words)
            
            # Allow common grammatical words
            allowed_de_extras = {'der', 'die', 'das', 'er', 'ich', 'sie', 'es', 'wir', 'ihr'}
            allowed_en_extras = {'the', 'i', 'he', 'her', 'she', 'it', 'we', 'you', "we're"}
            
            # Tokenize sentences (simple split by whitespace and remove punctuation)
            import re
            de_tokens = [re.sub(r'[^\w]', '', w.lower()) for w in german_sentence.split() if w]
            en_tokens = [re.sub(r'[^\w]', '', w.lower()) for w in english_sentence.split() if w]
            
            # Check if all German words are in the allowed list or exceptions
            invalid_de_words = [w for w in de_tokens if w and w not in de_words_lower and w not in allowed_de_extras]
            invalid_en_words = [w for w in en_tokens if w and w not in en_words_lower and w not in allowed_en_extras]
            
            if invalid_de_words:
                print(f"    ⚠ German sentence contains invalid words: {invalid_de_words}")
                return None, None, None
            
            if invalid_en_words:
                print(f"    ⚠ English sentence contains invalid words: {invalid_en_words}")
                return None, None, None
            
            # Extract which word pairs were actually used
            words_used = []
            de_words_in_sentence = german_sentence.lower().split()
            en_words_in_sentence = english_sentence.lower().split()
            
            # Simple check: see which word pairs appear in both sentences
            for de_word, en_word, similarity, category in word_pairs:
                de_in_german = any(de_word.lower() in w for w in de_words_in_sentence)
                en_in_english = any(en_word.lower() in w for w in en_words_in_sentence)
                if de_in_german and en_in_english:
                    words_used.append((de_word, en_word, similarity, category))
            
            return german_sentence, english_sentence, words_used
    
    return None, None, None


def build_sentences_with_llm(
    matches_file,
    output_file,
    llm_function,
    min_similarity=0.85,
    min_grammar_score=75,
    max_sentences=100,
    max_attempts=500
):
    """
    Build bilingual sentences using LLM generation and spaCy validation.
    """
    # Load spaCy models for grammar checking
    print("Loading spaCy models for validation...")
    try:
        nlp_de = spacy.load("de_dep_news_trf")
        nlp_en = spacy.load("en_core_web_trf")
    except OSError:
        print("Transformer models not found, using large models...")
        try:
            nlp_de = spacy.load("de_core_news_lg")
            nlp_en = spacy.load("en_core_web_lg")
        except OSError:
            print("Installing spaCy models...")
            subprocess.run(["python", "-m", "spacy", "download", "de_core_news_lg"], check=True)
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"], check=True)
            nlp_de = spacy.load("de_core_news_lg")
            nlp_en = spacy.load("en_core_web_lg")
    
    # Load word matches
    print(f"Loading matches from {matches_file}...")
    with open(matches_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract all matches from nested structure
    all_matches = []
    for category, category_data in data.items():
        if isinstance(category_data, dict) and 'matches' in category_data:
            for german_entry in category_data['matches']:
                german_word = german_entry['german']
                german_phonetic = german_entry['german_phonetic']
                
                for match in german_entry['matches']:
                    all_matches.append({
                        'german_word': german_word,
                        'german_phonetic': german_phonetic,
                        'english_word': match['english'],
                        'english_phonetic': match['english_phonetic'],
                        'similarity': match['similarity'],
                        'category': category
                    })
    
    # Filter high-quality matches
    high_quality = [
        match for match in all_matches 
        if match['similarity'] >= min_similarity
    ]
    
    print(f"Found {len(all_matches)} total matches")
    print(f"Found {len(high_quality)} high-quality matches (similarity >= {min_similarity})")
    
    # Group matches by category
    from collections import defaultdict
    by_category = defaultdict(list)
    for match in high_quality:
        by_category[match['category']].append(match)
    
    print(f"\nMatches by category:")
    for cat, matches in sorted(by_category.items()):
        print(f"  {cat}: {len(matches)} word pairs")
    
    # Combine ALL word pairs from ALL categories into one list
    all_word_pairs = [
        (m['german_word'], m['english_word'], m['similarity'], m['category'])
        for m in high_quality
    ]
    
    print(f"\nTotal words available for sentence generation: {len(all_word_pairs)} pairs")
    print(f"Categories: {', '.join(sorted(by_category.keys()))}")
    
    # Generate multiple sentences using ALL available words
    generated = []
    
    with tqdm(total=max_sentences, desc="Generating sentences") as pbar:
        for attempt in range(max_attempts):
            if len(generated) >= max_sentences:
                break
            
            print(f"\n\nAttempt {attempt + 1}: Generating with all {len(all_word_pairs)} word pairs...")
            
            # Generate matching German and English sentences
            de_sentence, en_sentence, words_used = generate_sentence_pair_with_llm(all_word_pairs, llm_function)
            
            if not de_sentence or not en_sentence:
                print(f"  Failed to generate sentence pair")
                continue
            
            print(f"  DE: {de_sentence}")
            print(f"  EN: {en_sentence}")
            print(f"  Words used: {len(words_used)}")
            
            # Check German grammar
            de_score, de_issues = check_grammar(de_sentence, nlp_de, "de")
            print(f"  DE grammar score: {de_score}")
            
            if de_score < min_grammar_score:
                print(f"  German grammar too low: {de_score}")
                continue
            
            # Check English grammar
            en_score, en_issues = check_grammar(en_sentence, nlp_en, "en")
            print(f"  EN grammar score: {en_score}")
            
            if en_score < min_grammar_score:
                print(f"  English grammar too low: {en_score}")
                continue
            
            # Calculate average similarity of words used
            if words_used:
                avg_similarity = sum(p[2] for p in words_used) / len(words_used)
            else:
                avg_similarity = 0
            
            # Count words by category
            category_counts = defaultdict(int)
            for p in words_used:
                category_counts[p[3]] += 1
            
            # Save successful sentence pair
            generated.append({
                'word_pairs_used': [
                    {
                        'german': p[0],
                        'english': p[1],
                        'similarity': p[2],
                        'category': p[3]
                    }
                    for p in words_used
                ],
                'categories': dict(category_counts),
                'german': {
                    'sentence': de_sentence,
                    'grammar_score': de_score,
                    'issues': de_issues
                },
                'english': {
                    'sentence': en_sentence,
                    'grammar_score': en_score,
                    'issues': en_issues
                },
                'avg_phonetic_similarity': avg_similarity
            })
            
            pbar.update(1)
            print(f"  ✓ Generated sentence pair {len(generated)}/{max_sentences}")
    
    # Save results
    print(f"\nGenerated {len(generated)} sentence pairs")
    print(f"Saving to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(generated, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved {len(generated)} sentences to {output_file}")
    
    return generated


if __name__ == "__main__":
    # Configuration
    matches_file = Path("data/grouped_matches/summary.json")
    output_file = Path("data/gen_sentences/generated_sentences_llm.json")
    
    # LLM Configuration
    llm_backend = "ollama"  # Options: "ollama", "openai", "anthropic"
    
    # Ollama settings (for Docker container with multiple models)
    use_docker_ollama = True
    ollama_container = "open-webui"
    ollama_model = "gpt-oss:20b"  # Options: llama3.1:8b, qwen3:32b, qwen2.5:32b (recommended), deepseek-r1:70b (reasoning), qwen3:235b, gpt-oss:20b (reasoning), gpt-oss:120b
    
    # Generation settings
    min_similarity = 0.85
    min_grammar_score = 75
    max_sentences = 10
    max_attempts = 500
    
    # Set up LLM function
    if llm_backend == "ollama":
        if use_docker_ollama:
            print(f"\nUsing Ollama in Docker container: {ollama_container}")
            print(f"Model: {ollama_model}")
        else:
            print("\nUsing Ollama (local)")
        
        llm_function = lambda prompt: call_ollama_llm(
            prompt, 
            model=ollama_model, 
            use_docker=use_docker_ollama, 
            container_name=ollama_container
        )
    
    else:
        raise ValueError(f"Unknown LLM backend: {llm_backend}")
    
    # Run sentence generation
    print(f"\nConfiguration:")
    print(f"  Input: {matches_file}")
    print(f"  Output: {output_file}")
    print(f"  Min similarity: {min_similarity}")
    print(f"  Min grammar score: {min_grammar_score}")
    print(f"  Max sentences: {max_sentences}")
    print(f"  Max attempts: {max_attempts}")
    
    sentences = build_sentences_with_llm(
        matches_file=matches_file,
        output_file=output_file,
        llm_function=llm_function,
        min_similarity=min_similarity,
        min_grammar_score=min_grammar_score,
        max_sentences=max_sentences,
        max_attempts=max_attempts
    )
    
    print(f"\n✅ Done! Generated {len(sentences)} bilingual sentence pairs")
