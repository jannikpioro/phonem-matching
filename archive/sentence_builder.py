from pathlib import Path
import json
import random
from collections import defaultdict
from itertools import product
import spacy
from tqdm import tqdm

def load_matched_words(matches_dir):
    """
    Load matched words from all POS group files.
    Returns a dict organized by POS category.
    """
    matches_path = Path(matches_dir)
    if not matches_path.exists():
        print(f"Error: Matches directory {matches_dir} not found")
        return {}
    
    word_groups = {}
    
    # Load each POS category file
    for json_file in matches_path.glob('*.json'):
        if json_file.stem == 'summary':
            continue
            
        pos = json_file.stem
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract word pairs
        pairs = []
        for match in data.get('matches', []):
            german_word = match['german']
            german_phonetic = match['german_phonetic']
            
            # Get the best match (first in list)
            if match['matches']:
                english_word = match['matches'][0]['english']
                english_phonetic = match['matches'][0]['english_phonetic']
                similarity = match['matches'][0]['similarity']
                
                pairs.append({
                    'de': german_word,
                    'de_ipa': german_phonetic,
                    'en': english_word,
                    'en_ipa': english_phonetic,
                    'similarity': similarity
                })
        
        word_groups[pos] = pairs
        print(f"Loaded {len(pairs)} {pos} pairs")
    
    return word_groups

def check_grammar(sentence, nlp):
    """
    Prüft die grammatikalische Korrektheit eines Satzes mit spaCy.
    Gibt einen Score zurück (höher = besser).
    """
    doc = nlp(sentence)
    
    score = 100
    issues = []
    
    # Prüfe auf grundlegende Satzstruktur
    has_verb = any(token.pos_ == 'VERB' for token in doc)
    has_noun = any(token.pos_ in ['NOUN', 'PROPN', 'PRON'] for token in doc)
    
    if not has_verb:
        score -= 30
        issues.append("no_verb")
    if not has_noun:
        score -= 20
        issues.append("no_noun")
    
    # Prüfe auf vollständige Dependencies
    root_count = sum(1 for token in doc if token.dep_ == 'ROOT')
    if root_count != 1:
        score -= 15
        issues.append(f"multiple_roots({root_count})")
    
    # Prüfe auf unvollständige Dependencies
    incomplete_deps = sum(1 for token in doc if token.dep_ == 'dep')
    if incomplete_deps > 0:
        score -= incomplete_deps * 5
        issues.append(f"incomplete_deps({incomplete_deps})")
    
    # Bonus für Subjekt-Verb Übereinstimmung
    subjects = [token for token in doc if 'subj' in token.dep_]
    if subjects:
        score += 5
    
    # Bonus für Artikel vor Nomen
    for i, token in enumerate(doc):
        if token.pos_ == 'DET' and i+1 < len(doc):
            if doc[i+1].pos_ in ['NOUN', 'ADJ']:
                score += 3
    
    # Cap score at 100
    score = min(score, 100)
    
    return score, issues

def build_sentences_iterative(word_groups, nlp_de, nlp_en, min_similarity=0.7, 
                              min_grammar_score=70, max_sentences=1000, 
                              max_per_pattern=100):
    """
    Iteriert durch mögliche Satzkombinationen und prüft grammatikalische Korrektheit.
    """
    sentences = []
    
    # Definiere Satzmuster
    patterns = [
        # Einfache Muster
        ['det', 'noun', 'verb'],
        ['det', 'adj', 'noun'],
        ['noun', 'verb'],
        ['verb', 'det', 'noun'],
        ['adj', 'noun', 'verb'],
        ['pron', 'verb'],
        ['pron', 'verb', 'noun'],
        
        # Mittlere Komplexität
        ['det', 'noun', 'verb', 'adv'],
        ['det', 'adj', 'noun', 'verb'],
        ['noun', 'verb', 'det', 'noun'],
        ['det', 'noun', 'verb', 'det', 'noun'],
        ['pron', 'verb', 'det', 'adj', 'noun'],
        
        # Mit Präpositionen
        ['det', 'noun', 'verb', 'adp', 'noun'],
        ['noun', 'verb', 'adp', 'det', 'noun'],
        ['det', 'adj', 'noun', 'verb', 'adp', 'noun'],
        ['pron', 'verb', 'adp', 'det', 'noun'],
    ]
    
    print(f"\nIterating through sentence patterns...")
    print(f"Grammar check thresholds: DE >= {min_grammar_score}, EN >= {min_grammar_score}")
    
    for pattern in patterns:
        print(f"\nPattern: {' '.join(pattern)}")
        
        # Prüfe ob alle POS Kategorien verfügbar sind
        if not all(pos in word_groups and word_groups[pos] for pos in pattern):
            print(f"  Skipped (missing categories)")
            continue
        
        # Filtere Wörter nach Mindest-Ähnlichkeit
        filtered_groups = {}
        for pos in pattern:
            filtered_groups[pos] = [
                p for p in word_groups[pos] 
                if p['similarity'] >= min_similarity
            ]
            if not filtered_groups[pos]:
                print(f"  Skipped (no words with similarity >= {min_similarity})")
                break
        
        if len(filtered_groups) != len(pattern):
            continue
        
        # Berechne mögliche Kombinationen
        total_combinations = 1
        for pos in pattern:
            total_combinations *= len(filtered_groups[pos])
        
        print(f"  Total combinations: {total_combinations:,}")
        
        # Begrenze Kombinationen für große Muster
        if total_combinations > max_per_pattern * 10:
            print(f"  Sampling {max_per_pattern * 10} random combinations...")
            # Sample zufällig
            pattern_sentences = 0
            attempts = 0
            max_attempts = max_per_pattern * 10
            
            with tqdm(total=max_attempts, desc=f"  Checking", unit="comb") as pbar:
                while pattern_sentences < max_per_pattern and attempts < max_attempts:
                    attempts += 1
                    pbar.update(1)
                    
                    # Zufällige Auswahl
                    combination = [random.choice(filtered_groups[pos]) for pos in pattern]
                    
                    de_words = [pair['de'] for pair in combination]
                    en_words = [pair['en'] for pair in combination]
                    avg_similarity = sum(pair['similarity'] for pair in combination) / len(pattern)
                    
                    de_sentence = ' '.join(de_words)
                    en_sentence = ' '.join(en_words)
                    
                    # Prüfe Grammatik
                    de_score, de_issues = check_grammar(de_sentence, nlp_de)
                    en_score, en_issues = check_grammar(en_sentence, nlp_en)
                    
                    if de_score >= min_grammar_score and en_score >= min_grammar_score:
                        # Vermeide Duplikate
                        if (de_sentence, en_sentence) not in [(s['de'], s['en']) for s in sentences]:
                            sentences.append({
                                'de': de_sentence,
                                'en': en_sentence,
                                'pattern': ' '.join(pattern),
                                'avg_similarity': round(avg_similarity, 4),
                                'de_grammar_score': de_score,
                                'en_grammar_score': en_score,
                                'length': len(pattern)
                            })
                            pattern_sentences += 1
                            pbar.set_postfix({'valid': pattern_sentences})
            
            print(f"  Generated: {pattern_sentences} valid sentences")
        else:
            # Iteriere durch alle Kombinationen
            valid_count = 0
            checked_count = 0
            
            # Erstelle alle Kombinationen als Liste für tqdm
            all_combinations = list(product(*[filtered_groups[pos] for pos in pattern]))
            total = min(len(all_combinations), max_per_pattern * 2)  # Limitiere falls nötig
            
            with tqdm(total=total, desc=f"  Checking", unit="comb") as pbar:
                for combination in all_combinations:
                    if valid_count >= max_per_pattern:
                        break
                        
                    checked_count += 1
                    pbar.update(1)
                    
                    de_words = [pair['de'] for pair in combination]
                    en_words = [pair['en'] for pair in combination]
                    avg_similarity = sum(pair['similarity'] for pair in combination) / len(pattern)
                    
                    de_sentence = ' '.join(de_words)
                    en_sentence = ' '.join(en_words)
                    
                    # Prüfe Grammatik
                    de_score, de_issues = check_grammar(de_sentence, nlp_de)
                    en_score, en_issues = check_grammar(en_sentence, nlp_en)
                    
                    if de_score >= min_grammar_score and en_score >= min_grammar_score:
                        # Vermeide Duplikate
                        if (de_sentence, en_sentence) not in [(s['de'], s['en']) for s in sentences]:
                            sentences.append({
                                'de': de_sentence,
                                'en': en_sentence,
                                'pattern': ' '.join(pattern),
                                'avg_similarity': round(avg_similarity, 4),
                                'de_grammar_score': de_score,
                                'en_grammar_score': en_score,
                                'de_issues': de_issues if de_issues else [],
                                'en_issues': en_issues if en_issues else [],
                                'length': len(pattern)
                            })
                            valid_count += 1
                            pbar.set_postfix({'valid': valid_count})
                    
                    if checked_count >= total:
                        break
            
            print(f"  Checked: {checked_count:,}, Generated: {valid_count} valid sentences")
        
        if len(sentences) >= max_sentences:
            print(f"\nReached maximum of {max_sentences} sentences, stopping.")
            break
    
    # Sortiere nach durchschnittlicher Ähnlichkeit
    sentences.sort(key=lambda x: (x['avg_similarity'], x['de_grammar_score'], x['en_grammar_score']), 
                   reverse=True)
    
    return sentences

def build_simple_sentences(word_groups, min_similarity=0.7, max_sentences=100):
    """
    Build simple sentences using matched words.
    Tries basic sentence patterns like:
    - DET + NOUN + VERB
    - DET + ADJ + NOUN
    - NOUN + VERB + NOUN
    etc.
    """
    sentences = []
    
    # Define sentence patterns (POS sequences)
    patterns = [
        # Simple patterns
        ['det', 'noun', 'verb'],
        ['det', 'adj', 'noun'],
        ['noun', 'verb'],
        ['verb', 'det', 'noun'],
        ['adj', 'noun', 'verb'],
        
        # Medium complexity
        ['det', 'noun', 'verb', 'adv'],
        ['det', 'adj', 'noun', 'verb'],
        ['noun', 'verb', 'det', 'noun'],
        ['det', 'noun', 'verb', 'det', 'noun'],
        
        # With prepositions
        ['det', 'noun', 'verb', 'adp', 'noun'],
        ['noun', 'verb', 'adp', 'det', 'noun'],
        ['det', 'adj', 'noun', 'verb', 'adp', 'noun'],
    ]
    
    for pattern in patterns:
        # Check if all POS categories in pattern are available
        if not all(pos in word_groups and word_groups[pos] for pos in pattern):
            continue
        
        # Generate multiple sentences for this pattern
        attempts = 0
        pattern_sentences = 0
        max_attempts = 1000
        target_per_pattern = max(10, max_sentences // len(patterns))
        
        while pattern_sentences < target_per_pattern and attempts < max_attempts:
            attempts += 1
            
            # Randomly select words for each POS in the pattern
            de_words = []
            en_words = []
            avg_similarity = 0
            
            valid = True
            for pos in pattern:
                # Filter by minimum similarity
                candidates = [p for p in word_groups[pos] if p['similarity'] >= min_similarity]
                if not candidates:
                    valid = False
                    break
                
                pair = random.choice(candidates)
                de_words.append(pair['de'])
                en_words.append(pair['en'])
                avg_similarity += pair['similarity']
            
            if not valid:
                continue
            
            avg_similarity /= len(pattern)
            
            # Build sentences
            de_sentence = ' '.join(de_words)
            en_sentence = ' '.join(en_words)
            
            # Avoid duplicates
            sentence_key = (de_sentence, en_sentence)
            if sentence_key in [(s['de'], s['en']) for s in sentences]:
                continue
            
            sentences.append({
                'de': de_sentence,
                'en': en_sentence,
                'pattern': ' '.join(pattern),
                'avg_similarity': round(avg_similarity, 4),
                'length': len(pattern)
            })
            
            pattern_sentences += 1
            
            if len(sentences) >= max_sentences:
                break
        
        if len(sentences) >= max_sentences:
            break
    
    # Sort by average similarity
    sentences.sort(key=lambda x: x['avg_similarity'], reverse=True)
    
    return sentences[:max_sentences]

def save_sentences(sentences, output_file):
    """Save generated sentences to JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'total_sentences': len(sentences),
        'sentences': sentences
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved {len(sentences)} sentences to {output_file}")

def display_sentences(sentences, count=20):
    """Display generated sentences."""
    print(f"\n{'='*80}")
    print(f"GENERATED SENTENCES (Top {count})")
    print(f"{'='*80}\n")
    
    for i, s in enumerate(sentences[:count], 1):
        de_score = s.get('de_grammar_score', 'N/A')
        en_score = s.get('en_grammar_score', 'N/A')
        print(f"{i}. [{s['avg_similarity']:.2%}] Pattern: {s['pattern']} | Grammar: DE={de_score}, EN={en_score}")
        print(f"   DE: {s['de']}")
        print(f"   EN: {s['en']}")
        if s.get('de_issues'):
            print(f"   DE issues: {', '.join(s['de_issues'])}")
        if s.get('en_issues'):
            print(f"   EN issues: {', '.join(s['en_issues'])}")
        print()

def main():
    # Configuration
    matches_dir = "data/grouped_matches"
    output_file = "generated_sentences.json"
    min_similarity = 0.85  # Minimum word similarity to use
    min_grammar_score = 70  # Minimum grammar score (0-100)
    max_sentences = 5000   # Maximum sentences to generate
    max_per_pattern = 50  # Maximum sentences per pattern
    use_iterative = True  # Use iterative method with grammar checking
    
    print("Loading spaCy models...")
    nlp_de = spacy.load("de_dep_news_trf")
    nlp_en = spacy.load("en_core_web_trf")
    
    print("Loading matched words...")
    word_groups = load_matched_words(matches_dir)
    
    if not word_groups:
        print("No word groups loaded. Please run phonem_matching.py first.")
        return
    
    print(f"\nLoaded {len(word_groups)} POS categories")
    print(f"Categories: {sorted(word_groups.keys())}")
    
    if use_iterative:
        print(f"\nBuilding sentences with grammar checking...")
        print(f"Min similarity: {min_similarity}")
        print(f"Min grammar score: {min_grammar_score}")
        sentences = build_sentences_iterative(
            word_groups, nlp_de, nlp_en, 
            min_similarity, min_grammar_score, 
            max_sentences, max_per_pattern
        )
    else:
        print(f"\nBuilding sentences (random sampling, min similarity: {min_similarity})...")
        sentences = build_simple_sentences(word_groups, min_similarity, max_sentences)
    
    print(f"\nGenerated {len(sentences)} sentences")
    
    if sentences:
        save_sentences(sentences, output_file)
        display_sentences(sentences, count=30)
        
        # Statistics
        print(f"\n{'='*80}")
        print("STATISTICS")
        print(f"{'='*80}")
        print(f"Total sentences: {len(sentences)}")
        
        if sentences:
            avg_sim = sum(s['avg_similarity'] for s in sentences) / len(sentences)
            print(f"Average similarity: {avg_sim:.2%}")
            
            if 'de_grammar_score' in sentences[0]:
                avg_de_grammar = sum(s.get('de_grammar_score', 0) for s in sentences) / len(sentences)
                avg_en_grammar = sum(s.get('en_grammar_score', 0) for s in sentences) / len(sentences)
                print(f"Average DE grammar score: {avg_de_grammar:.1f}/100")
                print(f"Average EN grammar score: {avg_en_grammar:.1f}/100")
            
            pattern_counts = defaultdict(int)
            for s in sentences:
                pattern_counts[s['pattern']] += 1
            
            print(f"\nSentences by pattern:")
            for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {pattern}: {count}")
    else:
        print("No sentences generated. Try lowering min_similarity or min_grammar_score threshold.")

if __name__ == "__main__":
    main()
