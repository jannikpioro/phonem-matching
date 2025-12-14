#!/usr/bin/env python3
"""
Pair English nouns with suitable English verbs based on common collocations.
This script uses spaCy's word vectors and statistical models to find natural
noun-verb combinations from actual language usage.
"""

import json
import random
import time
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple
import spacy
from spacy.tokens import Doc
from tqdm import tqdm


class NounVerbPairer:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.nouns: List[str] = []
        self.verbs: List[str] = []
        self.noun_data: Dict = {}
        self.verb_data: Dict = {}
        self.nlp = None
        
    def load_spacy_model(self):
        """Load spaCy model with word vectors."""
        print("Loading spaCy model (en_core_web_lg)...")
        try:
            self.nlp = spacy.load("en_core_web_lg")
            print("✓ spaCy model loaded successfully")
        except OSError:
            print("Model not found. Installing en_core_web_lg...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"])
            self.nlp = spacy.load("en_core_web_lg")
        
    def load_data(self):
        """Load noun and verb data from grouped_matches."""
        noun_file = self.data_dir / 'grouped_matches' / 'noun.json'
        verb_file = self.data_dir / 'grouped_matches' / 'verb.json'
        
        print(f"Loading nouns from {noun_file}...")
        with open(noun_file, 'r', encoding='utf-8') as f:
            self.noun_data = json.load(f)
            
        print(f"Loading verbs from {verb_file}...")
        with open(verb_file, 'r', encoding='utf-8') as f:
            self.verb_data = json.load(f)
        
        # Extract unique English nouns and verbs
        noun_set = set()
        verb_set = set()
        
        for match in self.noun_data['matches']:
            for eng_match in match['matches']:
                word = eng_match['english'].lower().strip()
                if word and not word.endswith("'s"):
                    noun_set.add(word)
        
        for match in self.verb_data['matches']:
            for eng_match in match['matches']:
                word = eng_match['english'].lower().strip()
                if word:
                    verb_set.add(word)
        
        self.nouns = sorted(list(noun_set))
        self.verbs = sorted(list(verb_set))
        
        print(f"Loaded {len(self.nouns)} unique nouns")
        print(f"Loaded {len(self.verbs)} unique verbs")
    
    def find_suitable_verbs(self, noun: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Find suitable verbs for a given noun using spaCy word vectors."""
        if not self.nlp:
            return []
        
        # Get the noun token
        noun_doc = self.nlp(noun)
        if not noun_doc or not noun_doc[0].has_vector:
            # Fallback: return random verbs
            return [(v, 0.5) for v in random.sample(self.verbs, min(top_n, len(self.verbs)))]
        
        # Calculate similarity scores with all available verbs
        verb_scores = []
        for verb in self.verbs:
            verb_doc = self.nlp(verb)
            if verb_doc and verb_doc[0].has_vector:
                similarity = noun_doc[0].similarity(verb_doc[0])
                verb_scores.append((verb, similarity))
        
        # Sort by similarity and return top N
        verb_scores.sort(key=lambda x: x[1], reverse=True)
        return verb_scores[:top_n]
    
    def create_pairings(self, min_phonetic_similarity: float = 0.8, 
                       min_semantic_similarity: float = 0.3,
                       max_pairs_per_noun: int = 5) -> List[Dict]:
        """Create noun-verb pairings using spaCy semantic similarity."""
        pairings = []
        
        print(f"\nGenerating noun-verb pairs...")
        print(f"Using minimum phonetic similarity threshold: {min_phonetic_similarity}")
        print(f"Using minimum semantic similarity threshold: {min_semantic_similarity}")
        
        # Pre-filter verbs by phonetic similarity
        high_similarity_verbs = {}
        for match in self.verb_data['matches']:
            for eng_match in match['matches']:
                if eng_match['similarity'] >= min_phonetic_similarity:
                    verb = eng_match['english'].lower().strip()
                    if verb and not verb.endswith("'s"):
                        if verb not in high_similarity_verbs:
                            high_similarity_verbs[verb] = {
                                'german': match['german'],
                                'german_phonetic': match['german_phonetic'],
                                'verb_phonetic': eng_match['english_phonetic'],
                                'phonetic_similarity': eng_match['similarity']
                            }
        
        print(f"Found {len(high_similarity_verbs)} verbs with phonetic similarity >= {min_phonetic_similarity}")
        
        # Count total nouns to process
        total_to_process = 0
        for match in self.noun_data['matches']:
            for eng_match in match['matches']:
                if eng_match['similarity'] >= min_phonetic_similarity:
                    noun = eng_match['english'].lower().strip()
                    if not noun.endswith("'s"):
                        total_to_process += 1
        
        print(f"Processing {total_to_process} nouns...\n")
        
        # Progress bar
        pbar = tqdm(total=total_to_process, desc="Creating pairs", unit="noun")
        
        for match in self.noun_data['matches']:
            german_noun = match['german']
            german_noun_phonetic = match['german_phonetic']
            
            for eng_match in match['matches']:
                if eng_match['similarity'] < min_phonetic_similarity:
                    continue
                
                noun = eng_match['english'].lower().strip()
                if noun.endswith("'s"):
                    continue
                
                # Find semantically similar verbs from high similarity verbs only
                if not self.nlp:
                    pbar.update(1)
                    continue
                    
                noun_doc = self.nlp(noun)
                if not noun_doc or not noun_doc[0].has_vector:
                    pbar.update(1)
                    continue
                
                verb_scores = []
                for verb, verb_info in high_similarity_verbs.items():
                    verb_doc = self.nlp(verb)
                    if verb_doc and verb_doc[0].has_vector:
                        similarity = noun_doc[0].similarity(verb_doc[0])
                        if similarity >= min_semantic_similarity:
                            verb_scores.append((verb, verb_info, similarity))
                
                # Sort by semantic similarity and take top N
                verb_scores.sort(key=lambda x: x[2], reverse=True)
                selected_verbs = verb_scores[:max_pairs_per_noun]
                
                for verb, verb_info, semantic_score in selected_verbs:
                    # Calculate combined phonetic similarity (average of noun and verb)
                    combined_phonetic_sim = (eng_match['similarity'] + verb_info['phonetic_similarity']) / 2
                    
                    pairing = {
                        'german_phrase': f"{german_noun} {verb_info['german']}",
                        'german_phonetic': f"{german_noun_phonetic} {verb_info['german_phonetic']}",
                        'english_phrase': f"{noun} {verb}",
                        'english_phonetic': f"{eng_match['english_phonetic']} {verb_info['verb_phonetic']}",
                        'phonetic_similarity': round(combined_phonetic_sim, 4),
                        'semantic_similarity': round(semantic_score, 4),
                        'noun_phonetic_similarity': eng_match['similarity'],
                        'verb_phonetic_similarity': verb_info['phonetic_similarity']
                    }
                    pairings.append(pairing)
                
                pbar.update(1)
        
        pbar.close()
        print(f"\n✓ Generated {len(pairings)} noun-verb pairs")
        return pairings
    
    def save_pairings(self, pairings: List[Dict], output_file: Path):
        """Save pairings to JSON file."""
        output_data = {
            'total_pairs': len(pairings),
            'description': 'English noun-verb collocations based on phonetically similar German-English nouns',
            'pairings': pairings
        }
        
        print(f"\nSaving to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(pairings)} pairs")
    
    def print_sample(self, pairings: List[Dict], n: int = 20):
        """Print sample pairings."""
        print(f"\n{'='*80}")
        print(f"Sample Noun-Verb Pairs (showing {min(n, len(pairings))} of {len(pairings)})")
        print(f"{'='*80}")
        
        sample = random.sample(pairings, min(n, len(pairings)))
        
        for i, pair in enumerate(sample, 1):
            print(f"\n{i}. {pair['english_phrase'].upper()}")
            print(f"   German: {pair['german_phrase']} ({pair['german_phonetic']})")
            print(f"   English: {pair['english_phrase']} ({pair['english_phonetic']})")
            print(f"   Phonetic similarity: {pair['phonetic_similarity']:.2%} (noun: {pair['noun_phonetic_similarity']:.2%}, verb: {pair['verb_phonetic_similarity']:.2%})")
            print(f"   Semantic similarity: {pair['semantic_similarity']:.2%}")


def main():
    data_dir = Path(__file__).parent.parent / 'data'
    output_dir = data_dir / 'pairing' / 'en'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'noun_verb_pairs.json'
    
    pairer = NounVerbPairer(data_dir)
    pairer.load_spacy_model()
    pairer.load_data()
    
    # Create pairings with adjustable parameters
    pairings = pairer.create_pairings(
        min_phonetic_similarity=0.80,   # Minimum phonetic similarity (German-English noun)
        min_semantic_similarity=0.3,    # Minimum semantic similarity (English noun-verb)
        max_pairs_per_noun=5            # Maximum verb pairings per noun
    )
    
    # Show sample
    pairer.print_sample(pairings, n=30)
    
    # Save results
    pairer.save_pairings(pairings, output_file)
    
    print(f"\n{'='*80}")
    print(f"Complete! Output saved to: {output_file}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
