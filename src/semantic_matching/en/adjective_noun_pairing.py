#!/usr/bin/env python3
"""
Pair English adjectives with suitable English nouns based on common collocations.
This script uses spaCy's word vectors and statistical models to find natural
adjective-noun combinations from actual language usage.
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


class AdjectiveNounPairer:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.adjectives: List[str] = []
        self.nouns: List[str] = []
        self.adj_data: Dict = {}
        self.noun_data: Dict = {}
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
        """Load adjective and noun data from grouped_matches."""
        adj_file = self.data_dir / 'grouped_matches' / 'adj.json'
        noun_file = self.data_dir / 'grouped_matches' / 'noun.json'
        
        print(f"Loading adjectives from {adj_file}...")
        with open(adj_file, 'r', encoding='utf-8') as f:
            self.adj_data = json.load(f)
            
        print(f"Loading nouns from {noun_file}...")
        with open(noun_file, 'r', encoding='utf-8') as f:
            self.noun_data = json.load(f)
        
        # Extract unique English adjectives and nouns
        adj_set = set()
        noun_set = set()
        
        for match in self.adj_data['matches']:
            for eng_match in match['matches']:
                word = eng_match['english'].lower().strip()
                if word and not word.endswith("'s"):  # Skip possessives
                    adj_set.add(word)
        
        for match in self.noun_data['matches']:
            for eng_match in match['matches']:
                word = eng_match['english'].lower().strip()
                if word:
                    noun_set.add(word)
        
        self.adjectives = sorted(list(adj_set))
        self.nouns = sorted(list(noun_set))
        
        print(f"Loaded {len(self.adjectives)} unique adjectives")
        print(f"Loaded {len(self.nouns)} unique nouns")
    
    def find_suitable_nouns(self, adjective: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Find suitable nouns for a given adjective using spaCy word vectors."""
        if not self.nlp:
            return []
        
        # Get the adjective token
        adj_doc = self.nlp(adjective)
        if not adj_doc or not adj_doc[0].has_vector:
            # Fallback: return random nouns
            return [(n, 0.5) for n in random.sample(self.nouns, min(top_n, len(self.nouns)))]
        
        # Calculate similarity scores with all available nouns
        noun_scores = []
        for noun in self.nouns:
            noun_doc = self.nlp(noun)
            if noun_doc and noun_doc[0].has_vector:
                similarity = adj_doc[0].similarity(noun_doc[0])
                noun_scores.append((noun, similarity))
        
        # Sort by similarity and return top N
        noun_scores.sort(key=lambda x: x[1], reverse=True)
        return noun_scores[:top_n]
    
    def create_pairings(self, min_phonetic_similarity: float = 0.8, 
                       min_semantic_similarity: float = 0.3,
                       max_pairs_per_adj: int = 5) -> List[Dict]:
        """Create adjective-noun pairings using spaCy semantic similarity."""
        pairings = []
        
        print(f"\nGenerating adjective-noun pairs...")
        print(f"Using minimum phonetic similarity threshold: {min_phonetic_similarity}")
        print(f"Using minimum semantic similarity threshold: {min_semantic_similarity}")
        
        # Pre-filter nouns by phonetic similarity
        high_similarity_nouns = {}
        for match in self.noun_data['matches']:
            for eng_match in match['matches']:
                if eng_match['similarity'] >= min_phonetic_similarity:
                    noun = eng_match['english'].lower().strip()
                    if noun and not noun.endswith("'s"):
                        if noun not in high_similarity_nouns:
                            high_similarity_nouns[noun] = {
                                'german': match['german'],
                                'german_phonetic': match['german_phonetic'],
                                'noun_phonetic': eng_match['english_phonetic'],
                                'phonetic_similarity': eng_match['similarity']
                            }
        
        print(f"Found {len(high_similarity_nouns)} nouns with phonetic similarity >= {min_phonetic_similarity}")
        
        # Count total adjectives to process
        total_to_process = 0
        for match in self.adj_data['matches']:
            for eng_match in match['matches']:
                if eng_match['similarity'] >= min_phonetic_similarity:
                    adjective = eng_match['english'].lower().strip()
                    if not adjective.endswith("'s"):
                        total_to_process += 1
        
        print(f"Processing {total_to_process} adjectives...\n")
        
        # Progress bar
        pbar = tqdm(total=total_to_process, desc="Creating pairs", unit="adj")
        
        for match in self.adj_data['matches']:
            german_adj = match['german']
            german_adj_phonetic = match['german_phonetic']
            
            for eng_match in match['matches']:
                if eng_match['similarity'] < min_phonetic_similarity:
                    continue
                
                adjective = eng_match['english'].lower().strip()
                if adjective.endswith("'s"):
                    continue
                
                # Find semantically similar nouns from high similarity nouns only
                if not self.nlp:
                    pbar.update(1)
                    continue
                    
                adj_doc = self.nlp(adjective)
                if not adj_doc or not adj_doc[0].has_vector:
                    pbar.update(1)
                    continue
                
                noun_scores = []
                for noun, noun_info in high_similarity_nouns.items():
                    noun_doc = self.nlp(noun)
                    if noun_doc and noun_doc[0].has_vector:
                        similarity = adj_doc[0].similarity(noun_doc[0])
                        if similarity >= min_semantic_similarity:
                            noun_scores.append((noun, noun_info, similarity))
                
                # Sort by semantic similarity and take top N
                noun_scores.sort(key=lambda x: x[2], reverse=True)
                selected_nouns = noun_scores[:max_pairs_per_adj]
                
                for noun, noun_info, semantic_score in selected_nouns:
                    # Calculate combined phonetic similarity (average of adjective and noun)
                    combined_phonetic_sim = (eng_match['similarity'] + noun_info['phonetic_similarity']) / 2
                    
                    pairing = {
                        'german_phrase': f"{german_adj} {noun_info['german']}",
                        'german_phonetic': f"{german_adj_phonetic} {noun_info['german_phonetic']}",
                        'english_phrase': f"{adjective} {noun}",
                        'english_phonetic': f"{eng_match['english_phonetic']} {noun_info['noun_phonetic']}",
                        'phonetic_similarity': round(combined_phonetic_sim, 4),
                        'semantic_similarity': round(semantic_score, 4),
                        'adjective_phonetic_similarity': eng_match['similarity'],
                        'noun_phonetic_similarity': noun_info['phonetic_similarity']
                    }
                    pairings.append(pairing)
                
                pbar.update(1)
        
        pbar.close()
        print(f"\n✓ Generated {len(pairings)} adjective-noun pairs")
        return pairings
    
    def save_pairings(self, pairings: List[Dict], output_file: Path):
        """Save pairings to JSON file."""
        output_data = {
            'total_pairs': len(pairings),
            'description': 'English adjective-noun collocations based on phonetically similar German-English adjectives',
            'pairings': pairings
        }
        
        print(f"\nSaving to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(pairings)} pairs")
    
    def print_sample(self, pairings: List[Dict], n: int = 20):
        """Print sample pairings."""
        print(f"\n{'='*80}")
        print(f"Sample Adjective-Noun Pairs (showing {min(n, len(pairings))} of {len(pairings)})")
        print(f"{'='*80}")
        
        sample = random.sample(pairings, min(n, len(pairings)))
        
        for i, pair in enumerate(sample, 1):
            print(f"\n{i}. {pair['english_phrase'].upper()}")
            print(f"   German: {pair['german_phrase']} ({pair['german_phonetic']})")
            print(f"   English: {pair['english_phrase']} ({pair['english_phonetic']})")
            print(f"   Phonetic similarity: {pair['phonetic_similarity']:.2%} (adj: {pair['adjective_phonetic_similarity']:.2%}, noun: {pair['noun_phonetic_similarity']:.2%})")
            print(f"   Semantic similarity: {pair['semantic_similarity']:.2%}")


def main():
    data_dir = Path(__file__).parent.parent / 'data'
    output_dir = data_dir / 'pairing' / 'en'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'adjective_noun_pairs.json'
    
    pairer = AdjectiveNounPairer(data_dir)
    pairer.load_spacy_model()
    pairer.load_data()
    
    # Create pairings with adjustable parameters
    pairings = pairer.create_pairings(
        min_phonetic_similarity=0.80,   # Minimum phonetic similarity (German-English adj)
        min_semantic_similarity=0.3,    # Minimum semantic similarity (English adj-noun)
        max_pairs_per_adj=5             # Maximum noun pairings per adjective
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
