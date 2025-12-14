from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import json
from Levenshtein import distance as levenshtein_distance

def load_words_from_file(filepath):
    """Load words and their phonetic pronunciations from a text file."""
    word_phonetics = []
    if Path(filepath).exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Split by tab to get word and phonetic(s)
                parts = line.split('\t')
                if len(parts) >= 2:
                    word = parts[0]
                    # Take the first phonetic pronunciation if multiple are provided
                    phonetics = parts[1].split(', ')[0].strip()
                    # Remove the slashes from phonetic notation
                    phonetics = phonetics.strip('/').strip()
                    if phonetics:
                        word_phonetics.append({
                            'word': word,
                            'phonetic': phonetics
                        })
    return word_phonetics

def phonetic_similarity(phonetic1, phonetic2):
    """
    Calculate phonetic similarity between two phonetic transcriptions.
    Returns a similarity score (0-1, where 1 is identical).
    """
    # Calculate Levenshtein distance
    max_len = max(len(phonetic1), len(phonetic2))
    if max_len == 0:
        return 0.0
    
    # Convert distance to similarity score (0-1)
    similarity = 1 - (levenshtein_distance(phonetic1, phonetic2) / max_len)
    return similarity

def match_words_by_group(de_dir, en_dir, output_file, min_similarity=0.5, test_group=None, log_file="phonetic_matching.log"):
    """
    Match German and English words by POS group and rank by phonetic similarity.
    
    Args:
        de_dir: Directory containing German grouped word files
        en_dir: Directory containing English grouped word files
        output_file: Output JSON file for matched pairs
        min_similarity: Minimum similarity threshold (0-1)
        test_group: Optional - test with only one POS group (e.g., 'noun', 'verb')
        log_file: Path to log file for detailed comparison records
    """
    # Exclude POS categories that don't make sense for phonetic matching
    excluded_pos = {'punct', 'sym', 'x', 'space'}
    
    # Clear and initialize log file in logs directory
    log_path = Path("../logs") / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("Phonetic Matching Log\n")
        f.write("=" * 80 + "\n\n")
    
    print("Loading word data...")
    
    de_path = Path(de_dir)
    en_path = Path(en_dir)
    
    if not de_path.exists():
        print(f"Error: German directory {de_dir} not found")
        return
    if not en_path.exists():
        print(f"Error: English directory {en_dir} not found")
        return
    
    # Get all POS categories that exist in both languages
    de_files = {f.stem: f for f in de_path.glob('*.txt')}
    en_files = {f.stem: f for f in en_path.glob('*.txt')}
    common_pos = (set(de_files.keys()) & set(en_files.keys())) - excluded_pos
    
    # If test_group is specified, only process that group
    if test_group:
        if test_group in common_pos:
            common_pos = {test_group}
            print(f"\nTesting with single group: {test_group}")
        else:
            print(f"\nError: Test group '{test_group}' not found in common categories")
            print(f"Available groups: {sorted(common_pos)}")
            return
    
    print(f"\nFound {len(common_pos)} common POS categories: {sorted(common_pos)}")
    print(f"Excluded categories: {sorted(excluded_pos)}")
    
    # Create output directory
    output_dir = Path("../data/grouped_matches")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_matches = {}
    
    # Process each POS category
    for pos in sorted(common_pos):
        print(f"\n{'='*60}")
        print(f"Processing {pos.upper()} category...")
        print(f"{'='*60}")
        
        # Load words for this category
        de_words = load_words_from_file(de_files[pos])
        en_words = load_words_from_file(en_files[pos])
        
        print(f"German words: {len(de_words)}, English words: {len(en_words)}")
        
        if not de_words or not en_words:
            print(f"Skipping {pos} - no words in one or both languages")
            continue
        
        category_matches = []
        
        # Open log file in append mode for this category
        with open(log_path, 'a', encoding='utf-8') as log:
            log.write(f"\n{'='*80}\n")
            log.write(f"Category: {pos.upper()}\n")
            log.write(f"{'='*80}\n\n")
        
            # Compare each German word with each English word
            for de_item in tqdm(de_words, desc=f"Matching {pos}", unit="word"):
                word_matches = []
                
                for en_item in en_words:
                    similarity = phonetic_similarity(de_item['phonetic'], en_item['phonetic'])
                    
                    word_matches.append({
                        'english': en_item['word'],
                        'english_phonetic': en_item['phonetic'],
                        'similarity': round(similarity, 4)
                    })
            
                # Sort by similarity to find the best match
                word_matches.sort(key=lambda x: x['similarity'], reverse=True)
                
                # Log the top match regardless of threshold
                if word_matches:
                    best_match = word_matches[0]
                    with open(log_path, 'a', encoding='utf-8') as log_inner:
                        log_inner.write(
                            f"{de_item['word']} [{de_item['phonetic']}] → "
                            f"{best_match['english']} [{best_match['english_phonetic']}]: "
                            f"{best_match['similarity']:.4f}\n"
                        )
                    
                    # Only include if best match is above threshold
                    if best_match['similarity'] >= min_similarity:
                        category_matches.append({
                            'german': de_item['word'],
                            'german_phonetic': de_item['phonetic'],
                            'matches': [best_match]
                        })
        
        # Sort category matches by best similarity score
        category_matches.sort(
            key=lambda x: x['matches'][0]['similarity'] if x['matches'] else 0,
            reverse=True
        )
        
        all_matches[pos] = {
            'total_pairs': len(category_matches),
            'matches': category_matches
        }
        
        # Save this category immediately
        category_file = output_dir / f"{pos}.json"
        with open(category_file, 'w', encoding='utf-8') as f:
            json.dump(all_matches[pos], f, ensure_ascii=False, indent=2)
        print(f"Saved {len(category_matches)} matches to {category_file.name}")
        
        print(f"Found {len(category_matches)} German words with matches")
        if category_matches:
            best = category_matches[0]
            print(f"Best match: {best['german']} → {best['matches'][0]['english']} "
                  f"(similarity: {best['matches'][0]['similarity']})")
    
    # Save combined summary file
    print(f"\n{'='*60}")
    
    print(f"Saving combined summary to {output_dir}...")
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_matches, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print(f"\nSummary:")
    total_matches = sum(cat['total_pairs'] for cat in all_matches.values())
    print(f"Categories processed: {len(all_matches)}")
    print(f"Total German words with matches: {total_matches}")
    print(f"Output directory: {output_dir}")
    print(f"Summary file: {summary_file}")
    print(f"Log file: {log_path}")

if __name__ == "__main__":
    # Configuration
    de_dir = "../data/grouped_de"
    en_dir = "../data/grouped_en"
    min_similarity = 0.6  # Minimum similarity threshold (0-1)
    test_group = None  # Test with a single group (set to None to process all)
    
    match_words_by_group(de_dir, en_dir, None, min_similarity, test_group)
