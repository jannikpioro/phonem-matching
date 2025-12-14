import spacy
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def categorize_german_words(input_file, output_dir):
    """
    Categorize German words by their part of speech using spaCy.
    
    Args:
        input_file: Path to the input text file containing German words (one per line)
        output_dir: Directory where categorized word files will be saved
    """
    # Load spaCy German model
    print("Loading spaCy German model...")
    nlp = spacy.load("de_core_news_lg")  # Use lg for better accuracy
    
    # Read words from input file
    input_path = Path("../data") / input_file
    if not input_path.exists():
        print(f"Error: Input file {input_file} not found")
        return
    
    print(f"Reading words from {input_file}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    
    # Limit to first 100 words for testing
    # words = words[:100]
    
    print(f"Processing {len(words)} words...")
    
    # Categorize words by POS
    categorized = defaultdict(list)
    
    for word in tqdm(words, desc="Categorizing words", unit="word"):
        # Process the word with spaCy
        doc = nlp(word)
        if doc:
            # Get the POS tag of the first token
            pos = doc[0].pos_
            categorized[pos].append(word)
    
    # Create output directory if it doesn't exist
    output_path = Path("../data/grouped_de") / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save categorized words to separate files
    print(f"\nSaving categorized words to {output_dir}...")
    for pos, word_list in sorted(categorized.items()):
        output_file = output_path / f"{pos.lower()}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for word in sorted(word_list):
                f.write(f"{word}\n")
        print(f"  {pos}: {len(word_list)} words -> {output_file.name}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"Total words processed: {len(words)}")
    print(f"Categories found: {len(categorized)}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    # Configuration
    input_file = "de.txt"
    output_dir = "grouped_de"
    
    categorize_german_words(input_file, output_dir)
