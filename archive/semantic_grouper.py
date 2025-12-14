#!/usr/bin/env python3
"""
Group phonetically matched words by semantic similarity using word embeddings.
Creates thematic clusters of words that are likely to appear together in sentences.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from collections import defaultdict
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Installing sentence-transformers...")
    import subprocess
    subprocess.run(["pip", "install", "sentence-transformers"], check=True)
    from sentence_transformers import SentenceTransformer


def load_word_matches(matches_dir, pos_types=['noun', 'propn', 'verb', 'adj']):
    """
    Load word matches from JSON files.
    
    Args:
        matches_dir: Path to directory containing match JSON files
        pos_types: List of POS types to include
    
    Returns:
        Dictionary of {category: [word_pairs]}
    """
    matches_by_pos = {}
    
    for pos in pos_types:
        file_path = matches_dir / f"{pos}.json"
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping")
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        word_pairs = []
        if 'matches' in data:
            for german_entry in data['matches']:
                german_word = german_entry['german']
                for match in german_entry['matches']:
                    word_pairs.append({
                        'german': german_word,
                        'english': match['english'],
                        'similarity': match['similarity'],
                        'pos': pos
                    })
        
        matches_by_pos[pos] = word_pairs
        print(f"Loaded {len(word_pairs)} word pairs for {pos}")
    
    return matches_by_pos


def cluster_words_by_semantics(word_pairs, model, n_clusters=10):
    """
    Cluster words using K-means on sentence transformer embeddings.
    
    Args:
        word_pairs: List of word pair dictionaries
        model: SentenceTransformer model
        n_clusters: Number of semantic clusters to create
    
    Returns:
        Dictionary of {cluster_id: [word_pairs]}
    """
    print(f"Computing embeddings for {len(word_pairs)} word pairs...")
    
    # Prepare words for embedding
    texts = []
    valid_pairs = []
    
    for pair in word_pairs:
        # Combine German and English word for context
        text = f"{pair['german']} {pair['english']}"
        texts.append(text)
        valid_pairs.append(pair)
    
    if not texts:
        print("Warning: No valid word pairs found!")
        return {}
    
    # Compute embeddings using sentence transformer
    print("Generating embeddings with SentenceTransformer...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    
    print(f"Got embeddings for {len(valid_pairs)} word pairs")
    
    # Determine optimal number of clusters (don't exceed number of samples)
    n_clusters = min(n_clusters, len(valid_pairs))
    
    # Cluster using K-means
    print(f"Clustering into {n_clusters} semantic groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Group word pairs by cluster
    clusters = defaultdict(list)
    for pair, label in zip(valid_pairs, cluster_labels):
        clusters[int(label)].append(pair)
    
    return clusters


def analyze_clusters(clusters, top_n=5):
    """
    Print analysis of each cluster.
    
    Args:
        clusters: Dictionary of {cluster_id: [word_pairs]}
        top_n: Number of example words to show per cluster
    """
    print("\n" + "="*80)
    print("SEMANTIC CLUSTER ANALYSIS")
    print("="*80)
    
    for cluster_id, pairs in sorted(clusters.items()):
        print(f"\nCluster {cluster_id}: {len(pairs)} word pairs")
        print("-" * 40)
        
        # Show top N examples
        examples = pairs[:top_n]
        for pair in examples:
            print(f"  {pair['german']:20} → {pair['english']:20} (sim: {pair['similarity']:.2f}, {pair['pos']})")
        
        if len(pairs) > top_n:
            print(f"  ... and {len(pairs) - top_n} more")


def save_semantic_groups(clusters, output_file):
    """
    Save semantic groups to JSON file.
    
    Args:
        clusters: Dictionary of {cluster_id: [word_pairs]}
        output_file: Path to output JSON file
    """
    # Convert to serializable format
    output_data = {}
    
    for cluster_id, pairs in clusters.items():
        cluster_key = f"cluster_{cluster_id}"
        output_data[cluster_key] = {
            'size': len(pairs),
            'word_pairs': [
                {
                    'german': pair['german'],
                    'english': pair['english'],
                    'similarity': pair['similarity'],
                    'pos': pair['pos']
                }
                for pair in pairs
            ]
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Saved semantic groups to {output_file}")


def main():
    # Configuration
    matches_dir = Path("data/grouped_matches")
    output_file = Path("data/semantic_groups.json")
    
    # POS types to include
    pos_types = ['noun', 'propn', 'verb', 'adj']
    
    # Number of semantic clusters
    n_clusters = 15
    
    # Minimum phonetic similarity threshold (0.0 to 1.0)
    # Higher values = only very similar sounding words
    # Lower values = include more word pairs
    min_similarity = 0.85  # Options: 0.7 (loose), 0.8 (medium), 0.85 (strict), 0.9 (very strict), 0.95 (extremely strict)
    
    # Sentence Transformer model to use
    # Options: 'all-MiniLM-L6-v2' (fast, good), 'all-mpnet-base-v2' (slower, better), 'paraphrase-multilingual-MiniLM-L12-v2' (multilingual)
    transformer_model = 'paraphrase-multilingual-MiniLM-L12-v2'
    
    print("\n" + "="*80)
    print("SEMANTIC WORD GROUPER (SentenceTransformer)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Input directory: {matches_dir}")
    print(f"  Output file: {output_file}")
    print(f"  POS types: {', '.join(pos_types)}")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Min similarity: {min_similarity}")
    print(f"  Transformer model: {transformer_model}")
    
    # Load SentenceTransformer model
    print("\nLoading SentenceTransformer model...")
    model = SentenceTransformer(transformer_model)
    print(f"✓ Loaded {transformer_model}")
    
    # Load word matches
    print("\nLoading word matches...")
    matches_by_pos = load_word_matches(matches_dir, pos_types)
    
    # Combine all word pairs and filter by similarity
    all_pairs = []
    for pos, pairs in matches_by_pos.items():
        filtered = [p for p in pairs if p['similarity'] >= min_similarity]
        all_pairs.extend(filtered)
        print(f"  {pos}: {len(filtered)} pairs (after filtering)")
    
    print(f"\nTotal word pairs: {len(all_pairs)}")
    
    # Cluster words by semantic similarity
    clusters = cluster_words_by_semantics(all_pairs, model, n_clusters)
    
    # Analyze clusters
    analyze_clusters(clusters, top_n=10)
    
    # Save results
    save_semantic_groups(clusters, output_file)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Created {len(clusters)} semantic clusters")
    print(f"Total word pairs: {sum(len(pairs) for pairs in clusters.values())}")
    print(f"Average cluster size: {sum(len(pairs) for pairs in clusters.values()) / len(clusters):.1f}")
    print("\nYou can now use these semantic groups to generate thematically coherent sentences!")


if __name__ == "__main__":
    main()
