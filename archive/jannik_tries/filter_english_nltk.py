#!/usr/bin/env python3
"""
Filtert en_US.txt um nur echte englische Wörter zu behalten.
Nutzt NLTK words corpus als Referenz.
"""

import os

def download_nltk_words():
    """Lade NLTK words corpus falls nicht vorhanden."""
    import nltk
    try:
        from nltk.corpus import words
        words.words()
    except LookupError:
        print("Lade NLTK words corpus...")
        nltk.download('words', quiet=True)

def load_nltk_words():
    """Lade alle englischen Wörter aus NLTK."""
    from nltk.corpus import words
    # Alle Wörter kleingeschrieben in ein Set
    return set(w.lower() for w in words.words())

def parse_ipa_file(filepath):
    """Parse IPA file und gib dict {word: ipa} zurück."""
    entries = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '\t' not in line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                word = parts[0].strip()
                ipa = parts[1].strip()
                if word and ipa:
                    entries[word] = ipa
    return entries

def is_real_english_word(word, nltk_words):
    """Prüft ob ein Wort ein echtes englisches Wort ist."""
    word_lower = word.lower()
    
    # Direkt in NLTK words?
    if word_lower in nltk_words:
        return True
    
    # Wörter mit Apostroph oder Bindestrich - Teile prüfen
    if "'" in word_lower or "-" in word_lower:
        parts = word_lower.replace("'", " ").replace("-", " ").split()
        return all(p in nltk_words for p in parts if len(p) > 1)
    
    return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Filtere en_US.txt nach echten englischen Wörtern')
    parser.add_argument('--input', default='data/en_US.txt', help='Input IPA file')
    parser.add_argument('--output', default='data/en_US_real.txt', help='Output filtered file')
    parser.add_argument('--min-length', type=int, default=2, help='Minimale Wortlänge')
    args = parser.parse_args()
    
    # NLTK words laden
    download_nltk_words()
    nltk_words = load_nltk_words()
    print(f"NLTK words corpus geladen: {len(nltk_words):,} Wörter")
    
    # en_US.txt laden
    entries = parse_ipa_file(args.input)
    print(f"Geladen aus {args.input}: {len(entries):,} Einträge")
    
    # Filtern
    filtered = {}
    rejected_examples = []
    
    for word, ipa in entries.items():
        if len(word) < args.min_length:
            continue
            
        if is_real_english_word(word, nltk_words):
            filtered[word] = ipa
        elif len(rejected_examples) < 50:
            rejected_examples.append(word)
    
    print(f"\nNach Filter: {len(filtered):,} echte englische Wörter")
    print(f"Entfernt: {len(entries) - len(filtered):,} Einträge (Namen, Abkürzungen, etc.)")
    
    # Speichern
    with open(args.output, 'w', encoding='utf-8') as f:
        for word, ipa in sorted(filtered.items()):
            f.write(f"{word}\t{ipa}\n")
    
    print(f"\nGespeichert in: {args.output}")
    
    # Beispiele zeigen
    print(f"\nBeispiele entfernter Wörter (erste 30):")
    for w in rejected_examples[:30]:
        print(f"  - {w}")
    
    print(f"\nBeispiele behaltener Wörter (erste 30):")
    for w in list(filtered.keys())[:30]:
        print(f"  + {w}")

if __name__ == '__main__':
    main()
