#!/usr/bin/env python3
"""
Filtert IPA-Dateien um nur echte Wörter zu behalten.
- Englisch: NLTK words corpus
- Deutsch: spaCy de_core_news_sm Vokabular
"""

import os
import argparse

def download_nltk_words():
    """Lade NLTK words corpus falls nicht vorhanden."""
    import nltk
    try:
        from nltk.corpus import words
        words.words()
    except LookupError:
        print("Lade NLTK words corpus...")
        nltk.download('words', quiet=True)

def load_english_words():
    """Lade alle englischen Wörter aus NLTK."""
    download_nltk_words()
    from nltk.corpus import words
    # Alle Wörter kleingeschrieben in ein Set
    word_set = set(w.lower() for w in words.words())
    print(f"NLTK English words: {len(word_set):,} Wörter")
    return word_set

def load_german_words(wordlist_path='data/german_wordlist.txt'):
    """Lade deutsches Vokabular aus Wortliste."""
    word_set = set()
    
    if os.path.exists(wordlist_path):
        with open(wordlist_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                if word and len(word) >= 2:
                    word_set.add(word)
        print(f"German wordlist: {len(word_set):,} Wörter")
    else:
        print(f"WARNUNG: {wordlist_path} nicht gefunden!")
        print("Lade von GitHub...")
        import urllib.request
        url = "https://raw.githubusercontent.com/enz/german-wordlist/master/words"
        urllib.request.urlretrieve(url, wordlist_path)
        return load_german_words(wordlist_path)
    
    return word_set

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

def is_real_word(word, word_set):
    """Prüft ob ein Wort ein echtes Wort ist."""
    word_lower = word.lower().rstrip("'")
    
    # Direkt im Wörterbuch?
    if word_lower in word_set:
        return True
    
    # Wörter mit Sonderzeichen - Hauptteil prüfen
    if "'" in word_lower or "-" in word_lower:
        parts = word_lower.replace("'", " ").replace("-", " ").split()
        main_parts = [p for p in parts if len(p) > 2]
        if main_parts and all(p in word_set for p in main_parts):
            return True
    
    return False

def filter_file(input_path, output_path, word_set, lang_name, min_length=2):
    """Filtert eine IPA-Datei nach echten Wörtern."""
    entries = parse_ipa_file(input_path)
    print(f"\nGeladen aus {input_path}: {len(entries):,} Einträge")
    
    filtered = {}
    rejected_examples = []
    
    for word, ipa in entries.items():
        if len(word) < min_length:
            continue
        
        # Zahlen und reine Sonderzeichen überspringen
        if any(c.isdigit() for c in word):
            continue
            
        if is_real_word(word, word_set):
            filtered[word] = ipa
        elif len(rejected_examples) < 50:
            rejected_examples.append(word)
    
    print(f"Nach Filter ({lang_name}): {len(filtered):,} echte Wörter")
    print(f"Entfernt: {len(entries) - len(filtered):,} Einträge")
    
    # Speichern
    with open(output_path, 'w', encoding='utf-8') as f:
        for word, ipa in sorted(filtered.items()):
            f.write(f"{word}\t{ipa}\n")
    
    print(f"Gespeichert in: {output_path}")
    
    # Beispiele zeigen
    print(f"\nBeispiele entfernter Wörter (erste 20):")
    for w in rejected_examples[:20]:
        print(f"  - {w}")
    
    return filtered

def main():
    parser = argparse.ArgumentParser(description='Filtere IPA-Dateien nach echten Wörtern')
    parser.add_argument('--english', default='data/en_US.txt', help='Englische IPA-Datei')
    parser.add_argument('--german', default='data/de (1).txt', help='Deutsche IPA-Datei')
    parser.add_argument('--english-out', default='data/en_US_real.txt', help='Gefilterte englische Datei')
    parser.add_argument('--german-out', default='data/de_real.txt', help='Gefilterte deutsche Datei')
    parser.add_argument('--min-length', type=int, default=3, help='Minimale Wortlänge')
    parser.add_argument('--only', choices=['en', 'de', 'both'], default='both', help='Welche Sprache filtern')
    args = parser.parse_args()
    
    if args.only in ['en', 'both']:
        print("=" * 60)
        print("ENGLISCH FILTERN")
        print("=" * 60)
        english_words = load_english_words()
        filter_file(args.english, args.english_out, english_words, "Englisch", args.min_length)
    
    if args.only in ['de', 'both']:
        print("\n" + "=" * 60)
        print("DEUTSCH FILTERN")
        print("=" * 60)
        german_words = load_german_words()
        filter_file(args.german, args.german_out, german_words, "Deutsch", args.min_length)
    
    print("\n" + "=" * 60)
    print("FERTIG!")
    print("=" * 60)

if __name__ == '__main__':
    main()
