#!/usr/bin/env python3
"""
IPA Phonetic Similarity Finder

Findet ähnlich klingende Wortpaare zwischen Deutsch und Englisch
basierend auf IPA-Transkriptionen.
"""

import re
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import heapq
from dataclasses import dataclass
from functools import lru_cache
import multiprocessing as mp
from itertools import islice
import json


@dataclass
class Word:
    """Repräsentiert ein Wort mit seiner IPA-Transkription."""
    text: str
    ipa: str
    ipa_clean: str  # IPA ohne diakritische Zeichen und Sonderzeichen


def clean_ipa(ipa: str) -> str:
    """
    Bereinigt IPA-String für Vergleiche.
    Entfernt Betonungszeichen, Silbengrenzen und normalisiert Zeichen.
    """
    # Entferne Slashes und eckige Klammern
    ipa = ipa.strip('/[]')
    
    # Entferne Betonungszeichen und andere diakritische Zeichen
    remove_chars = 'ˈˌ.ˑːʰʷʲˠˤ̩̯̃̈̊̚'
    for char in remove_chars:
        ipa = ipa.replace(char, '')
    
    # Normalisiere einige IPA-Zeichen für besseren Vergleich
    # (z.B. ähnliche Laute zusammenfassen)
    normalizations = {
        'ɫ': 'l',  # Dark L zu normalem L
        'ɾ': 'r',  # Flap zu R
        'ɹ': 'r',  # Approximant zu R
        'ɝ': 'ɐ',  # Rhotischer Vokal
        'ɚ': 'ə',  # Rhotischer Schwa
        'ʁ': 'r',  # Uvular R zu R
        'ɐ': 'a',  # Near-open central zu a
        'æ': 'e',  # Near-open front zu e
        'ɛ': 'e',  # Open-mid front zu e
        'ɪ': 'i',  # Near-close front zu i
        'ʊ': 'u',  # Near-close back zu u
        'ɔ': 'o',  # Open-mid back zu o
        'ʌ': 'a',  # Open-mid back zu a
        'ɑ': 'a',  # Open back zu a
        'ŋ': 'n',  # Velar nasal zu n
        'θ': 's',  # Dental fricative zu s
        'ð': 'd',  # Voiced dental fricative zu d
        'ʃ': 'ʃ',  # Behalte sh
        'ʒ': 'ʒ',  # Behalte zh
        'tʃ': 'tʃ', # Behalte tsch
        'dʒ': 'dʒ', # Behalte dsch
        'ç': 'ʃ',  # Palataler Frikativ zu sch
        'x': 'k',  # Velarer Frikativ zu k
        'ʔ': '',   # Glottal stop entfernen
    }
    
    for old, new in normalizations.items():
        ipa = ipa.replace(old, new)
    
    return ipa


def parse_ipa_file(filepath: str, max_words: Optional[int] = None, 
                   sample_random: bool = False, min_ipa_length: int = 3,
                   max_ipa_length: int = 15) -> List[Word]:
    """
    Parst eine IPA-Wortliste.
    Format: wort\t/ipa/, /ipa2/
    
    Args:
        filepath: Pfad zur Datei
        max_words: Maximale Anzahl Wörter (None = alle)
        sample_random: Zufälliges Sampling statt erste N Wörter
        min_ipa_length: Minimale IPA-Länge (filtert zu kurze Wörter)
        max_ipa_length: Maximale IPA-Länge (filtert zu lange/komplexe Wörter)
    """
    import random
    
    all_words = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '\t' not in line:
                continue
            
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            
            text = parts[0].strip()
            ipa_raw = parts[1].strip()
            
            # Überspringe Einträge mit Zahlen/Sonderzeichen am Anfang
            if text and text[0].isdigit():
                continue
            
            # Nehme nur die erste Aussprache wenn mehrere vorhanden
            if ',' in ipa_raw:
                ipa_raw = ipa_raw.split(',')[0].strip()
            
            # Entferne Slashes
            ipa = ipa_raw.strip('/')
            ipa_clean = clean_ipa(ipa)
            
            # Filtere nach IPA-Länge
            if min_ipa_length <= len(ipa_clean) <= max_ipa_length:
                all_words.append(Word(text=text, ipa=ipa, ipa_clean=ipa_clean))
    
    # Sampling
    if max_words and len(all_words) > max_words:
        if sample_random:
            random.seed(42)  # Reproduzierbar
            words = random.sample(all_words, max_words)
        else:
            words = all_words[:max_words]
    else:
        words = all_words
    
    # Mische die Reihenfolge für buntere Verarbeitung
    if sample_random:
        random.seed(123)  # Anderer Seed für Shuffle
        random.shuffle(words)
    
    return words


@lru_cache(maxsize=100000)
def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Berechnet die Levenshtein-Distanz zwischen zwei Strings.
    Cached für Performance.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def ipa_similarity(ipa1: str, ipa2: str) -> float:
    """
    Berechnet die phonetische Ähnlichkeit zwischen zwei IPA-Strings.
    Gibt einen Wert zwischen 0 (völlig unterschiedlich) und 1 (identisch) zurück.
    """
    if not ipa1 or not ipa2:
        return 0.0
    
    distance = levenshtein_distance(ipa1, ipa2)
    max_len = max(len(ipa1), len(ipa2))
    
    # Normalisierte Ähnlichkeit
    similarity = 1 - (distance / max_len)
    
    # Bonus für gleiche Länge (klingen oft ähnlicher)
    length_ratio = min(len(ipa1), len(ipa2)) / max(len(ipa1), len(ipa2))
    
    # Gewichtete Kombination
    return similarity * 0.8 + length_ratio * 0.2


def _save_checkpoint(pairs: List[Tuple], filepath: str, processed: int, total: int):
    """
    Speichert Zwischenergebnisse als JSON.
    """
    # Sortiere nach Ähnlichkeit und nehme Top 5000
    sorted_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:5000]
    
    results = []
    for source, target, similarity in sorted_pairs:
        results.append({
            'similarity': round(similarity, 4),
            'source': {'word': source.text, 'ipa': source.ipa},
            'target': {'word': target.text, 'ipa': target.ipa}
        })
    
    checkpoint_data = {
        'status': 'in_progress',
        'processed': processed,
        'total': total,
        'progress_percent': round(100 * processed / total, 1),
        'pairs_found': len(pairs),
        'top_pairs': results
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
    
    print(f"  💾 Checkpoint gespeichert: {processed}/{total} ({len(pairs)} Paare)")


def create_length_index(words: List[Word]) -> Dict[int, List[Word]]:
    """
    Erstellt einen Index nach IPA-Länge für schnellere Suche.
    """
    index = defaultdict(list)
    for word in words:
        length = len(word.ipa_clean)
        index[length].append(word)
    return dict(index)


def create_prefix_index(words: List[Word], prefix_length: int = 2) -> Dict[str, List[Word]]:
    """
    Erstellt einen Prefix-Index für schnellere Suche.
    """
    index = defaultdict(list)
    for word in words:
        if len(word.ipa_clean) >= prefix_length:
            prefix = word.ipa_clean[:prefix_length]
            index[prefix].append(word)
    return dict(index)


def find_similar_words(
    source_word: Word,
    target_words: List[Word],
    length_index: Dict[int, List[Word]],
    min_similarity: float = 0.5,
    top_n: int = 5,
    length_tolerance: int = 3
) -> List[Tuple[Word, float]]:
    """
    Findet die ähnlichsten Wörter für ein gegebenes Quellwort.
    Verwendet Length-Index für Effizienz.
    """
    source_len = len(source_word.ipa_clean)
    candidates = []
    
    # Nur Wörter mit ähnlicher Länge betrachten
    for length in range(max(2, source_len - length_tolerance), 
                        source_len + length_tolerance + 1):
        if length in length_index:
            candidates.extend(length_index[length])
    
    # Berechne Ähnlichkeit für alle Kandidaten
    results = []
    for target_word in candidates:
        similarity = ipa_similarity(source_word.ipa_clean, target_word.ipa_clean)
        if similarity >= min_similarity:
            results.append((target_word, similarity))
    
    # Sortiere nach Ähnlichkeit und gib Top-N zurück
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


def find_all_similar_pairs(
    source_words: List[Word],
    target_words: List[Word],
    min_similarity: float = 0.7,
    top_n_per_word: int = 3,
    total_top_n: int = 1000,
    progress_interval: int = 1000,
    checkpoint_interval: int = 10000,
    checkpoint_file: str = "checkpoint_pairs.json"
) -> List[Tuple[Word, Word, float]]:
    """
    Findet alle ähnlichen Wortpaare zwischen zwei Sprachen.
    Speichert Zwischenergebnisse alle checkpoint_interval Wörter.
    """
    print(f"Erstelle Index für {len(target_words)} Zielwörter...")
    length_index = create_length_index(target_words)
    
    all_pairs = []
    
    print(f"Vergleiche {len(source_words)} Quellwörter...")
    print(f"Zwischenspeicherung alle {checkpoint_interval} Wörter in: {checkpoint_file}")
    
    for i, source_word in enumerate(source_words):
        if (i + 1) % progress_interval == 0:
            print(f"  Fortschritt: {i + 1}/{len(source_words)} ({100*(i+1)/len(source_words):.1f}%) - {len(all_pairs)} Paare gefunden")
        
        # Zwischenspeicherung
        if (i + 1) % checkpoint_interval == 0:
            _save_checkpoint(all_pairs, checkpoint_file, i + 1, len(source_words))
        
        similar = find_similar_words(
            source_word, 
            target_words, 
            length_index,
            min_similarity=min_similarity,
            top_n=top_n_per_word
        )
        
        for target_word, similarity in similar:
            all_pairs.append((source_word, target_word, similarity))
    
    # Sortiere alle Paare nach Ähnlichkeit
    print(f"Sortiere {len(all_pairs)} Paare...")
    all_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Entferne Duplikate (gleiches Wortpaar mit verschiedener Reihenfolge)
    seen = set()
    unique_pairs = []
    for source, target, sim in all_pairs:
        key = (source.text.lower(), target.text.lower())
        if key not in seen:
            seen.add(key)
            unique_pairs.append((source, target, sim))
    
    return unique_pairs[:total_top_n]


def format_results(pairs: List[Tuple[Word, Word, float]], source_lang: str, target_lang: str) -> str:
    """
    Formatiert die Ergebnisse als lesbare Tabelle.
    """
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"TOP {len(pairs)} ÄHNLICH KLINGENDE WORTPAARE")
    lines.append(f"{source_lang} → {target_lang}")
    lines.append(f"{'='*80}\n")
    lines.append(f"{'Rang':<6} {'Ähnl.':<8} {source_lang + ' Wort':<25} {target_lang + ' Wort':<25} {'IPA Vergleich'}")
    lines.append(f"{'-'*6} {'-'*8} {'-'*25} {'-'*25} {'-'*40}")
    
    for i, (source, target, similarity) in enumerate(pairs, 1):
        ipa_comparison = f"/{source.ipa}/ ↔ /{target.ipa}/"
        lines.append(f"{i:<6} {similarity:>6.1%}   {source.text:<25} {target.text:<25} {ipa_comparison}")
    
    return '\n'.join(lines)


def save_results_json(pairs: List[Tuple[Word, Word, float]], filepath: str):
    """
    Speichert Ergebnisse als JSON.
    """
    results = []
    for source, target, similarity in pairs:
        results.append({
            'rank': len(results) + 1,
            'similarity': round(similarity, 4),
            'source': {
                'word': source.text,
                'ipa': source.ipa,
                'ipa_clean': source.ipa_clean
            },
            'target': {
                'word': target.text,
                'ipa': target.ipa,
                'ipa_clean': target.ipa_clean
            }
        })
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Ergebnisse gespeichert in: {filepath}")


def main():
    """
    Hauptfunktion: Lädt Daten, findet ähnliche Wortpaare, speichert Ergebnisse.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Finde ähnlich klingende Wortpaare zwischen Sprachen')
    parser.add_argument('--german', '-g', default='data/de (1).txt', help='Pfad zur deutschen Wortliste')
    parser.add_argument('--english', '-e', default='data/en_US.txt', help='Pfad zur englischen Wortliste')
    parser.add_argument('--max-words', '-m', type=int, default=10000, help='Maximale Anzahl Wörter pro Sprache (für schnellere Tests)')
    parser.add_argument('--min-similarity', '-s', type=float, default=0.7, help='Minimale Ähnlichkeit (0-1)')
    parser.add_argument('--top-n', '-n', type=int, default=100, help='Anzahl der Top-Ergebnisse')
    parser.add_argument('--output', '-o', default='similar_words.json', help='Ausgabedatei (JSON)')
    parser.add_argument('--random', '-r', action='store_true', help='Zufälliges Sampling statt erste N Wörter')
    parser.add_argument('--min-length', type=int, default=3, help='Minimale IPA-Länge')
    parser.add_argument('--max-length', type=int, default=12, help='Maximale IPA-Länge')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("IPA PHONETISCHE ÄHNLICHKEITS-ANALYSE")
    print(f"{'='*60}\n")
    
    # Lade Wortlisten
    print(f"Lade deutsche Wörter aus: {args.german}")
    german_words = parse_ipa_file(
        args.german, 
        max_words=args.max_words,
        sample_random=args.random,
        min_ipa_length=args.min_length,
        max_ipa_length=args.max_length
    )
    print(f"  → {len(german_words)} Wörter geladen")
    
    print(f"Lade englische Wörter aus: {args.english}")
    english_words = parse_ipa_file(
        args.english, 
        max_words=args.max_words,
        sample_random=args.random,
        min_ipa_length=args.min_length,
        max_ipa_length=args.max_length
    )
    print(f"  → {len(english_words)} Wörter geladen\n")
    
    # Finde ähnliche Paare (Deutsch → Englisch)
    print("Suche ähnlich klingende Wortpaare (Deutsch → Englisch)...")
    pairs = find_all_similar_pairs(
        german_words,
        english_words,
        min_similarity=args.min_similarity,
        top_n_per_word=3,
        total_top_n=args.top_n
    )
    
    # Zeige Ergebnisse
    print(format_results(pairs, "Deutsch", "Englisch"))
    
    # Speichere als JSON
    save_results_json(pairs, args.output)
    
    print(f"\n✓ Analyse abgeschlossen!")
    print(f"  {len(pairs)} ähnliche Wortpaare gefunden")


if __name__ == '__main__':
    main()
