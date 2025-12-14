#!/usr/bin/env python3
"""
IPA Phonetic Similarity Finder - ALLE PAARE VERSION

Findet ähnlich klingende Wortpaare zwischen Deutsch und Englisch
basierend auf IPA-Transkriptionen.

Diese Version speichert ALLE gefundenen Paare sofort in eine JSON-Datei,
nicht nur die Top-N.
"""

import re
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import json
import os
from dataclasses import dataclass
from functools import lru_cache


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
    normalizations = [
        ('ɫ', 'l'),   # Dark L zu normalem L
        ('ɾ', 'r'),   # Flap zu R
        ('ɹ', 'r'),   # Approximant zu R
        ('ɝ', 'ɐ'),   # Rhotischer Vokal
        ('ɚ', 'ə'),   # Rhotischer Schwa
        ('ʁ', 'r'),   # Uvular R zu R
        ('ɐ', 'a'),   # Near-open central zu a
        ('æ', 'e'),   # Near-open front zu e
        ('ɛ', 'e'),   # Open-mid front zu e
        ('ɪ', 'i'),   # Near-close front zu i
        ('ʊ', 'u'),   # Near-close back zu u
        ('ɔ', 'o'),   # Open-mid back zu o
        ('ʌ', 'a'),   # Open-mid back zu a
        ('ɑ', 'a'),   # Open back zu a
        ('ŋ', 'n'),   # Velar nasal zu n
        ('θ', 's'),   # Dental fricative zu s
        ('ð', 'd'),   # Voiced dental fricative zu d
        ('ç', 'ʃ'),   # Palataler Frikativ zu sch
        ('x', 'k'),   # Velarer Frikativ zu k
        ('ʔ', ''),    # Glottal stop entfernen
    ]
    
    for old, new in normalizations:
        ipa = ipa.replace(old, new)
    
    return ipa


def parse_ipa_file(filepath: str, max_words: Optional[int] = None, 
                   sample_random: bool = False, min_ipa_length: int = 3,
                   max_ipa_length: int = 15) -> List[Word]:
    """
    Parst eine IPA-Wortliste.
    Format: wort\t/ipa/, /ipa2/
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
            random.seed(42)
            words = random.sample(all_words, max_words)
        else:
            words = all_words[:max_words]
    else:
        words = all_words
    
    # Mische die Reihenfolge für buntere Verarbeitung
    if sample_random:
        random.seed(123)
        random.shuffle(words)
    
    return words


@lru_cache(maxsize=100000)
def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Berechnet die Levenshtein-Distanz zwischen zwei Strings.
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
    """
    if not ipa1 or not ipa2:
        return 0.0
    
    distance = levenshtein_distance(ipa1, ipa2)
    max_len = max(len(ipa1), len(ipa2))
    
    similarity = 1 - (distance / max_len)
    length_ratio = min(len(ipa1), len(ipa2)) / max(len(ipa1), len(ipa2))
    
    return similarity * 0.8 + length_ratio * 0.2


def create_length_index(words: List[Word]) -> Dict[int, List[Word]]:
    """
    Erstellt einen Index nach IPA-Länge für schnellere Suche.
    """
    index = defaultdict(list)
    for word in words:
        length = len(word.ipa_clean)
        index[length].append(word)
    return dict(index)


def find_similar_words(
    source_word: Word,
    length_index: Dict[int, List[Word]],
    min_similarity: float = 0.5,
    top_n: int = 5,
    length_tolerance: int = 3
) -> List[Tuple[Word, float]]:
    """
    Findet die ähnlichsten Wörter für ein gegebenes Quellwort.
    """
    source_len = len(source_word.ipa_clean)
    candidates = []
    
    for length in range(max(2, source_len - length_tolerance), 
                        source_len + length_tolerance + 1):
        if length in length_index:
            candidates.extend(length_index[length])
    
    results = []
    for target_word in candidates:
        similarity = ipa_similarity(source_word.ipa_clean, target_word.ipa_clean)
        if similarity >= min_similarity:
            results.append((target_word, similarity))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


class IncrementalPairSaver:
    """
    Speichert Paare inkrementell in eine JSON-Datei.
    Hält alle Paare im Speicher und schreibt periodisch die komplette Liste.
    """
    
    def __init__(self, filepath: str, save_interval: int = 500):
        self.filepath = filepath
        self.save_interval = save_interval
        self.pairs = []
        self.seen_pairs = set()  # Für Duplikat-Check
        self.pairs_since_last_save = 0
        
        # Lade existierende Paare falls vorhanden
        if os.path.exists(filepath):
            self._load_existing()
    
    def _load_existing(self):
        """Lädt existierende Paare aus der Datei."""
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.pairs = data
                    # Rebuild seen set
                    for p in self.pairs:
                        key = (p['source']['word'].lower(), p['target']['word'].lower())
                        self.seen_pairs.add(key)
                    print(f"  📂 {len(self.pairs)} existierende Paare geladen aus {self.filepath}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  ⚠️ Konnte existierende Datei nicht laden: {e}")
            self.pairs = []
            self.seen_pairs = set()
    
    def add_pair(self, source: Word, target: Word, similarity: float) -> bool:
        """
        Fügt ein Paar hinzu. Gibt True zurück wenn es ein neues Paar war.
        """
        key = (source.text.lower(), target.text.lower())
        
        if key in self.seen_pairs:
            return False
        
        self.seen_pairs.add(key)
        self.pairs.append({
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
        
        self.pairs_since_last_save += 1
        
        # Auto-save wenn genug neue Paare
        if self.pairs_since_last_save >= self.save_interval:
            self.save()
        
        return True
    
    def save(self):
        """Speichert alle Paare in die JSON-Datei."""
        if self.pairs_since_last_save == 0:
            return
        
        # Sortiere nach Ähnlichkeit vor dem Speichern
        sorted_pairs = sorted(self.pairs, key=lambda x: x['similarity'], reverse=True)
        
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(sorted_pairs, f, ensure_ascii=False, indent=2)
        
        self.pairs_since_last_save = 0
    
    def __len__(self):
        return len(self.pairs)


def save_progress(filepath: str, processed: int, total: int):
    """Speichert den Fortschritt in eine separate Datei."""
    progress = {
        'processed': processed,
        'total': total,
        'progress_percent': round(100 * processed / total, 2),
        'status': 'completed' if processed >= total else 'in_progress'
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2)


def load_progress(filepath: str) -> Optional[int]:
    """Lädt den gespeicherten Fortschritt."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if data.get('status') == 'completed':
                return None  # Von vorne anfangen wenn completed
            return data.get('processed', 0)
    except:
        return None


def find_all_similar_pairs(
    source_words: List[Word],
    target_words: List[Word],
    output_file: str,
    min_similarity: float = 0.9,
    top_n_per_word: int = 3,
    progress_interval: int = 1000,
    save_interval: int = 500,
    resume: bool = True
):
    """
    Findet alle ähnlichen Wortpaare und speichert sie inkrementell.
    """
    progress_file = output_file.replace('.json', '_progress.json')
    
    # Erstelle Saver
    saver = IncrementalPairSaver(output_file, save_interval=save_interval)
    
    # Resume-Funktion
    start_index = 0
    if resume:
        saved_progress = load_progress(progress_file)
        if saved_progress:
            start_index = saved_progress
            print(f"  ▶️ Fortsetzen ab Position {start_index}/{len(source_words)}")
    
    print(f"Erstelle Index für {len(target_words)} Zielwörter...")
    length_index = create_length_index(target_words)
    
    print(f"Vergleiche {len(source_words)} Quellwörter...")
    print(f"Ausgabedatei: {output_file}")
    print(f"Minimale Ähnlichkeit: {min_similarity:.0%}")
    print(f"Speichern alle {save_interval} neue Paare\n")
    
    new_pairs_found = 0
    
    for i in range(start_index, len(source_words)):
        source_word = source_words[i]
        
        if (i + 1) % progress_interval == 0:
            print(f"  Fortschritt: {i + 1}/{len(source_words)} ({100*(i+1)/len(source_words):.1f}%) - {len(saver)} Paare total ({new_pairs_found} neu)")
            save_progress(progress_file, i + 1, len(source_words))
        
        similar = find_similar_words(
            source_word, 
            length_index,
            min_similarity=min_similarity,
            top_n=top_n_per_word
        )
        
        for target_word, similarity in similar:
            if saver.add_pair(source_word, target_word, similarity):
                new_pairs_found += 1
    
    # Finales Speichern
    saver.save()
    save_progress(progress_file, len(source_words), len(source_words))
    
    print(f"\n✅ Fertig! {len(saver)} Paare total gespeichert in {output_file}")
    print(f"   ({new_pairs_found} neue Paare in diesem Durchlauf)")
    
    return saver.pairs


def format_top_results(pairs: List[dict], n: int = 100) -> str:
    """
    Formatiert die Top-N Ergebnisse als lesbare Tabelle.
    """
    sorted_pairs = sorted(pairs, key=lambda x: x['similarity'], reverse=True)[:n]
    
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"TOP {len(sorted_pairs)} ÄHNLICH KLINGENDE WORTPAARE")
    lines.append(f"Deutsch → Englisch")
    lines.append(f"{'='*80}\n")
    lines.append(f"{'Rang':<6} {'Ähnl.':<8} {'Deutsch Wort':<25} {'Englisch Wort':<25} {'IPA Vergleich'}")
    lines.append(f"{'-'*6} {'-'*8} {'-'*25} {'-'*25} {'-'*40}")
    
    for i, pair in enumerate(sorted_pairs, 1):
        source = pair['source']
        target = pair['target']
        similarity = pair['similarity']
        ipa_comparison = f"/{source['ipa']}/ ↔ /{target['ipa']}/"
        lines.append(f"{i:<6} {similarity:>6.1%}   {source['word']:<25} {target['word']:<25} {ipa_comparison}")
    
    return '\n'.join(lines)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Finde ähnlich klingende Wortpaare - ALLE PAARE VERSION')
    parser.add_argument('--german', '-g', default='data/de_real.txt', help='Pfad zur deutschen Wortliste')
    parser.add_argument('--english', '-e', default='data/en_US_real.txt', help='Pfad zur englischen Wortliste')
    parser.add_argument('--max-words', '-m', type=int, default=None, help='Maximale Anzahl Wörter pro Sprache')
    parser.add_argument('--min-similarity', '-s', type=float, default=0.9, help='Minimale Ähnlichkeit (0-1)')
    parser.add_argument('--output', '-o', default='all_pairs.json', help='Ausgabedatei für ALLE Paare')
    parser.add_argument('--random', '-r', action='store_true', help='Zufälliges Sampling')
    parser.add_argument('--min-length', type=int, default=3, help='Minimale IPA-Länge')
    parser.add_argument('--max-length', type=int, default=12, help='Maximale IPA-Länge')
    parser.add_argument('--no-resume', action='store_true', help='Nicht fortsetzen, von vorne anfangen')
    parser.add_argument('--show-top', type=int, default=100, help='Zeige Top-N Ergebnisse am Ende')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("IPA PHONETISCHE ÄHNLICHKEITS-ANALYSE")
    print(">>> ALLE PAARE VERSION <<<")
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
    
    # Finde ähnliche Paare
    print("Suche ähnlich klingende Wortpaare (Deutsch → Englisch)...")
    pairs = find_all_similar_pairs(
        german_words,
        english_words,
        output_file=args.output,
        min_similarity=args.min_similarity,
        resume=not args.no_resume
    )
    
    # Zeige Top Ergebnisse
    if pairs and args.show_top > 0:
        print(format_top_results(pairs, args.show_top))
    
    print(f"\n✓ Analyse abgeschlossen!")
    print(f"  {len(pairs)} ähnliche Wortpaare in {args.output}")


if __name__ == '__main__':
    main()
