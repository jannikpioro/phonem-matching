#!/usr/bin/env python3
"""
Satzgenerator für gleichklingende deutsch-englische Sätze.

Ziel: Den längsten Satz bauen, der auf Deutsch und Englisch ähnlich klingt,
syntaktisch einigermaßen korrekt ist (muss keinen Sinn ergeben).
"""

import json
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class WordPair:
    german: str
    english: str
    german_ipa: str
    english_ipa: str
    similarity: float
    pos_de: Optional[str] = None  # Part of Speech Deutsch
    pos_en: Optional[str] = None  # Part of Speech Englisch


def load_pairs(filepath: str) -> List[WordPair]:
    """Lädt Wortpaare aus JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    pairs = []
    for p in data.get('top_pairs', []):
        pairs.append(WordPair(
            german=p['source']['word'],
            english=p['target']['word'],
            german_ipa=p['source']['ipa'],
            english_ipa=p['target']['ipa'],
            similarity=p['similarity']
        ))
    return pairs


def guess_pos_german(word: str) -> str:
    """
    Rät die Wortart basierend auf deutschen Wortendungen.
    Sehr vereinfacht - für bessere Ergebnisse würde man spaCy nutzen.
    """
    word_lower = word.lower()
    
    # Substantive (großgeschrieben im Deutschen)
    if word[0].isupper() and len(word) > 1:
        return 'NOUN'
    
    # Verben
    verb_endings = ('en', 'st', 't', 'te', 'tet', 'est', 'et')
    if word_lower.endswith(verb_endings):
        # Aber nicht Substantive wie "Garten"
        if not word[0].isupper():
            return 'VERB'
    
    # Adjektive
    adj_endings = ('ig', 'lich', 'isch', 'bar', 'sam', 'haft', 'los', 'voll', 'er', 'ere', 'eres', 'em', 'en')
    if word_lower.endswith(adj_endings) and not word[0].isupper():
        return 'ADJ'
    
    # Adverbien
    if word_lower in ('sehr', 'oft', 'nie', 'immer', 'hier', 'dort', 'jetzt', 'dann'):
        return 'ADV'
    
    # Artikel/Pronomen
    if word_lower in ('der', 'die', 'das', 'ein', 'eine', 'mein', 'dein', 'sein', 'ihr'):
        return 'DET'
    
    # Präpositionen
    if word_lower in ('in', 'an', 'auf', 'mit', 'bei', 'nach', 'von', 'zu', 'für'):
        return 'PREP'
    
    return 'OTHER'


def guess_pos_english(word: str) -> str:
    """
    Rät die Wortart basierend auf englischen Wortendungen.
    """
    word_lower = word.lower()
    
    # Verben
    verb_endings = ('ing', 'ed', 'es', 's')
    if word_lower.endswith(verb_endings):
        return 'VERB'
    
    # Adjektive
    adj_endings = ('ful', 'less', 'ous', 'ive', 'able', 'ible', 'al', 'ish', 'ly')
    if word_lower.endswith(adj_endings):
        return 'ADJ'
    
    # Substantive
    noun_endings = ('tion', 'ness', 'ment', 'er', 'or', 'ist', 'ism')
    if word_lower.endswith(noun_endings):
        return 'NOUN'
    
    return 'OTHER'


def categorize_pairs(pairs: List[WordPair]) -> Dict[str, List[WordPair]]:
    """
    Kategorisiert Wortpaare nach Wortart.
    """
    for pair in pairs:
        pair.pos_de = guess_pos_german(pair.german)
        pair.pos_en = guess_pos_english(pair.english)
    
    categories = defaultdict(list)
    for pair in pairs:
        categories[pair.pos_de].append(pair)
    
    return dict(categories)


# Deutsche Satzmuster (sehr vereinfacht)
SENTENCE_PATTERNS_DE = [
    # Einfache Muster
    ['NOUN', 'VERB'],                          # "Haus brennt"
    ['DET', 'NOUN', 'VERB'],                   # "Das Haus brennt"
    ['NOUN', 'VERB', 'NOUN'],                  # "Mann sieht Haus"
    ['DET', 'NOUN', 'VERB', 'NOUN'],           # "Der Mann sieht Haus"
    ['DET', 'NOUN', 'VERB', 'DET', 'NOUN'],    # "Der Mann sieht das Haus"
    
    # Mit Adjektiven
    ['DET', 'ADJ', 'NOUN', 'VERB'],            # "Der große Mann läuft"
    ['NOUN', 'VERB', 'ADJ'],                   # "Haus ist groß"
    ['DET', 'NOUN', 'VERB', 'ADJ', 'NOUN'],    # "Der Mann sieht großes Haus"
    
    # Längere Muster
    ['DET', 'ADJ', 'NOUN', 'VERB', 'DET', 'NOUN'],
    ['DET', 'NOUN', 'VERB', 'DET', 'ADJ', 'NOUN'],
    ['DET', 'ADJ', 'NOUN', 'VERB', 'DET', 'ADJ', 'NOUN'],
    
    # Noch länger
    ['DET', 'ADJ', 'NOUN', 'VERB', 'DET', 'ADJ', 'NOUN', 'PREP', 'DET', 'NOUN'],
]


def build_sentence(pattern: List[str], categories: Dict[str, List[WordPair]], 
                   min_similarity: float = 0.95) -> Optional[Tuple[str, str, float]]:
    """
    Versucht einen Satz nach dem gegebenen Muster zu bauen.
    Gibt (deutscher_satz, englischer_satz, durchschnittliche_ähnlichkeit) zurück.
    """
    german_words = []
    english_words = []
    similarities = []
    
    used_words = set()  # Vermeide Wiederholungen
    
    for pos in pattern:
        if pos not in categories or not categories[pos]:
            return None
        
        # Finde passendes Wortpaar
        candidates = [p for p in categories[pos] 
                      if p.similarity >= min_similarity 
                      and p.german.lower() not in used_words]
        
        if not candidates:
            return None
        
        # Wähle zufällig
        pair = random.choice(candidates)
        used_words.add(pair.german.lower())
        
        german_words.append(pair.german)
        english_words.append(pair.english)
        similarities.append(pair.similarity)
    
    german_sentence = ' '.join(german_words)
    english_sentence = ' '.join(english_words)
    avg_similarity = sum(similarities) / len(similarities)
    
    return german_sentence, english_sentence, avg_similarity


def generate_sentences(pairs: List[WordPair],
                       num_attempts: int = 1000,
                       min_similarity: float = 0.95,
                       spacy_lang: str = "de",
                       perplexity_filter: bool = True,
                       perplexity_threshold: float = 100.0
                       ) -> List[Tuple[str, str, float, int, float]]:
    """
    Generiert viele Sätze und gibt die besten zurück.
    Garantiert mindestens 1 Ergebnis (Fallback), auch wenn Filter alles rauswerfen.
    """
    import spacy
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Lade spaCy-Modell
    if spacy_lang == "de":
        nlp = spacy.load("de_core_news_sm")
    else:
        nlp = spacy.load("en_core_web_sm")

    # Lade Perplexity-Modell (deutsch)
    if perplexity_filter:
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
        model = AutoModelForCausalLM.from_pretrained("dbmdz/german-gpt2")
        model.eval()

        def calc_perplexity(sentence: str) -> float:
            input_ids = tokenizer.encode(sentence, return_tensors="pt")
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
            return torch.exp(loss).item()
    else:
        def calc_perplexity(sentence: str) -> float:
            return 0.0

    categories = categorize_pairs(pairs)

    print(f"Wortpaare nach Kategorie:")
    for pos, items in sorted(categories.items()):
        high_sim = len([p for p in items if p.similarity >= min_similarity])
        print(f"  {pos}: {len(items)} Paare ({high_sim} mit ≥{min_similarity:.0%} Ähnlichkeit)")

    results = []
    backup = []  # Kandidaten, die Filter nicht bestehen (Fallback)
    patterns = sorted(SENTENCE_PATTERNS_DE, key=len, reverse=True)

    for pattern in patterns:
        for _ in range(num_attempts):
            built = build_sentence(pattern, categories, min_similarity)
            if not built:
                continue

            de, en, sim = built
            word_count = len(pattern)

            ppl = calc_perplexity(de)

            doc = nlp(de)
            has_verb_root = any(t.dep_ == "ROOT" and t.pos_ == "VERB" for t in doc)

            # immer ins Backup
            backup.append((de, en, sim, word_count, ppl))

            # nur "gute" in results
            if not has_verb_root:
                continue
            if perplexity_filter and ppl > perplexity_threshold:
                continue

            results.append((de, en, sim, word_count, ppl))

    # falls Results leer: nimm besten aus backup
    if not results:
        if not backup:
            return []  # bedeutet: build_sentence hat nie was gebaut (z.B. keine passenden POS)
        backup.sort(key=lambda x: (x[3], x[2], -x[4]), reverse=True)
        # Duplikate vermeiden
        seen = set()
        for r in backup:
            if r[0] not in seen:
                return [r]
        return [backup[0]]

    # Sortiere nach Wortanzahl, dann Ähnlichkeit, dann (niedrige) PPL bevorzugen
    results.sort(key=lambda x: (x[3], x[2], -x[4]), reverse=True)

    # Duplikate entfernen
    seen = set()
    unique_results = []
    for r in results:
        if r[0] not in seen:
            seen.add(r[0])
            unique_results.append(r)

    return unique_results



def greedy_longest_sentence(pairs: List[WordPair], 
                            min_similarity: float = 0.90) -> Tuple[str, str, List[WordPair]]:
    """
    Greedy-Algorithmus: Baue den längsten möglichen Satz,
    indem wir einfach so viele Wortpaare wie möglich aneinanderreihen.
    
    Für maximale Länge ignorieren wir strikte Grammatik und fokussieren
    auf phonetische Ähnlichkeit.
    """
    # Filtere Paare nach Ähnlichkeit
    good_pairs = [p for p in pairs if p.similarity >= min_similarity]
    
    # Sortiere nach Ähnlichkeit (höchste zuerst)
    good_pairs.sort(key=lambda x: x.similarity, reverse=True)
    
    used_german = set()
    selected_pairs = []
    
    for pair in good_pairs:
        # Vermeide gleiche Wörter
        if pair.german.lower() not in used_german:
            selected_pairs.append(pair)
            used_german.add(pair.german.lower())
    
    german_sentence = ' '.join(p.german for p in selected_pairs)
    english_sentence = ' '.join(p.english for p in selected_pairs)
    
    return german_sentence, english_sentence, selected_pairs


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generiere gleichklingende Sätze')
    parser.add_argument('--input', '-i', default='all_good_pairs.json', help='Eingabedatei')
    parser.add_argument('--min-similarity', '-s', type=float, default=0.95, help='Minimale Ähnlichkeit')
    parser.add_argument('--attempts', '-a', type=int, default=500, help='Anzahl Versuche pro Muster')
    parser.add_argument('--top-n', '-n', type=int, default=20, help='Anzahl beste Sätze')
    parser.add_argument('--mode', '-m', choices=['grammar', 'greedy'], default='grammar', 
                        help='Modus: grammar (mit Satzmuster) oder greedy (maximale Länge)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("GLEICHKLINGENDER SATZ-GENERATOR")
    print(f"{'='*70}\n")
    
    pairs = load_pairs(args.input)
    print(f"Geladen: {len(pairs)} Wortpaare\n")
    
    if args.mode == 'grammar':
        print("\n[INFO] Grammar-Mode mit spaCy- und Perplexity-Filter aktiv!")
        results = generate_sentences(
            pairs, args.attempts, args.min_similarity,
            spacy_lang="de", perplexity_filter=True, perplexity_threshold=100.0
        )
        print(f"\n{'='*70}")
        print(f"TOP {min(args.top_n, len(results))} SÄTZE (nach Länge, Ähnlichkeit, Perplexity sortiert)")
        print(f"{'='*70}\n")
        for i, (de, en, sim, word_count, ppl) in enumerate(results[:args.top_n], 1):
            print(f"{i:2}. [{word_count} Wörter, {sim:.0%} Ähnlichkeit, Perplexity: {ppl:.1f}]")
            print(f"    DE: {de}")
            print(f"    EN: {en}")
            print()
    
    else:  # greedy mode
        print("Greedy-Modus: Maximale Wortanzahl (ohne Grammatik-Prüfung)\n")
        
        de_sentence, en_sentence, selected = greedy_longest_sentence(pairs, args.min_similarity)
        
        print(f"Längster Satz: {len(selected)} Wörter")
        print(f"\nDeutsch:\n{de_sentence}")
        print(f"\nEnglisch:\n{en_sentence}")
        
        print(f"\n\nWortpaare im Detail:")
        for i, p in enumerate(selected[:30], 1):  # Zeige erste 30
            print(f"  {i:2}. {p.german:<15} ↔ {p.english:<15} ({p.similarity:.0%})")
        
        if len(selected) > 30:
            print(f"  ... und {len(selected) - 30} weitere")


if __name__ == '__main__':
    main()
