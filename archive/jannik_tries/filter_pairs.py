#!/usr/bin/env python3
"""
Filter für IPA-Ähnlichkeitsergebnisse.
Filtert die "guten" Wortpaare heraus.
"""

import json
import argparse
from typing import List, Dict


def is_common_word(word: str) -> bool:
    """Prüft ob es ein häufiges/interessantes Wort ist (keine Eigennamen)."""
    # Wenn es mit Großbuchstabe beginnt, ist es wahrscheinlich ein Eigenname
    # Ausnahme: Deutsche Substantive beginnen immer groß
    return True  # Erstmal alle akzeptieren


def filter_pairs(pairs: List[Dict], 
                 exclude_same: bool = True,
                 exclude_proper_nouns: bool = False,
                 min_length: int = 4,
                 max_length: int = 15,
                 exclude_english_names: bool = True,
                 min_similarity: float = 0.85) -> List[Dict]:
    """
    Filtert Wortpaare nach verschiedenen Kriterien.
    """
    
    # Bekannte englische Vornamen/Nachnamen (Auszug)
    english_names = {
        'john', 'james', 'mary', 'michael', 'david', 'smith', 'johnson', 
        'williams', 'brown', 'jones', 'miller', 'davis', 'garcia', 'wilson',
        'martinez', 'anderson', 'taylor', 'thomas', 'moore', 'jackson',
        'martin', 'lee', 'harris', 'clark', 'lewis', 'walker', 'hall',
        'allen', 'young', 'king', 'wright', 'lopez', 'hill', 'scott',
        'aaron', 'adam', 'alan', 'albert', 'alex', 'alice', 'amanda',
        'amy', 'andrew', 'angela', 'anna', 'anthony', 'arthur', 'ashley',
        'barbara', 'benjamin', 'betty', 'beverly', 'billy', 'bobby', 'brandon',
        'brenda', 'brian', 'bruce', 'carl', 'carol', 'carolyn', 'catherine',
        'charles', 'cheryl', 'chris', 'christina', 'christine', 'christopher',
        'cynthia', 'daniel', 'deborah', 'debra', 'dennis', 'diana', 'diane',
        'donald', 'donna', 'dorothy', 'douglas', 'earl', 'edward', 'elizabeth',
        'emily', 'eric', 'eugene', 'eva', 'evelyn', 'frances', 'frank',
        'fred', 'gary', 'george', 'gerald', 'gloria', 'grace', 'gregory',
        'harold', 'harry', 'heather', 'helen', 'henry', 'howard', 'irene',
        'jack', 'jacob', 'jacqueline', 'janet', 'janice', 'jason', 'jean',
        'jeffrey', 'jennifer', 'jeremy', 'jerry', 'jesse', 'jessica', 'jimmy',
        'joan', 'joe', 'johnny', 'jonathan', 'jose', 'joseph', 'joshua',
        'joyce', 'juan', 'judith', 'judy', 'julia', 'julie', 'justin',
        'karen', 'katherine', 'kathleen', 'kathryn', 'kathy', 'keith', 'kelly',
        'kenneth', 'kevin', 'kimberly', 'larry', 'laura', 'lawrence', 'linda',
        'lisa', 'lois', 'lori', 'louis', 'louise', 'margaret', 'maria',
        'marie', 'marilyn', 'mark', 'martha', 'martin', 'matthew', 'melissa',
        'michelle', 'mildred', 'nancy', 'nathan', 'nicholas', 'nicole', 'norma',
        'pamela', 'patricia', 'patrick', 'paul', 'paula', 'peter', 'philip',
        'phillip', 'phyllis', 'rachel', 'ralph', 'randy', 'raymond', 'rebecca',
        'richard', 'robert', 'robin', 'roger', 'ronald', 'rose', 'roy',
        'ruby', 'russell', 'ruth', 'ryan', 'samantha', 'samuel', 'sandra',
        'sara', 'sarah', 'scott', 'sean', 'sharon', 'shirley', 'stephanie',
        'stephen', 'steven', 'susan', 'teresa', 'terry', 'theresa', 'thomas',
        'timothy', 'tina', 'todd', 'victoria', 'vincent', 'virginia', 'walter',
        'wanda', 'wayne', 'wendy', 'william', 'willie', 'zachary',
        # Städte/Orte
        'boston', 'chicago', 'houston', 'phoenix', 'seattle', 'denver',
        'austin', 'dallas', 'portland', 'oakland', 'atlanta', 'miami',
    }
    
    filtered = []
    
    for pair in pairs:
        src = pair['source']['word']
        tgt = pair['target']['word']
        sim = pair['similarity']
        
        # Filter: Minimale Ähnlichkeit
        if sim < min_similarity:
            continue
        
        # Filter: Gleiche Wörter ausschließen
        if exclude_same and src.lower() == tgt.lower():
            continue
        
        # Filter: Zu kurze Wörter
        if len(src) < min_length or len(tgt) < min_length:
            continue
        
        # Filter: Zu lange Wörter
        if len(src) > max_length or len(tgt) > max_length:
            continue
        
        # Filter: Englische Namen ausschließen
        if exclude_english_names and tgt.lower() in english_names:
            continue
        
        # Filter: Wörter die mit Großbuchstaben enden (oft Abkürzungen)
        if src[-1].isupper() or tgt[-1].isupper():
            continue
        
        filtered.append(pair)
    
    return filtered


def categorize_pairs(pairs: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Kategorisiert Paare nach Interessantheit.
    """
    categories = {
        'bedeutungswandel': [],  # Wörter mit anderer Bedeutung
        'aehnlich': [],          # Phonetisch ähnlich
        'lustig': [],            # Lustige Kombinationen
    }
    
    # Wörter mit bekannter unterschiedlicher Bedeutung
    false_friends = {
        ('gift', 'gift'),        # DE: Gift (poison) vs EN: gift (present)
        ('rat', 'rat'),          # DE: Rat (advice) vs EN: rat (animal)
        ('kind', 'kind'),        # DE: Kind (child) vs EN: kind (freundlich)
        ('art', 'art'),          # DE: Art (type) vs EN: art (Kunst)
        ('will', 'will'),        # DE: will (wants) vs EN: will (future)
        ('mist', 'mist'),        # DE: Mist (manure) vs EN: mist (fog)
        ('see', 'see'),          # DE: See (lake) vs EN: see (sehen)
        ('bad', 'bad'),          # DE: Bad (bath) vs EN: bad (schlecht)
    }
    
    for pair in pairs:
        src = pair['source']['word'].lower()
        tgt = pair['target']['word'].lower()
        
        if (src, tgt) in false_friends:
            categories['bedeutungswandel'].append(pair)
        else:
            categories['aehnlich'].append(pair)
    
    return categories


def main():
    parser = argparse.ArgumentParser(description='Filtere IPA-Ähnlichkeitsergebnisse')
    parser.add_argument('--input', '-i', default='checkpoint_pairs.json', help='Eingabedatei')
    parser.add_argument('--output', '-o', default='filtered_pairs.json', help='Ausgabedatei')
    parser.add_argument('--min-length', type=int, default=4, help='Minimale Wortlänge')
    parser.add_argument('--min-similarity', type=float, default=0.90, help='Minimale Ähnlichkeit')
    parser.add_argument('--include-same', action='store_true', help='Gleiche Wörter einschließen')
    parser.add_argument('--include-names', action='store_true', help='Englische Namen einschließen')
    parser.add_argument('--top-n', type=int, default=100, help='Anzahl Top-Ergebnisse')
    
    args = parser.parse_args()
    
    # Lade Daten
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    pairs = data.get('top_pairs', [])
    print(f"Geladen: {len(pairs)} Paare")
    
    # Filtere
    filtered = filter_pairs(
        pairs,
        exclude_same=not args.include_same,
        min_length=args.min_length,
        min_similarity=args.min_similarity,
        exclude_english_names=not args.include_names
    )
    
    print(f"Nach Filter: {len(filtered)} Paare")
    
    # Sortiere nach Ähnlichkeit
    filtered.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Top N
    top_filtered = filtered[:args.top_n]
    
    # Ausgabe
    print(f"\n{'='*70}")
    print(f"TOP {len(top_filtered)} GEFILTERTE WORTPAARE")
    print(f"{'='*70}\n")
    
    for i, pair in enumerate(top_filtered, 1):
        src = pair['source']['word']
        tgt = pair['target']['word']
        sim = pair['similarity']
        src_ipa = pair['source']['ipa']
        tgt_ipa = pair['target']['ipa']
        print(f"{i:3}. {sim:.0%}  {src:<20} ↔ {tgt:<20}  /{src_ipa}/ ↔ /{tgt_ipa}/")
    
    # Speichere
    output_data = {
        'filtered_count': len(filtered),
        'top_pairs': top_filtered
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Gespeichert in: {args.output}")


if __name__ == '__main__':
    main()
