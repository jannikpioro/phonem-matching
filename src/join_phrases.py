#!/usr/bin/env python3
"""
Join the three pairing JSONs to produce:
- All possible 3-word and 4-word English and German phrases:
  - adj noun verb (where adj-noun and noun-verb overlap on the noun)
  - noun adv verb (where adv-verb and noun-verb overlap on the verb)
  - adj noun adv verb (where all three overlap)
For each phrase, output:
- german_phrase
- german_phonetic
- english_phrase
- english_phonetic
- phonetic_similarity (average of all components)
- semantic_similarity (spaCy similarity of the full English phrase)
Output: data/pairing/en/joined_phrases.json
"""
import json
from pathlib import Path
from collections import defaultdict
from itertools import product
import spacy
from tqdm import tqdm

PAIRING_DIR = Path(__file__).parent.parent / 'data' / 'pairing' / 'en'
ADJ_NOUN_FILE = PAIRING_DIR / 'adjective_noun_pairs.json'
ADV_VERB_FILE = PAIRING_DIR / 'adverb_verb_pairs.json'
NOUN_VERB_FILE = PAIRING_DIR / 'noun_verb_pairs.json'

nlp = spacy.load("en_core_web_lg")

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)['pairings']

def phrase_similarity(phrase):
    doc = nlp(phrase)
    return doc.vector_norm if doc.has_vector else 0.0

def main():
    adj_noun = load_json(ADJ_NOUN_FILE)
    adv_verb = load_json(ADV_VERB_FILE)
    noun_verb = load_json(NOUN_VERB_FILE)

    # Index by noun and verb
    adj_noun_by_noun = defaultdict(list)
    for p in adj_noun:
        noun = p['english_phrase'].split()[1]
        adj_noun_by_noun[noun].append(p)

    noun_verb_by_noun = defaultdict(list)
    noun_verb_by_verb = defaultdict(list)
    for p in noun_verb:
        noun, verb = p['english_phrase'].split()
        noun_verb_by_noun[noun].append(p)
        noun_verb_by_verb[verb].append(p)

    adv_verb_by_verb = defaultdict(list)
    for p in adv_verb:
        # adverb_verb_pairs.json is stored as "verb adverb" -> verb is first token
        verb = p['english_phrase'].split()[0]
        adv_verb_by_verb[verb].append(p)

    # Count total for progress bar
    total = 0
    total += sum(len(adj_noun_by_noun[noun]) * len(noun_verb_by_noun[noun]) for noun in set(adj_noun_by_noun) & set(noun_verb_by_noun))
    total += sum(len(noun_verb_by_verb[verb]) * len(adv_verb_by_verb[verb]) for verb in set(noun_verb_by_verb) & set(adv_verb_by_verb))
    total += sum(len([nv for nv in noun_verb_by_noun[noun] if nv['english_phrase'].split()[1] == verb]) * len(adj_noun_by_noun[noun]) * len(adv_verb_by_verb[verb])
                 for noun in set(adj_noun_by_noun) & set(noun_verb_by_noun)
                 for verb in set(noun_verb_by_verb) & set(adv_verb_by_verb))

    results_3 = []
    results_4 = []
    pbar = tqdm(total=total, desc="Joining phrases", unit="phrase")

    # 3-word: adj noun verb
    for noun in set(adj_noun_by_noun) & set(noun_verb_by_noun):
        for adjn in adj_noun_by_noun[noun]:
            for nv in noun_verb_by_noun[noun]:
                english_phrase = f"{adjn['english_phrase']} {nv['english_phrase'].split()[1]}"
                german_phrase = f"{adjn['german_phrase']} {nv['german_phrase'].split()[1]}"
                english_phonetic = f"{adjn['english_phonetic']} {nv['english_phonetic'].split()[1]}"
                german_phonetic = f"{adjn['german_phonetic']} {nv['german_phonetic'].split()[1]}"
                phonetic_sim = (adjn['adjective_phonetic_similarity'] + adjn['noun_phonetic_similarity'] + nv['verb_phonetic_similarity']) / 3
                semantic_sim = phrase_similarity(english_phrase)
                results_3.append({
                    'english_phrase': english_phrase,
                    'german_phrase': german_phrase,
                    'english_phonetic': english_phonetic,
                    'german_phonetic': german_phonetic,
                    'phonetic_similarity': round(phonetic_sim, 4),
                    'semantic_similarity': round(semantic_sim, 4)
                })
                pbar.update(1)

    # 3-word: noun adv verb
    for verb in set(noun_verb_by_verb) & set(adv_verb_by_verb):
        for nv in noun_verb_by_verb[verb]:
            noun_en, _ = nv['english_phrase'].split()
            noun_de, _ = nv['german_phrase'].split()
            noun_en_phon, _ = nv['english_phonetic'].split()
            noun_de_phon, _ = nv['german_phonetic'].split()

            for advv in adv_verb_by_verb[verb]:
                # adverb_verb_pairs.json is "verb adverb" -> reorder to "adverb verb"
                verb_en, adv_en = advv['english_phrase'].split()
                verb_de, adv_de = advv['german_phrase'].split()
                verb_en_phon, adv_en_phon = advv['english_phonetic'].split()
                verb_de_phon, adv_de_phon = advv['german_phonetic'].split()

                english_phrase = f"{noun_en} {adv_en} {verb_en}"
                german_phrase = f"{noun_de} {adv_de} {verb_de}"
                english_phonetic = f"{noun_en_phon} {adv_en_phon} {verb_en_phon}"
                german_phonetic = f"{noun_de_phon} {adv_de_phon} {verb_de_phon}"
                phonetic_sim = (nv['noun_phonetic_similarity'] + nv['verb_phonetic_similarity'] + advv['adverb_phonetic_similarity']) / 3
                semantic_sim = phrase_similarity(english_phrase)
                results_3.append({
                    'english_phrase': english_phrase,
                    'german_phrase': german_phrase,
                    'english_phonetic': english_phonetic,
                    'german_phonetic': german_phonetic,
                    'phonetic_similarity': round(phonetic_sim, 4),
                    'semantic_similarity': round(semantic_sim, 4)
                })
                pbar.update(1)

        # 4-word: adj noun verb adv  (OUTPUT ORDER CHANGED)
    for noun in set(adj_noun_by_noun) & set(noun_verb_by_noun):
        for verb in set(noun_verb_by_verb) & set(adv_verb_by_verb):
            for nv in [p for p in noun_verb_by_noun[noun] if p['english_phrase'].split()[1] == verb]:
                for adjn in adj_noun_by_noun[noun]:
                    for advv in adv_verb_by_verb[verb]:
                        # adverb_verb_pairs.json is stored as "verb adverb"
                        verb_en, adv_en = advv['english_phrase'].split()
                        verb_de, adv_de = advv['german_phrase'].split()
                        verb_en_phon, adv_en_phon = advv['english_phonetic'].split()
                        verb_de_phon, adv_de_phon = advv['german_phonetic'].split()

                        # NEW ORDER: adj noun verb adv
                        english_phrase = f"{adjn['english_phrase']} {verb_en} {adv_en}"
                        german_phrase = f"{adjn['german_phrase']} {verb_de} {adv_de}"
                        english_phonetic = f"{adjn['english_phonetic']} {verb_en_phon} {adv_en_phon}"
                        german_phonetic = f"{adjn['german_phonetic']} {verb_de_phon} {adv_de_phon}"

                        phonetic_sim = (
                            adjn['adjective_phonetic_similarity']
                            + adjn['noun_phonetic_similarity']
                            + nv['verb_phonetic_similarity']
                            + advv['adverb_phonetic_similarity']
                        ) / 4

                        semantic_sim = phrase_similarity(english_phrase)

                        results_4.append({
                            'english_phrase': english_phrase,
                            'german_phrase': german_phrase,
                            'english_phonetic': english_phonetic,
                            'german_phonetic': german_phonetic,
                            'phonetic_similarity': round(phonetic_sim, 4),
                            'semantic_similarity': round(semantic_sim, 4)
                        })
                        pbar.update(1)


    pbar.close()
    out_file_3 = PAIRING_DIR / 'joined_phrases_3.json'
    out_file_4 = PAIRING_DIR / 'joined_phrases_4.json'
    with open(out_file_3, 'w', encoding='utf-8') as f:
        json.dump(results_3, f, indent=2, ensure_ascii=False)
    with open(out_file_4, 'w', encoding='utf-8') as f:
        json.dump(results_4, f, indent=2, ensure_ascii=False)
    print(f"Output written to {out_file_3} and {out_file_4}")

if __name__ == '__main__':
    main()
