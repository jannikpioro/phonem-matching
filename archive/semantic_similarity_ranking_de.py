#!/usr/bin/env python3
"""
Sort joined phrases by semantic strength of the German phrase using spaCy.

- Reads joined_phrases_3.json and/or joined_phrases_4.json from data/pairing/de/
- Computes german_semantic_similarity = doc.vector_norm for german_phrase
- Sorts descending by german_semantic_similarity
- Writes *_sorted_de.json next to the input

Input JSON can be:
- a list of dicts, OR
- a dict with {"pairings": [...]}
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import spacy
from tqdm import tqdm


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_items(data: Any) -> Tuple[List[Dict], bool]:
    """Return (items, had_wrapper)."""
    if isinstance(data, dict) and isinstance(data.get("pairings"), list):
        return data["pairings"], True
    if isinstance(data, list):
        return data, False
    raise ValueError("Unsupported JSON structure. Expected list or dict with 'pairings' list.")


def write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def ensure_spacy_model(name: str):
    try:
        return spacy.load(name)
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", name], check=False)
        return spacy.load(name)


def compute_vector_norms(nlp, phrases: List[str], batch_size: int = 256) -> List[float]:
    norms: List[float] = []
    for doc in tqdm(nlp.pipe(phrases, batch_size=batch_size), total=len(phrases), desc="spaCy(de) vectors"):
        norms.append(float(doc.vector_norm) if doc.has_vector else 0.0)
    return norms


def main():
    pairing_dir = Path(__file__).parent.parent / "data" / "pairing" / "en"
    inputs = [
        pairing_dir / "joined_phrases_3.json"
    ]

    nlp = ensure_spacy_model("de_core_news_lg")

    for in_path in inputs:
        if not in_path.exists():
            continue

        data = load_json(in_path)
        items, had_wrapper = extract_items(data)

        german_phrases = [str(it.get("german_phrase", "")) for it in items]
        norms = compute_vector_norms(nlp, german_phrases)

        for it, score in zip(items, norms):
            it["german_semantic_similarity"] = round(score, 4)

        items_sorted = sorted(items, key=lambda d: float(d.get("german_semantic_similarity", 0.0)), reverse=True)

        out_path = in_path.with_name(in_path.stem + "_sorted_de.json")
        if had_wrapper:
            data["pairings"] = items_sorted
            write_json(out_path, data)
        else:
            write_json(out_path, items_sorted)

        print(f"✅ Sorted by german_semantic_similarity: {in_path} -> {out_path} (n={len(items_sorted)})")


if __name__ == "__main__":
    main()
