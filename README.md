# Phonem-Matching

## Projektübersicht
Dieses Repository enthält ein NLP-Projekt, das sich mit der phonetischen Ähnlichkeit und semantischen Gruppierung von Wörtern beschäftigt. Ziel ist es, deutsche und englische Wörter basierend auf ihrer phonetischen und semantischen Ähnlichkeit zu analysieren und zu paaren.

## Ordnerstruktur

### Hauptverzeichnis
- **`run_pipeline.py`**: Hauptskript, um die gesamte Pipeline auszuführen.
- **`start_pipeline.sh`**: Shell-Skript, um die Pipeline zu starten.
- **`README.md`**: Diese Datei, die eine Übersicht über das Projekt bietet.

### `archive/`
Enthält ältere oder experimentelle Skripte:
- **`join_phrases.py`**: Skript zum Verbinden von Phrasen.
- **`semantic_grouper.py`**: Gruppiert Wörter basierend auf semantischen Ähnlichkeiten.
- **`sentence_builder.py`**: Erstellt Sätze aus den gruppierten Wörtern.
- **`jannik_tries/`**: Experimentelle Dateien und JSON-Daten.

### `data/`
Enthält die Daten, die für die Verarbeitung verwendet werden:
- **`raw_analysis.ipynb`**: Jupyter Notebook für die Analyse.
- **`grouped_de/`**: Enthält nach POS-Tags gruppierte deutsche Wörter.
- **`grouped_en/`**: Enthält nach POS-Tags gruppierte englische Wörter.
- **`pairing/`**: Enthält JSON-Dateien mit gepaarten Wörtern.

### `logs/`
- Enthält Logdateien, die während der Ausführung der Skripte generiert werden.

### `src/`
Enthält die Hauptskripte des Projekts:
- **`join_phrases.py`**: Verbindet Phrasen basierend auf bestimmten Regeln.
- **`phonem_matching/phonem_matching.py`**: Hauptskript für die phonetische Ähnlichkeitsberechnung.
- **`pos_categorising/`**: Enthält Skripte zur Kategorisierung von Wörtern nach POS-Tags:
  - `categoriser_de.py`: Für deutsche Wörter.
  - `categoriser_en.py`: Für englische Wörter.
- **`semantic_matching/`**: Enthält Skripte zur semantischen Paarung:
  - `de/`: Für deutsche Wörter.
  - `en/`: Für englische Wörter.
- **`sentence_builder_llm.py`**: Erstellt Sätze mithilfe eines LLM (Large Language Model).

## Nutzung
1. **Pipeline starten**:
   ```bash
   ./start_pipeline.sh
   ```
2. **Phonetisches Matching ausführen**:
   ```bash
   python src/phonem_matching/phonem_matching.py
   ```
3. **POS-Kategorisierung**:
   - Für Deutsch: `src/pos_categorising/categoriser_de.py`
   - Für Englisch: `src/pos_categorising/categoriser_en.py`

## Anforderungen
- Python 3.8 oder höher
- Abhängigkeiten:
  - `tqdm`
  - `Levenshtein`

Installiere die Abhängigkeiten mit:
```bash
pip install -r requirements.txt
```

## Lizenz
Dieses Projekt steht unter der MIT-Lizenz.