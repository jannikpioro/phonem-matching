# Phoneme Matching

## Project Overview
This repository contains an NLP project focused on phonetic similarity and semantic grouping of words. The goal is to analyze and pair German and English words based on their phonetic and semantic similarity.

## Important Note
The `archive/` directory contains deprecated code and resources. It is not actively maintained and was created as a result of "vibe coding"—quick, experimental coding without a structured approach. Use the contents of this directory with caution, as they may not adhere to best practices or current project standards.

## Folder Structure

### Main Directory
- **`run_pipeline.py`**: Main script to run the entire pipeline.
- **`start_pipeline.sh`**: Shell script to start the pipeline.
- **`README.md`**: This file, providing an overview of the project.

### `archive/`
Contains older or experimental scripts:
- **`join_phrases.py`**: Script for joining phrases.
- **`semantic_grouper.py`**: Groups words based on semantic similarities.
- **`sentence_builder.py`**: Builds sentences from grouped words.
- **`jannik_tries/`**: Experimental files and JSON data.

### `data/`
Contains the data used for processing:
- **`raw_analysis.ipynb`**: Jupyter notebook for analysis.
- **`grouped_de/`**: German words grouped by POS tags.
- **`grouped_en/`**: English words grouped by POS tags.
- **`pairing/`**: JSON files with paired words.

### `logs/`
- Contains log files generated during script execution.

### `src/`
Contains the main scripts of the project:
- **`join_phrases.py`**: Joins phrases based on specific rules.
- **`phonem_matching/phonem_matching.py`**: Main script for phonetic similarity calculation.
- **`pos_categorising/`**: Scripts for categorizing words by POS tags:
  - `categoriser_de.py`: For German words.
  - `categoriser_en.py`: For English words.
- **`semantic_matching/`**: Scripts for semantic pairing:
  - `de/`: For German words.
  - `en/`: For English words.
- **`sentence_builder_llm.py`**: Builds sentences using a large language model (LLM).

## Usage
1. **Start the pipeline**:
   ```bash
   ./start_pipeline.sh
   ```
2. **Run phonetic matching**:
   ```bash
   python src/phonem_matching/phonem_matching.py
   ```
3. **POS categorization**:
   - For German: `src/pos_categorising/categoriser_de.py`
   - For English: `src/pos_categorising/categoriser_en.py`

## Requirements
- Python 3.8 or higher
- Dependencies:
  - `tqdm`
  - `Levenshtein`

Install the dependencies with:
```bash
pip install -r requirements.txt
```

## License
This project is licensed under the MIT License.