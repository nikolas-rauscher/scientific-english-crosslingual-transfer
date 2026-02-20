"""
DataTrove Pipeline Scripts

Contains the main pipeline scripts for data preprocessing and statistics:

- clean_data.py: Modular text cleaning pipeline (Citation removal + normalization)
- run_stats.py: Standard statistics pipeline (NumPy 2.0+ compatible)
- run_spacy_stats.py: spaCy-based statistics pipeline (NumPy 1.x required)

Usage:
    Run from project root:
    
    # Text cleaning
    python src/dataprep/pipelines/clean_data.py
    
    # Standard stats
    python src/dataprep/pipelines/run_stats.py
    
    # spaCy stats
    python src/dataprep/pipelines/run_spacy_stats.py

Or via Makefile:
    make clean      # Text cleaning
    make stats      # Standard stats
    make stats-spacy # spaCy stats
""" 