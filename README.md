# Religious Texts Analysis

A comprehensive Python library for biblical text analysis and visualization.

## Overview

This project provides tools for analyzing biblical texts across multiple dimensions:
- Loading and parsing biblical texts from various sources and translations
- Performing textual analysis (frequency, concordance, co-occurrence)
- Specialized theological analysis functions
- Visualization tools for understanding textual patterns
- Debate response modules for validating claims about biblical content

## Project Structure

```
religious-texts/
├── religious_texts/             # Main package
│   ├── __init__.py
│   ├── data_acquisition/        # Text loading and parsing
│   │   ├── __init__.py
│   │   ├── loaders.py           # Functions to load text files
│   │   └── parsers.py           # Format-specific parsers
│   ├── text_analysis/           # Basic text analysis tools
│   │   ├── __init__.py
│   │   ├── frequency.py         # Word/phrase frequency analysis
│   │   ├── concordance.py       # Concordance generation
│   │   ├── cooccurrence.py      # Co-occurrence pattern detection
│   │   ├── sentiment.py         # Sentiment analysis
│   │   └── entities.py          # Named entity recognition
│   ├── theological_analysis/    # Specialized religious text analysis
│   │   ├── __init__.py
│   │   ├── divine_names.py      # Analysis of divine name usage
│   │   ├── speech.py            # Speech attribution tracking
│   │   ├── worship.py           # Worship language identification
│   │   ├── authority.py         # Authority claims analysis
│   │   └── parallel.py          # Parallel passage comparison
│   ├── visualization/           # Data visualization tools
│   │   ├── __init__.py
│   │   ├── heatmaps.py          # Argument strength heatmaps
│   │   ├── distributions.py     # Word/concept distribution
│   │   ├── timelines.py         # Timeline representations
│   │   └── networks.py          # Concept network graphs
│   └── debate_response/         # Debate analysis tools
│       ├── __init__.py
│       ├── validators.py        # Statistical claim validation
│       ├── comparisons.py       # Interpretive framework comparison
│       ├── context.py           # Context analysis for quotes
│       └── consensus.py         # Scholarly consensus assessment
├── tests/                       # Test directory
├── examples/                    # Example scripts and notebooks
├── data/                        # Sample data directory
├── requirements.txt             # Project dependencies
├── setup.py                     # Package installation
└── .gitignore                   # Git ignore file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/asabaal/religious-texts.git
cd religious-texts

# Install in development mode
pip install -e .
```

## Usage

Basic example:
```python
from religious_texts.data_acquisition import loaders
from religious_texts.text_analysis import frequency

# Load a biblical text
bible = loaders.load_text("data/kjv.txt")

# Analyze word frequency in Genesis
frequencies = frequency.word_frequency(bible["Genesis"])
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.