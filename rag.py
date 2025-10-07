"""
RAG (Retrieval-Augmented Generation) Module for Biblical Text Analysis

This module provides functions for semantic search and retrieval of biblical passages
to support content generation and analysis tasks.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import spacy

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Global variables for caching
_bible_data = {}
_vectorizer = None
_tfidf_matrix = None
_passage_index = []
_initialized = False

# Sample Bible data structure (in a real implementation, this would be loaded from files)
_SAMPLE_BIBLE = {
    "Genesis": {
        1: {
            1: "In the beginning God created the heavens and the earth.",
            2: "Now the earth was formless and empty, darkness was over the surface of the deep, and the Spirit of God was hovering over the waters.",
            3: "And God said, 'Let there be light,' and there was light."
        },
        2: {
            7: "Then the LORD God formed a man from the dust of the ground and breathed into his nostrils the breath of life, and the man became a living being.",
            16: "And the LORD God commanded the man, 'You are free to eat from any tree in the garden; but you must not eat from the tree of the knowledge of good and evil, for when you eat from it you will certainly die.'"
        },
        3: {
            1: "Now the serpent was more crafty than any of the wild animals the LORD God had made. He said to the woman, 'Did God really say, You must not eat from any tree in the garden?'"
        }
    },
    "Exodus": {
        20: {
            1: "And God spoke all these words:",
            2: "I am the LORD your God, who brought you out of Egypt, out of the land of slavery.",
            3: "You shall have no other gods before me."
        }
    },
    "Matthew": {
        5: {
            3: "Blessed are the poor in spirit, for theirs is the kingdom of heaven.",
            4: "Blessed are those who mourn, for they will be comforted.",
            5: "Blessed are the meek, for they will inherit the earth.",
            7: "Blessed are the merciful, for they will be shown mercy.",
            9: "Blessed are the peacemakers, for they will be called children of God."
        },
        7: {
            1: "Do not judge, or you too will be judged.",
            12: "So in everything, do to others what you would have them do to you, for this sums up the Law and the Prophets."
        }
    },
    "John": {
        3: {
            16: "For God so loved the world that he gave his one and only Son, that whoever believes in him shall not perish but have eternal life.",
            17: "For God did not send his Son into the world to condemn the world, but to save the world through him."
        },
        14: {
            6: "Jesus answered, 'I am the way and the truth and the life. No one comes to the Father except through me.'"
        }
    },
    "Romans": {
        8: {
            28: "And we know that in all things God works for the good of those who love him, who have been called according to his purpose.",
            31: "What, then, shall we say in response to these things? If God is for us, who can be against us?"
        },
        12: {
            2: "Do not conform to the pattern of this world, but be transformed by the renewing of your mind."
        }
    }
}

# Sample translations for demonstration
_SAMPLE_TRANSLATIONS = {
    "ESV": "English Standard Version",
    "KJV": "King James Version", 
    "NIV": "New International Version",
    "NLT": "New Living Translation"
}


def ensure_indexes() -> None:
    """
    Initialize or ensure that the search indexes are built and ready.
    This function creates TF-IDF vectors for semantic search.
    """
    global _initialized, _vectorizer, _tfidf_matrix, _passage_index, _bible_data
    
    if _initialized:
        return
    
    # Load Bible data (using sample data for now)
    _bible_data = _SAMPLE_BIBLE.copy()
    
    # Create passage index and texts
    passages = []
    _passage_index = []
    
    for book, chapters in _bible_data.items():
        for chapter, verses in chapters.items():
            for verse, text in verses.items():
                ref = f"{book} {chapter}:{verse}"
                passages.append(text)
                _passage_index.append(ref)
    
    # Create TF-IDF vectors
    _vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        lowercase=True,
        strip_accents='unicode'
    )
    
    _tfidf_matrix = _vectorizer.fit_transform(passages)
    
    _initialized = True
    print(f"Indexed {len(passages)} biblical passages for search.")


def search_passages_for_concept(concept: str, top_k: int = 10) -> List[str]:
    """
    Search for biblical passages related to a given concept using semantic similarity.
    
    Args:
        concept: The concept or query to search for
        top_k: Number of top results to return
        
    Returns:
        List of biblical references (e.g., ["Genesis 1:1", "John 3:16"])
    """
    if not _initialized:
        ensure_indexes()
    
    # Transform the concept query
    query_vector = _vectorizer.transform([concept])
    
    # Calculate similarity scores
    similarities = cosine_similarity(query_vector, _tfidf_matrix).flatten()
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Return references for top passages
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.1:  # Threshold for relevance
            results.append(_passage_index[idx])
    
    return results


def get_passage_texts_by_ref(reference: str) -> Dict[str, str]:
    """
    Get the text of a biblical passage in different translations.
    
    Args:
        reference: Biblical reference (e.g., "Genesis 1:1", "John 3:16-17")
        
    Returns:
        Dictionary mapping translation names to passage texts
    """
    if not _initialized:
        ensure_indexes()
    
    # Parse the reference
    ref_pattern = r'^(\w+)\s+(\d+):(\d+)(?:-(\d+))?$'
    match = re.match(ref_pattern, reference.strip())
    
    if not match:
        return {}
    
    book, chapter, verse_start, verse_end = match.groups()
    chapter = int(chapter)
    verse_start = int(verse_start)
    verse_end = int(verse_end) if verse_end else verse_start
    
    # Get the base text from our sample data
    if book not in _bible_data or chapter not in _bible_data[book]:
        return {}
    
    texts = {}
    base_text = ""
    
    for verse_num in range(verse_start, verse_end + 1):
        if verse_num in _bible_data[book][chapter]:
            if base_text:
                base_text += " "
            base_text += _bible_data[book][chapter][verse_num]
    
    if not base_text:
        return {}
    
    # Simulate different translations (in a real implementation, these would be actual translations)
    translations = {
        "ESV": base_text,  # Keep original as ESV
        "KJV": _simulate_kjv_style(base_text),
        "NIV": _simulate_niv_style(base_text),
        "NLT": _simulate_nlt_style(base_text)
    }
    
    return translations


def _simulate_kjv_style(text: str) -> str:
    """Simulate King James Version style."""
    # Simple transformations to mimic KJV style
    text = text.replace("you", "thou").replace("your", "thy").replace("yours", "thine")
    text = text.replace("are", "art").replace("will", "shalt").replace("have", "hast")
    text = text.replace("do not", "doth not").replace("don't", "dost not")
    # Add -eth endings for verbs
    text = re.sub(r'\b(\w+)s\b', r'\1eth', text)
    return text


def _simulate_niv_style(text: str) -> str:
    """Simulate New International Version style."""
    # NIV tends to be more contemporary but formal
    return text.replace("God's", "the LORD's").replace("God", "the LORD")


def _simulate_nlt_style(text: str) -> str:
    """Simulate New Living Translation style."""
    # NLT tends to be more conversational
    return text.replace("shall", "will").replace("thus", "so").replace("therefore", "so")


def get_available_books() -> List[str]:
    """Get list of available Bible books."""
    if not _initialized:
        ensure_indexes()
    return list(_bible_data.keys())


def get_available_translations() -> List[str]:
    """Get list of available Bible translations."""
    return list(_SAMPLE_TRANSLATIONS.keys())


def load_bible_data(file_path: Optional[str] = None) -> None:
    """
    Load Bible data from a file. If no file is provided, uses sample data.
    
    Args:
        file_path: Path to Bible data file (JSON format)
    """
    global _bible_data, _initialized
    
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                _bible_data = json.load(f)
            print(f"Loaded Bible data from {file_path}")
        except Exception as e:
            print(f"Error loading Bible data: {e}")
            print("Using sample data instead.")
            _bible_data = _SAMPLE_BIBLE.copy()
    else:
        _bible_data = _SAMPLE_BIBLE.copy()
        print("Using sample Bible data.")
    
    # Reset initialization to rebuild indexes
    _initialized = False


if __name__ == "__main__":
    # Test the functions
    print("Testing RAG module...")
    
    # Test indexing
    ensure_indexes()
    
    # Test search
    concept = "love and forgiveness"
    results = search_passages_for_concept(concept, top_k=5)
    print(f"\nSearch results for '{concept}':")
    for ref in results:
        print(f"  {ref}")
    
    # Test passage retrieval
    ref = "John 3:16"
    texts = get_passage_texts_by_ref(ref)
    print(f"\nTexts for {ref}:")
    for translation, text in texts.items():
        print(f"  {translation}: {text}")
    
    print(f"\nAvailable books: {get_available_books()}")
    print(f"Available translations: {get_available_translations()}")