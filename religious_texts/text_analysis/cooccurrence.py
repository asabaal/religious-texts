"""
Co-occurrence Analysis Module

This module provides functions for analyzing co-occurrence patterns of words
and concepts in biblical texts.
"""

import re
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Union, Any, Tuple, Set, DefaultDict

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams

# Try to import optional dependencies
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


def word_cooccurrence(bible_dict: Dict[str, Any], words: List[str], 
                     window_size: int = 5, ignore_case: bool = True,
                     unit: str = 'verse', books: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate co-occurrence frequencies between specified words.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        words: List of words to analyze co-occurrence for
        window_size: Size of word window for co-occurrence (only used if unit='window')
        ignore_case: Whether to ignore case when matching words
        unit: Unit for co-occurrence calculation ('verse', 'sentence', or 'window')
        books: Optional list of books to include
        
    Returns:
        DataFrame with co-occurrence counts between all pairs of words
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> divine_names = ["god", "lord", "jesus", "christ", "spirit"]
        >>> cooccur = word_cooccurrence(bible, divine_names)
    """
    # Prepare words list
    if ignore_case:
        words = [w.lower() for w in words]
    
    # Initialize co-occurrence matrix as a nested dictionary
    cooccur_dict = {w1: {w2: 0 for w2 in words} for w1 in words}
    
    # Determine which books to include
    if books:
        filtered_books = {book: bible_dict[book] for book in books if book in bible_dict}
    else:
        filtered_books = bible_dict
    
    # Process based on unit type
    if unit.lower() == 'verse':
        # Count co-occurrences within verses
        for book, chapters in filtered_books.items():
            for chapter_num, verses in chapters.items():
                for verse_num, verse_text in verses.items():
                    # Tokenize and process text
                    text = verse_text.lower() if ignore_case else verse_text
                    tokens = word_tokenize(text)
                    
                    # Count word occurrences in this verse
                    word_counts = Counter()
                    for token in tokens:
                        if token in words:
                            word_counts[token] += 1
                    
                    # Update co-occurrence counts
                    for w1 in word_counts:
                        for w2 in word_counts:
                            if w1 != w2:  # Don't count co-occurrence with self
                                cooccur_dict[w1][w2] += word_counts[w1] * word_counts[w2]
    
    elif unit.lower() == 'sentence':
        # Count co-occurrences within sentences
        for book, chapters in filtered_books.items():
            for chapter_num, verses in chapters.items():
                for verse_num, verse_text in verses.items():
                    # Split into sentences
                    sentences = sent_tokenize(verse_text)
                    
                    for sentence in sentences:
                        # Tokenize and process text
                        text = sentence.lower() if ignore_case else sentence
                        tokens = word_tokenize(text)
                        
                        # Count word occurrences in this sentence
                        word_counts = Counter()
                        for token in tokens:
                            if token in words:
                                word_counts[token] += 1
                        
                        # Update co-occurrence counts
                        for w1 in word_counts:
                            for w2 in word_counts:
                                if w1 != w2:  # Don't count co-occurrence with self
                                    cooccur_dict[w1][w2] += word_counts[w1] * word_counts[w2]
    
    elif unit.lower() == 'window':
        # Count co-occurrences within sliding windows
        for book, chapters in filtered_books.items():
            for chapter_num, verses in chapters.items():
                for verse_num, verse_text in verses.items():
                    # Tokenize and process text
                    text = verse_text.lower() if ignore_case else verse_text
                    tokens = word_tokenize(text)
                    
                    # Check each token
                    for i, token in enumerate(tokens):
                        if token in words:
                            # Define window around this token
                            window_start = max(0, i - window_size)
                            window_end = min(len(tokens), i + window_size + 1)
                            window = tokens[window_start:window_end]
                            
                            # Count other target words in window
                            for other_token in window:
                                if other_token in words and other_token != token:
                                    cooccur_dict[token][other_token] += 1
    
    else:
        raise ValueError(f"Unit must be 'verse', 'sentence', or 'window', got '{unit}'")
    
    # Convert to DataFrame
    cooccur_df = pd.DataFrame(cooccur_dict)
    
    return cooccur_df


def concept_cooccurrence(bible_dict: Dict[str, Any], 
                        concepts: Dict[str, List[str]],
                        ignore_case: bool = True,
                        unit: str = 'verse', 
                        books: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate co-occurrence between concepts (groups of related words).
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        concepts: Dictionary mapping concept names to lists of related words
        ignore_case: Whether to ignore case when matching words
        unit: Unit for co-occurrence calculation ('verse', 'sentence', or 'chapter')
        books: Optional list of books to include
        
    Returns:
        DataFrame with co-occurrence counts between all pairs of concepts
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> concepts = {
        ...     "divine": ["god", "lord", "almighty"],
        ...     "love": ["love", "charity", "compassion"],
        ...     "judgment": ["judge", "judgment", "wrath", "punish"]
        ... }
        >>> concept_co = concept_cooccurrence(bible, concepts)
    """
    # Prepare concepts dictionary
    if ignore_case:
        concepts = {name: [w.lower() for w in words] for name, words in concepts.items()}
    
    # Initialize co-occurrence matrix
    concept_names = list(concepts.keys())
    cooccur_dict = {c1: {c2: 0 for c2 in concept_names} for c1 in concept_names}
    
    # Create a mapping from words to their concepts
    word_to_concept = {}
    for concept, words in concepts.items():
        for word in words:
            if word in word_to_concept:
                word_to_concept[word].append(concept)
            else:
                word_to_concept[word] = [concept]
    
    # Determine which books to include
    if books:
        filtered_books = {book: bible_dict[book] for book in books if book in bible_dict}
    else:
        filtered_books = bible_dict
    
    # Process based on unit type
    if unit.lower() == 'verse':
        # Count co-occurrences within verses
        for book, chapters in filtered_books.items():
            for chapter_num, verses in chapters.items():
                for verse_num, verse_text in verses.items():
                    # Find which concepts are present in this verse
                    text = verse_text.lower() if ignore_case else verse_text
                    tokens = word_tokenize(text)
                    
                    # Count concepts present in this verse
                    concept_present = defaultdict(bool)
                    
                    for token in tokens:
                        if token in word_to_concept:
                            for concept in word_to_concept[token]:
                                concept_present[concept] = True
                    
                    # Update co-occurrence counts for all pairs of present concepts
                    concepts_found = [c for c in concept_names if concept_present[c]]
                    
                    for i, c1 in enumerate(concepts_found):
                        for c2 in concepts_found[i+1:]:  # Avoid duplicates
                            cooccur_dict[c1][c2] += 1
                            cooccur_dict[c2][c1] += 1  # Symmetric relation
    
    elif unit.lower() == 'sentence':
        # Count co-occurrences within sentences
        for book, chapters in filtered_books.items():
            for chapter_num, verses in chapters.items():
                for verse_num, verse_text in verses.items():
                    # Split into sentences
                    sentences = sent_tokenize(verse_text)
                    
                    for sentence in sentences:
                        # Find which concepts are present in this sentence
                        text = sentence.lower() if ignore_case else sentence
                        tokens = word_tokenize(text)
                        
                        # Count concepts present in this sentence
                        concept_present = defaultdict(bool)
                        
                        for token in tokens:
                            if token in word_to_concept:
                                for concept in word_to_concept[token]:
                                    concept_present[concept] = True
                        
                        # Update co-occurrence counts for all pairs of present concepts
                        concepts_found = [c for c in concept_names if concept_present[c]]
                        
                        for i, c1 in enumerate(concepts_found):
                            for c2 in concepts_found[i+1:]:  # Avoid duplicates
                                cooccur_dict[c1][c2] += 1
                                cooccur_dict[c2][c1] += 1  # Symmetric relation
    
    elif unit.lower() == 'chapter':
        # Count co-occurrences within chapters
        for book, chapters in filtered_books.items():
            for chapter_num, verses in chapters.items():
                # Combine all verses in the chapter
                chapter_text = ' '.join(verse_text for verse_text in verses.values())
                
                # Find which concepts are present in this chapter
                text = chapter_text.lower() if ignore_case else chapter_text
                tokens = word_tokenize(text)
                
                # Count concepts present in this chapter
                concept_present = defaultdict(bool)
                
                for token in tokens:
                    if token in word_to_concept:
                        for concept in word_to_concept[token]:
                            concept_present[concept] = True
                
                # Update co-occurrence counts for all pairs of present concepts
                concepts_found = [c for c in concept_names if concept_present[c]]
                
                for i, c1 in enumerate(concepts_found):
                    for c2 in concepts_found[i+1:]:  # Avoid duplicates
                        cooccur_dict[c1][c2] += 1
                        cooccur_dict[c2][c1] += 1  # Symmetric relation
    
    else:
        raise ValueError(f"Unit must be 'verse', 'sentence', or 'chapter', got '{unit}'")
    
    # Convert to DataFrame
    cooccur_df = pd.DataFrame(cooccur_dict)
    
    return cooccur_df


def proximity_analysis(bible_dict: Dict[str, Any], target_word: str, 
                      context_words: List[str], window_size: int = 5, 
                      ignore_case: bool = True,
                      books: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Analyze proximity between a target word and context words.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        target_word: The main word to analyze
        context_words: List of words to check proximity for
        window_size: Maximum distance to consider for proximity
        ignore_case: Whether to ignore case when matching words
        books: Optional list of books to include
        
    Returns:
        Dictionary mapping context words to their average proximity to the target word
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> prox = proximity_analysis(bible, "love", ["god", "neighbor", "enemy", "command"])
        >>> for word, score in sorted(prox.items(), key=lambda x: x[1]):
        ...     print(f"{word}: {score:.2f}")
    """
    # Prepare words
    if ignore_case:
        target_word = target_word.lower()
        context_words = [w.lower() for w in context_words]
    
    # Initialize counters and distance sums
    total_counts = {word: 0 for word in context_words}
    distance_sums = {word: 0 for word in context_words}
    
    # Determine which books to include
    if books:
        filtered_books = {book: bible_dict[book] for book in books if book in bible_dict}
    else:
        filtered_books = bible_dict
    
    # Process each verse
    for book, chapters in filtered_books.items():
        for chapter_num, verses in chapters.items():
            for verse_num, verse_text in verses.items():
                # Tokenize and process text
                text = verse_text.lower() if ignore_case else verse_text
                tokens = word_tokenize(text)
                
                # Find all positions of the target word
                target_positions = [i for i, token in enumerate(tokens) if token == target_word]
                
                if not target_positions:
                    continue
                
                # For each context word, find positions and calculate distances
                for word in context_words:
                    word_positions = [i for i, token in enumerate(tokens) if token == word]
                    
                    if not word_positions:
                        continue
                    
                    # Calculate minimum distance between each target and context word occurrence
                    for t_pos in target_positions:
                        # Find closest occurrence
                        min_distance = min(abs(t_pos - w_pos) for w_pos in word_positions)
                        
                        # Only count if within window
                        if min_distance <= window_size:
                            distance_sums[word] += min_distance
                            total_counts[word] += 1
    
    # Calculate average distances
    avg_distances = {}
    for word in context_words:
        if total_counts[word] > 0:
            avg_distances[word] = distance_sums[word] / total_counts[word]
        else:
            avg_distances[word] = float('inf')  # No co-occurrences found
    
    # Convert to proximity scores (higher is closer)
    proximity_scores = {}
    max_distance = window_size
    
    for word in context_words:
        if avg_distances[word] == float('inf'):
            proximity_scores[word] = 0.0  # No proximity
        else:
            # Convert distance to proximity score (1 - normalized distance)
            proximity_scores[word] = 1.0 - (avg_distances[word] / max_distance)
    
    return proximity_scores


def create_cooccurrence_network(cooccur_df: pd.DataFrame, 
                               threshold: float = 0.0) -> Optional[Any]:
    """
    Create a network graph from co-occurrence data.
    
    Args:
        cooccur_df: DataFrame with co-occurrence counts
        threshold: Minimum co-occurrence value to include as edge
        
    Returns:
        NetworkX graph object if networkx is available, None otherwise
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> concepts = {"divine": ["god", "lord"], "love": ["love", "charity"]}
        >>> cooccur = concept_cooccurrence(bible, concepts)
        >>> G = create_cooccurrence_network(cooccur)
        >>> # Now you can visualize G using networkx drawing functions
    """
    if not HAS_NETWORKX:
        raise ImportError("NetworkX is required for network creation. Please install it with 'pip install networkx'.")
    
    # Create empty graph
    G = nx.Graph()
    
    # Add nodes (words/concepts)
    for node in cooccur_df.columns:
        G.add_node(node)
    
    # Add edges with co-occurrence counts as weights
    for i, row_name in enumerate(cooccur_df.index):
        for col_name in cooccur_df.columns[i+1:]:  # Upper triangle only to avoid duplicates
            weight = cooccur_df.loc[row_name, col_name]
            
            # Only add edges above threshold
            if weight > threshold:
                G.add_edge(row_name, col_name, weight=weight)
    
    return G


def find_concept_clusters(bible_dict: Dict[str, Any], words: List[str],
                         unit: str = 'verse', min_cooccurrence: int = 2,
                         ignore_case: bool = True,
                         books: Optional[List[str]] = None) -> List[Set[str]]:
    """
    Identify clusters of words that frequently appear together.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        words: List of words to analyze for clustering
        unit: Unit for co-occurrence calculation ('verse', 'sentence', or 'chapter')
        min_cooccurrence: Minimum number of co-occurrences to form a connection
        ignore_case: Whether to ignore case when matching words
        books: Optional list of books to include
        
    Returns:
        List of sets containing words that cluster together
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> theological_terms = ["grace", "faith", "works", "law", "gospel", "jesus", 
        ...                      "christ", "justified", "saved", "born", "again"]
        >>> clusters = find_concept_clusters(bible, theological_terms)
        >>> for i, cluster in enumerate(clusters, 1):
        ...     print(f"Cluster {i}: {', '.join(cluster)}")
    """
    # Get co-occurrence data
    cooccur_df = word_cooccurrence(bible_dict, words, ignore_case=ignore_case, 
                                  unit=unit, books=books)
    
    # Create a network if networkx is available
    if not HAS_NETWORKX:
        raise ImportError("NetworkX is required for clustering. Please install it with 'pip install networkx'.")
    
    # Create graph with a threshold
    G = nx.Graph()
    
    # Add nodes
    for word in words:
        G.add_node(word)
    
    # Add edges based on co-occurrence threshold
    for i, word1 in enumerate(words):
        for word2 in words[i+1:]:
            # Get co-occurrence count (use the larger of the two directions)
            count = max(cooccur_df.loc[word1, word2], cooccur_df.loc[word2, word1])
            
            if count >= min_cooccurrence:
                G.add_edge(word1, word2, weight=count)
    
    # Find connected components (clusters)
    clusters = [set(c) for c in nx.connected_components(G)]
    
    # Sort clusters by size (largest first)
    clusters.sort(key=len, reverse=True)
    
    return clusters
