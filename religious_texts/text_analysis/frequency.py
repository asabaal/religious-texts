"""
Frequency Analysis Module

This module provides functions for word and phrase frequency analysis in biblical texts.
"""

import re
import string
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union, Any, Tuple, Set

import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams

# Try to download NLTK resources; handle offline scenarios gracefully
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Initialize stopwords
STOPWORDS = set()
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
except:
    # Fallback basic English stopwords
    STOPWORDS = {
        'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
        'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
        'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
        'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
        'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
        's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm',
        'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn',
        'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn',
        'wasn', 'weren', 'won', 'wouldn'
    }


def _extract_text(bible_dict: Dict[str, Any], book: Optional[str] = None, 
                 chapter: Optional[int] = None, verse: Optional[int] = None) -> str:
    """
    Helper function to extract text from a Bible dictionary based on filters.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        book: Optional book name to filter by
        chapter: Optional chapter number to filter by
        verse: Optional verse number to filter by
        
    Returns:
        String containing the combined text of all matching passages
    """
    text = []
    
    # Filter by book
    if book:
        if book not in bible_dict:
            return ""
        books_to_check = {book: bible_dict[book]}
    else:
        books_to_check = bible_dict
    
    # Extract text based on filters
    for book_name, chapters in books_to_check.items():
        # Filter by chapter
        if chapter:
            if chapter not in chapters:
                continue
            chapters_to_check = {chapter: chapters[chapter]}
        else:
            chapters_to_check = chapters
        
        for chapter_num, verses in chapters_to_check.items():
            # Filter by verse
            if verse:
                if verse not in verses:
                    continue
                verses_to_check = {verse: verses[verse]}
            else:
                verses_to_check = verses
            
            for verse_num, verse_text in verses_to_check.items():
                text.append(verse_text)
    
    return " ".join(text)


def _preprocess_text(text: str, remove_stopwords: bool = False, 
                    remove_punctuation: bool = True, lowercase: bool = True) -> List[str]:
    """
    Preprocess text for frequency analysis.
    
    Args:
        text: Input text to process
        remove_stopwords: Whether to remove common stopwords
        remove_punctuation: Whether to remove punctuation
        lowercase: Whether to convert text to lowercase
        
    Returns:
        List of preprocessed tokens
    """
    if lowercase:
        text = text.lower()
    
    if remove_punctuation:
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords if requested
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    
    return tokens


def word_frequency(bible_dict: Dict[str, Any], book: Optional[str] = None, 
                  chapter: Optional[int] = None, verse: Optional[int] = None,
                  remove_stopwords: bool = True, top_n: Optional[int] = None,
                  custom_stopwords: Optional[Set[str]] = None) -> Dict[str, int]:
    """
    Calculate word frequency in the specified biblical text.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        book: Optional book name to filter by
        chapter: Optional chapter number to filter by
        verse: Optional verse number to filter by
        remove_stopwords: Whether to remove common stopwords
        top_n: Optional limit to return only the top N most frequent words
        custom_stopwords: Optional additional stopwords to remove
        
    Returns:
        Dictionary of {word: frequency} sorted by frequency (descending)
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> frequencies = word_frequency(bible, book="Genesis")
        >>> print(frequencies)
        {'god': 245, 'said': 203, 'land': 140, ...}
    """
    # Extract text based on filters
    text = _extract_text(bible_dict, book, chapter, verse)
    
    if not text:
        return {}
    
    # Create combined stopwords set if needed
    all_stopwords = STOPWORDS
    if custom_stopwords:
        all_stopwords = STOPWORDS.union(custom_stopwords)
    
    # Preprocess text
    tokens = _preprocess_text(text, remove_stopwords=remove_stopwords, 
                             lowercase=True, remove_punctuation=True)
    
    if remove_stopwords and custom_stopwords:
        tokens = [token for token in tokens if token not in custom_stopwords]
    
    # Count frequencies
    word_counts = Counter(tokens)
    
    # Sort by frequency (descending)
    sorted_counts = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True))
    
    # Limit to top_n if specified
    if top_n:
        sorted_counts = dict(list(sorted_counts.items())[:top_n])
    
    return sorted_counts


def phrase_frequency(bible_dict: Dict[str, Any], n: int = 2, book: Optional[str] = None,
                    chapter: Optional[int] = None, verse: Optional[int] = None,
                    remove_stopwords: bool = True, top_n: Optional[int] = None) -> Dict[str, int]:
    """
    Calculate n-gram (phrase) frequency in the specified biblical text.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        n: Number of words in each phrase (n-gram size)
        book: Optional book name to filter by
        chapter: Optional chapter number to filter by
        verse: Optional verse number to filter by
        remove_stopwords: Whether to remove common stopwords
        top_n: Optional limit to return only the top N most frequent phrases
        
    Returns:
        Dictionary of {phrase: frequency} sorted by frequency (descending)
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> bigrams = phrase_frequency(bible, n=2, book="Genesis")
        >>> print(bigrams)
        {'the lord': 60, 'came to': 18, ...}
    """
    # Extract text based on filters
    text = _extract_text(bible_dict, book, chapter, verse)
    
    if not text:
        return {}
    
    # Preprocess text
    tokens = _preprocess_text(text, remove_stopwords=remove_stopwords,
                             lowercase=True, remove_punctuation=True)
    
    # Generate n-grams
    n_grams = list(ngrams(tokens, n))
    
    # Convert to strings for easier counting
    n_gram_strings = [' '.join(gram) for gram in n_grams]
    
    # Count frequencies
    phrase_counts = Counter(n_gram_strings)
    
    # Sort by frequency (descending)
    sorted_counts = dict(sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True))
    
    # Limit to top_n if specified
    if top_n:
        sorted_counts = dict(list(sorted_counts.items())[:top_n])
    
    return sorted_counts


def frequency_distribution(bible_dict: Dict[str, Any], words: List[str], 
                          unit: str = 'chapter', normalize: bool = False,
                          books: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate the distribution of specified words across books or chapters.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        words: List of words to track
        unit: Unit for distribution ('book' or 'chapter')
        normalize: Whether to normalize frequencies by word count
        books: Optional list of books to include
        
    Returns:
        DataFrame with units as rows and words as columns, containing frequencies
    
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> divine_names = ["god", "lord", "jehovah", "jesus", "christ"]
        >>> distribution = frequency_distribution(bible, divine_names, unit='book')
    """
    # Determine which books to include
    if books:
        filtered_books = {book: bible_dict[book] for book in books if book in bible_dict}
    else:
        filtered_books = bible_dict
    
    # Initialize results structure based on unit
    results = []
    
    if unit.lower() == 'book':
        # Process each book
        for book, chapters in filtered_books.items():
            # Extract all text from the book
            book_text = _extract_text(bible_dict, book)
            book_tokens = _preprocess_text(book_text, remove_stopwords=False, lowercase=True)
            
            # Count total words for normalization
            total_words = len(book_tokens)
            
            # Count occurrences of each target word
            word_counts = {}
            for word in words:
                count = book_tokens.count(word.lower())
                if normalize and total_words > 0:
                    word_counts[word] = count / total_words * 1000  # Per 1000 words
                else:
                    word_counts[word] = count
            
            # Add to results
            word_counts['unit'] = book
            word_counts['total_words'] = total_words
            results.append(word_counts)
            
    elif unit.lower() == 'chapter':
        # Process each chapter in each book
        for book, chapters in filtered_books.items():
            for chapter_num, verses in chapters.items():
                # Extract chapter text
                chapter_text = _extract_text(bible_dict, book, chapter_num)
                chapter_tokens = _preprocess_text(chapter_text, remove_stopwords=False, lowercase=True)
                
                # Count total words for normalization
                total_words = len(chapter_tokens)
                
                # Count occurrences of each target word
                word_counts = {}
                for word in words:
                    count = chapter_tokens.count(word.lower())
                    if normalize and total_words > 0:
                        word_counts[word] = count / total_words * 1000  # Per 1000 words
                    else:
                        word_counts[word] = count
                
                # Add to results
                word_counts['unit'] = f"{book} {chapter_num}"
                word_counts['book'] = book
                word_counts['chapter'] = chapter_num
                word_counts['total_words'] = total_words
                results.append(word_counts)
    else:
        raise ValueError(f"Unit must be 'book' or 'chapter', got '{unit}'")
    
    # Convert to DataFrame
    if results:
        df = pd.DataFrame(results)
        # Set unit as index
        if 'unit' in df.columns:
            df = df.set_index('unit')
    else:
        # Return empty DataFrame with expected columns
        df = pd.DataFrame(columns=['unit'] + words + ['total_words']).set_index('unit')
    
    return df


def relative_frequency(bible_dict: Dict[str, Any], word: str, 
                      books: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate the relative frequency of a word across all books.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        word: Word to analyze
        books: Optional list of books to include
        
    Returns:
        DataFrame with book names and relative frequencies
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> god_freq = relative_frequency(bible, "god")
    """
    # Determine which books to include
    if books:
        filtered_books = {book: bible_dict[book] for book in books if book in bible_dict}
    else:
        filtered_books = bible_dict
    
    results = []
    
    # Calculate word frequency for each book
    for book, chapters in filtered_books.items():
        book_text = _extract_text(bible_dict, book)
        tokens = _preprocess_text(book_text, remove_stopwords=False, lowercase=True)
        
        total_words = len(tokens)
        word_count = tokens.count(word.lower())
        
        if total_words > 0:
            rel_frequency = word_count / total_words * 1000  # Per 1000 words
        else:
            rel_frequency = 0
            
        results.append({
            'book': book,
            'word_count': word_count,
            'total_words': total_words,
            'relative_frequency': rel_frequency
        })
    
    # Convert to DataFrame
    if results:
        df = pd.DataFrame(results)
        # Sort by relative frequency (descending)
        df = df.sort_values('relative_frequency', ascending=False)
    else:
        df = pd.DataFrame(columns=['book', 'word_count', 'total_words', 'relative_frequency'])
    
    return df


def comparative_frequency(bible_dict: Dict[str, Any], word1: str, word2: str,
                         books: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compare the relative frequencies of two words across books.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        word1: First word to analyze
        word2: Second word to analyze
        books: Optional list of books to include
        
    Returns:
        DataFrame with comparative frequencies of both words
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> comparison = comparative_frequency(bible, "god", "lord")
    """
    # Get relative frequencies for each word
    word1_freq = relative_frequency(bible_dict, word1, books)
    word2_freq = relative_frequency(bible_dict, word2, books)
    
    # Merge the data frames
    df = pd.merge(
        word1_freq[['book', 'relative_frequency']], 
        word2_freq[['book', 'relative_frequency']], 
        on='book', 
        suffixes=(f'_{word1}', f'_{word2}')
    )
    
    # Add ratio column
    df[f'ratio_{word1}_to_{word2}'] = df[f'relative_frequency_{word1}'] / df[f'relative_frequency_{word2}'].replace(0, np.nan)
    
    # Add difference column
    df[f'diff_{word1}_minus_{word2}'] = df[f'relative_frequency_{word1}'] - df[f'relative_frequency_{word2}']
    
    # Sort by ratio (descending)
    df = df.sort_values(f'ratio_{word1}_to_{word2}', ascending=False)
    
    return df
