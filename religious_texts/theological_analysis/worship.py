"""
Worship Language Analysis Module

This module provides functions for identifying and analyzing worship contexts
and language patterns in biblical texts, including prayers, songs, and
ritual language.
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union, Any, Tuple, Set

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize

# Worship terms and patterns
WORSHIP_TERMS = {
    # Worship actions
    'praise': {'praise', 'glorify', 'exalt', 'magnify', 'extol', 'honor', 'bless', 'adore'},
    'prayer': {'pray', 'prayer', 'supplication', 'petition', 'intercession', 'entreat', 'beseech'},
    'sacrifice': {'sacrifice', 'offering', 'oblation', 'altar', 'burnt offering', 'sin offering'},
    'ritual': {'wash', 'cleanse', 'purify', 'anoint', 'consecrate', 'dedicate', 'sanctify'},
    
    # Worship expressions
    'thanksgiving': {'thank', 'thanks', 'thanksgiving', 'grateful', 'gratitude', 'appreciate'},
    'confession': {'confess', 'confession', 'acknowledge', 'admit', 'repent'},
    'submission': {'submit', 'surrender', 'yield', 'bow', 'kneel', 'prostrate', 'humble'},
    
    # Worship contexts
    'temple': {'temple', 'sanctuary', 'tabernacle', 'holy place', 'holy of holies', 'court'},
    'music': {'sing', 'song', 'psalm', 'hymn', 'music', 'melody', 'harp', 'lyre', 'trumpet', 'cymbal'},
    'assembly': {'assembly', 'congregation', 'gather', 'assemble', 'festival', 'feast', 'sabbath'}
}

# Prayer patterns
PRAYER_PATTERNS = {
    'invocation': {'o god', 'o lord', 'hear my prayer', 'hear me', 'listen to', 'attend unto'},
    'petition': {'give', 'grant', 'bestow', 'provide', 'help', 'save', 'deliver'},
    'intercession': {'pray for', 'intercede', 'beseech thee for', 'entreat for'},
    'thanksgiving': {'thank', 'thanks', 'thanksgiving', 'grateful', 'gratitude', 'praise'},
    'praise': {'praise', 'bless', 'blessed be', 'glory', 'honor', 'magnify', 'exalt'},
    'confession': {'forgive', 'confess', 'sinned', 'transgressed', 'iniquity', 'repent'},
    'lament': {'why', 'how long', 'forsaken', 'forgotten', 'far from', 'cry'}
}

# Worship books - Psalms, some prophetic books, and other books with high worship content
WORSHIP_BOOKS = {
    'psalms', 'song of solomon', 'lamentations',  # Poetic books
    'isaiah', 'jeremiah', 'ezekiel',              # Major prophets
    'revelation',                                 # Apocalyptic
    'exodus', 'leviticus',                        # Torah (ritual portions)
    'chronicles', 'ezra', 'nehemiah'              # Historical (temple focus)
}


def identify_worship_contexts(bible_dict: Dict[str, Any], book: Optional[str] = None,
                            chapter: Optional[int] = None, verse: Optional[int] = None,
                            categories: Optional[List[str]] = None,
                            threshold: int = 2) -> pd.DataFrame:
    """
    Identify passages likely to contain worship contexts based on worship terminology.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        book: Optional book name to filter by
        chapter: Optional chapter number to filter by
        verse: Optional verse number to filter by
        categories: Optional list of worship term categories to include
        threshold: Minimum number of worship terms required to identify a worship context
        
    Returns:
        DataFrame with identified worship contexts
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Identify worship contexts in Psalms
        >>> worship_df = identify_worship_contexts(bible, book="Psalms")
    """
    results = []
    
    # Determine which categories to include
    if categories:
        categories_to_check = {cat: WORSHIP_TERMS[cat] for cat in categories if cat in WORSHIP_TERMS}
    else:
        categories_to_check = WORSHIP_TERMS
    
    # Combine all terms into a single set for efficient checking
    all_terms = set()
    for term_set in categories_to_check.values():
        all_terms.update(term_set)
    
    # Prepare regex pattern for all worship terms
    term_patterns = []
    for term in all_terms:
        # Escape special regex characters and convert to word boundary pattern
        pattern = r'\b' + re.escape(term) + r'\b'
        term_patterns.append(pattern)
    
    # Combine all patterns
    worship_pattern = re.compile('|'.join(term_patterns), re.IGNORECASE)
    
    # Determine which books to include
    if book:
        if book not in bible_dict:
            return pd.DataFrame()
        books_to_check = {book: bible_dict[book]}
    else:
        books_to_check = bible_dict
    
    # Process each verse
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
                # Find all matching worship terms
                matches = worship_pattern.findall(verse_text)
                
                # Skip if below threshold
                if len(matches) < threshold:
                    continue
                
                # Identify which categories are present
                found_categories = {}
                for cat, terms in categories_to_check.items():
                    cat_matches = [match for match in matches if match.lower() in terms or 
                                  any(term in verse_text.lower() for term in terms)]
                    if cat_matches:
                        found_categories[cat] = cat_matches
                
                # Calculate contextual indicators
                # 1. Is the book a known worship-focused book?
                in_worship_book = book_name.lower() in WORSHIP_BOOKS
                
                # 2. Density of worship terms
                term_density = len(matches) / len(word_tokenize(verse_text))
                
                # 3. Variety of categories
                category_variety = len(found_categories)
                
                # Create combined confidence score (simple formula, can be refined)
                confidence = term_density * 10 + category_variety + (1 if in_worship_book else 0)
                confidence = min(1.0, confidence / 10)  # Normalize to 0-1
                
                # Add to results
                result = {
                    'book': book_name,
                    'chapter': chapter_num,
                    'verse': verse_num,
                    'reference': f"{book_name} {chapter_num}:{verse_num}",
                    'text': verse_text,
                    'worship_term_count': len(matches),
                    'term_density': term_density,
                    'category_variety': category_variety,
                    'in_worship_book': in_worship_book,
                    'confidence': confidence,
                    'categories': ', '.join(found_categories.keys()),
                    'matched_terms': ', '.join(matches)
                }
                
                results.append(result)
    
    # Convert results to DataFrame
    if results:
        df = pd.DataFrame(results)
        
        # Sort by confidence (descending)
        df = df.sort_values('confidence', ascending=False)
    else:
        # Create empty DataFrame with expected columns
        df = pd.DataFrame(columns=['book', 'chapter', 'verse', 'reference', 'text',
                                 'worship_term_count', 'term_density', 'category_variety',
                                 'in_worship_book', 'confidence', 'categories', 'matched_terms'])
    
    return df


def worship_language_analysis(bible_dict: Dict[str, Any], book: Optional[str] = None,
                            unit: str = 'book', normalize: bool = True) -> pd.DataFrame:
    """
    Analyze the distribution of worship language across books or chapters.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        book: Optional book name to filter by
        unit: Unit for analysis ('book' or 'chapter')
        normalize: Whether to normalize counts by text length
        
    Returns:
        DataFrame with worship language distribution
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Analyze worship language across all books
        >>> worship_dist = worship_language_analysis(bible)
    """
    # Combine all worship terms for efficient checking
    all_terms = {}
    for category, terms in WORSHIP_TERMS.items():
        for term in terms:
            all_terms[term] = category
    
    # Prepare regex pattern for all worship terms
    term_patterns = []
    for term in all_terms.keys():
        # Escape special regex characters and convert to word boundary pattern
        pattern = r'\b' + re.escape(term) + r'\b'
        term_patterns.append(pattern)
    
    # Combine all patterns
    worship_pattern = re.compile('|'.join(term_patterns), re.IGNORECASE)
    
    # Initialize results
    results = []
    
    # Determine which books to include
    if book:
        if book not in bible_dict:
            return pd.DataFrame()
        books_to_check = {book: bible_dict[book]}
    else:
        books_to_check = bible_dict
    
    # Process based on unit
    if unit == 'book':
        for book_name, chapters in books_to_check.items():
            # Initialize category counts
            category_counts = {category: 0 for category in WORSHIP_TERMS}
            
            # Combine all text in the book
            book_text = ' '.join(
                verse_text 
                for chapter_verses in chapters.values() 
                for verse_text in chapter_verses.values()
            )
            
            # Count total words for normalization
            total_words = len(word_tokenize(book_text))
            
            # Find all worship terms
            matches = worship_pattern.findall(book_text.lower())
            
            # Count by category
            for match in matches:
                # Find which category this term belongs to
                found_category = None
                for term, category in all_terms.items():
                    if match.lower() == term.lower() or term.lower() in match.lower():
                        found_category = category
                        break
                
                if found_category:
                    category_counts[found_category] += 1
            
            # Create result
            result = {
                'book': book_name,
                'total_worship_terms': sum(category_counts.values()),
                'total_words': total_words
            }
            
            # Add category counts
            for category, count in category_counts.items():
                result[category] = count
                if normalize and total_words > 0:
                    result[f"{category}_normalized"] = count / total_words * 1000
            
            # Calculate overall worship density
            if normalize and total_words > 0:
                result['worship_density'] = result['total_worship_terms'] / total_words * 1000
            
            results.append(result)
    
    elif unit == 'chapter':
        for book_name, chapters in books_to_check.items():
            for chapter_num, verses in chapters.items():
                # Initialize category counts
                category_counts = {category: 0 for category in WORSHIP_TERMS}
                
                # Combine all text in the chapter
                chapter_text = ' '.join(verse_text for verse_text in verses.values())
                
                # Count total words for normalization
                total_words = len(word_tokenize(chapter_text))
                
                # Find all worship terms
                matches = worship_pattern.findall(chapter_text.lower())
                
                # Count by category
                for match in matches:
                    # Find which category this term belongs to
                    found_category = None
                    for term, category in all_terms.items():
                        if match.lower() == term.lower() or term.lower() in match.lower():
                            found_category = category
                            break
                    
                    if found_category:
                        category_counts[found_category] += 1
                
                # Create result
                result = {
                    'book': book_name,
                    'chapter': chapter_num,
                    'reference': f"{book_name} {chapter_num}",
                    'total_worship_terms': sum(category_counts.values()),
                    'total_words': total_words
                }
                
                # Add category counts
                for category, count in category_counts.items():
                    result[category] = count
                    if normalize and total_words > 0:
                        result[f"{category}_normalized"] = count / total_words * 1000
                
                # Calculate overall worship density
                if normalize and total_words > 0:
                    result['worship_density'] = result['total_worship_terms'] / total_words * 1000
                
                results.append(result)
    
    else:
        raise ValueError(f"Unknown unit '{unit}'. Choose from 'book' or 'chapter'.")
    
    # Convert results to DataFrame
    if results:
        df = pd.DataFrame(results)
        
        # Sort by worship density (descending)
        if normalize:
            df = df.sort_values('worship_density', ascending=False)
        else:
            df = df.sort_values('total_worship_terms', ascending=False)
    else:
        # Create empty DataFrame with expected columns
        base_cols = ['book']
        if unit == 'chapter':
            base_cols.extend(['chapter', 'reference'])
        
        data_cols = ['total_worship_terms', 'total_words']
        if normalize:
            data_cols.append('worship_density')
        
        category_cols = list(WORSHIP_TERMS.keys())
        if normalize:
            category_cols.extend([f"{cat}_normalized" for cat in WORSHIP_TERMS])
        
        df = pd.DataFrame(columns=base_cols + data_cols + category_cols)
    
    return df


def worship_term_distribution(bible_dict: Dict[str, Any], terms: List[str],
                            books: Optional[List[str]] = None,
                            normalize: bool = True) -> pd.DataFrame:
    """
    Analyze the distribution of specific worship terms across books.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        terms: List of worship terms to analyze
        books: Optional list of books to include
        normalize: Whether to normalize counts by text length
        
    Returns:
        DataFrame with term distribution
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Analyze distribution of praise terms
        >>> praise_terms = list(WORSHIP_TERMS['praise'])
        >>> praise_dist = worship_term_distribution(bible, praise_terms)
    """
    results = []
    
    # Prepare regex patterns for terms
    term_patterns = {}
    for term in terms:
        # Escape special regex characters and convert to word boundary pattern
        pattern = r'\b' + re.escape(term) + r'\b'
        term_patterns[term] = re.compile(pattern, re.IGNORECASE)
    
    # Determine which books to include
    if books:
        books_to_check = {book: bible_dict[book] for book in books if book in bible_dict}
    else:
        books_to_check = bible_dict
    
    # Process each book
    for book_name, chapters in books_to_check.items():
        # Combine all text in the book
        book_text = ' '.join(
            verse_text 
            for chapter_verses in chapters.values() 
            for verse_text in chapter_verses.values()
        )
        
        # Count total words for normalization
        total_words = len(word_tokenize(book_text))
        
        # Initialize result
        result = {
            'book': book_name,
            'total_words': total_words
        }
        
        # Count occurrences of each term
        total_count = 0
        for term, pattern in term_patterns.items():
            matches = pattern.findall(book_text)
            count = len(matches)
            
            result[term] = count
            total_count += count
            
            if normalize and total_words > 0:
                result[f"{term}_normalized"] = count / total_words * 1000
        
        result['total_term_count'] = total_count
        
        if normalize and total_words > 0:
            result['term_density'] = total_count / total_words * 1000
        
        results.append(result)
    
    # Convert results to DataFrame
    if results:
        df = pd.DataFrame(results)
        
        # Sort by term density (descending)
        if normalize:
            df = df.sort_values('term_density', ascending=False)
        else:
            df = df.sort_values('total_term_count', ascending=False)
    else:
        # Create empty DataFrame with expected columns
        base_cols = ['book', 'total_words', 'total_term_count']
        if normalize:
            base_cols.append('term_density')
        
        term_cols = list(terms)
        if normalize:
            term_cols.extend([f"{term}_normalized" for term in terms])
        
        df = pd.DataFrame(columns=base_cols + term_cols)
    
    return df


def prayer_pattern_analysis(bible_dict: Dict[str, Any], book: Optional[str] = None,
                          chapter: Optional[int] = None, verse: Optional[int] = None) -> pd.DataFrame:
    """
    Analyze prayer patterns and structures in biblical texts.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        book: Optional book name to filter by
        chapter: Optional chapter number to filter by
        verse: Optional verse number to filter by
        
    Returns:
        DataFrame with prayer pattern analysis
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Analyze prayer patterns in Psalms
        >>> prayer_analysis = prayer_pattern_analysis(bible, book="Psalms")
    """
    # First identify potential prayers
    # We'll use a broader definition of prayer to catch more examples
    
    # Combine all prayer indicators
    prayer_indicators = set()
    for pattern_set in PRAYER_PATTERNS.values():
        prayer_indicators.update(pattern_set)
    
    # Add basic prayer verbs and nouns
    prayer_verbs = {'pray', 'entreat', 'beseech', 'plead', 'implore', 'petition', 'cry', 'call'}
    prayer_nouns = {'prayer', 'supplication', 'petition', 'request', 'intercession'}
    prayer_indicators.update(prayer_verbs)
    prayer_indicators.update(prayer_nouns)
    
    # Compile regex for prayer indicators
    indicator_patterns = []
    for indicator in prayer_indicators:
        pattern = r'\b' + re.escape(indicator) + r'\b'
        indicator_patterns.append(pattern)
    
    prayer_indicator_pattern = re.compile('|'.join(indicator_patterns), re.IGNORECASE)
    
    # Compile regex patterns for invocations (often at beginning of prayers)
    invocation_patterns = []
    for invocation in PRAYER_PATTERNS['invocation']:
        pattern = r'\b' + re.escape(invocation) + r'\b'
        invocation_patterns.append(pattern)
    
    invocation_pattern = re.compile('|'.join(invocation_patterns), re.IGNORECASE)
    
    # Identify prayers and analyze their structures
    prayers = []
    
    # Determine which books to include
    if book:
        if book not in bible_dict:
            return pd.DataFrame()
        books_to_check = {book: bible_dict[book]}
    else:
        books_to_check = bible_dict
    
    # Process each verse
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
            
            # Check each verse
            for verse_num, verse_text in verses_to_check.items():
                # Check for prayer indicators
                indicator_matches = prayer_indicator_pattern.findall(verse_text.lower())
                invocation_matches = invocation_pattern.findall(verse_text.lower())
                
                # Skip if no prayer indicators found
                if not (indicator_matches or invocation_matches):
                    continue
                
                # Analyze prayer patterns
                patterns = {}
                for pattern, terms in PRAYER_PATTERNS.items():
                    pattern_matches = []
                    for term in terms:
                        if term.lower() in verse_text.lower():
                            pattern_matches.append(term)
                    
                    if pattern_matches:
                        patterns[pattern] = pattern_matches
                
                # Skip if no specific prayer patterns found
                if not patterns:
                    continue
                
                # Create prayer record
                prayer = {
                    'book': book_name,
                    'chapter': chapter_num,
                    'verse': verse_num,
                    'reference': f"{book_name} {chapter_num}:{verse_num}",
                    'text': verse_text,
                    'prayer_indicators': ', '.join(indicator_matches),
                    'pattern_count': len(patterns),
                    'patterns': ', '.join(patterns.keys()),
                    'word_count': len(word_tokenize(verse_text)),
                }
                
                # Add pattern-specific flags
                for pattern in PRAYER_PATTERNS:
                    prayer[f"has_{pattern}"] = pattern in patterns
                
                prayers.append(prayer)
    
    # Convert to DataFrame
    if prayers:
        df = pd.DataFrame(prayers)
        
        # Sort by pattern count (most complex prayers first)
        df = df.sort_values('pattern_count', ascending=False)
    else:
        # Create empty DataFrame with expected columns
        base_cols = ['book', 'chapter', 'verse', 'reference', 'text', 
                   'prayer_indicators', 'pattern_count', 'patterns', 'word_count']
        
        pattern_cols = [f"has_{pattern}" for pattern in PRAYER_PATTERNS]
        
        df = pd.DataFrame(columns=base_cols + pattern_cols)
    
    return df
