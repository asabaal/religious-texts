"""
Concordance Module

This module provides functions for generating concordances and keyword-in-context
analysis for biblical texts.
"""

import re
from collections import defaultdict, namedtuple
from typing import Dict, List, Optional, Union, Any, Tuple, Set

import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize

# Named tuple for storing verse references
VerseRef = namedtuple('VerseRef', ['book', 'chapter', 'verse', 'text'])


def generate_concordance(bible_dict: Dict[str, Any], word: str, 
                       ignore_case: bool = True, context_words: int = 0,
                       books: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """
    Generate a concordance for a specific word in the biblical text.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        word: Word to generate concordance for
        ignore_case: Whether to ignore case when matching words
        context_words: Number of words of context to include before/after the target word
        books: Optional list of books to include in the search
        
    Returns:
        Dictionary mapping book names to lists of verse references containing the word
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> mercy_refs = generate_concordance(bible, "mercy")
        >>> for book, verses in mercy_refs.items():
        ...     print(f"{book}: {len(verses)} occurrences")
    """
    concordance = defaultdict(list)
    
    # Process search term
    if ignore_case:
        word = word.lower()
        pattern = r'\b' + re.escape(word) + r'\b'
        matcher = lambda text: re.finditer(pattern, text.lower())
    else:
        pattern = r'\b' + re.escape(word) + r'\b'
        matcher = lambda text: re.finditer(pattern, text)
    
    # Determine which books to include
    if books:
        filtered_books = {book: bible_dict[book] for book in books if book in bible_dict}
    else:
        filtered_books = bible_dict
    
    # Search through the text
    for book, chapters in filtered_books.items():
        for chapter_num, verses in chapters.items():
            for verse_num, verse_text in verses.items():
                # Check if the verse contains the word
                matches = list(matcher(verse_text))
                
                if matches:
                    if context_words > 0:
                        # For context, process each match
                        for match in matches:
                            # Get match position
                            match_start = match.start()
                            match_end = match.end()
                            
                            # Tokenize the verse
                            tokens = word_tokenize(verse_text)
                            
                            # Find which token(s) contain the match
                            char_pos = 0
                            match_token_indices = []
                            
                            for i, token in enumerate(tokens):
                                token_start = verse_text.find(token, char_pos)
                                token_end = token_start + len(token)
                                
                                # Check if token overlaps with match
                                if token_end > match_start and token_start < match_end:
                                    match_token_indices.append(i)
                                
                                char_pos = token_end
                            
                            if match_token_indices:
                                # Get context words
                                start_idx = max(0, match_token_indices[0] - context_words)
                                end_idx = min(len(tokens), match_token_indices[-1] + context_words + 1)
                                
                                # Extract context
                                context = " ".join(tokens[start_idx:end_idx])
                                
                                # Add to concordance
                                result = f"{chapter_num}:{verse_num} {context}"
                                concordance[book].append(result)
                    else:
                        # Without context, just add the full verse
                        result = f"{chapter_num}:{verse_num} {verse_text}"
                        concordance[book].append(result)
    
    return dict(concordance)


def keyword_in_context(bible_dict: Dict[str, Any], word: str, 
                      context_size: int = 5, ignore_case: bool = True,
                      books: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Generate a keyword-in-context (KWIC) view for a specific word.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        word: Word to analyze
        context_size: Number of words to include as context before and after
        ignore_case: Whether to ignore case when matching words
        books: Optional list of books to include in the search
        
    Returns:
        DataFrame with columns for reference, left context, keyword, right context
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> kwic = keyword_in_context(bible, "beginning", context_size=4)
        >>> print(kwic.head())
    """
    results = []
    
    # Process search term
    if ignore_case:
        search_pattern = r'\b' + re.escape(word.lower()) + r'\b'
        
        def matcher(text):
            return [(m.start(), m.end()) for m in re.finditer(search_pattern, text.lower())]
    else:
        search_pattern = r'\b' + re.escape(word) + r'\b'
        
        def matcher(text):
            return [(m.start(), m.end()) for m in re.finditer(search_pattern, text)]
    
    # Determine which books to include
    if books:
        filtered_books = {book: bible_dict[book] for book in books if book in bible_dict}
    else:
        filtered_books = bible_dict
    
    # Search through the text
    for book, chapters in filtered_books.items():
        for chapter_num, verses in chapters.items():
            for verse_num, verse_text in verses.items():
                # Check if the verse contains the word
                matches = matcher(verse_text)
                
                for match_start, match_end in matches:
                    # Tokenize the verse
                    tokens = word_tokenize(verse_text)
                    
                    # Find token that contains the match
                    char_pos = 0
                    match_token_idx = -1
                    match_token = ""
                    
                    for i, token in enumerate(tokens):
                        token_start = verse_text.find(token, char_pos)
                        token_end = token_start + len(token)
                        
                        # Check if token contains the match
                        if token_start <= match_start and token_end >= match_end:
                            match_token_idx = i
                            match_token = token
                            break
                            
                        # Or if match spans multiple tokens
                        if token_end > match_start and token_start < match_end:
                            # Get the actual matched text from the verse
                            match_token = verse_text[match_start:match_end]
                            match_token_idx = i
                            break
                            
                        char_pos = token_end
                    
                    if match_token_idx >= 0:
                        # Get context
                        left_start = max(0, match_token_idx - context_size)
                        left_context = " ".join(tokens[left_start:match_token_idx])
                        
                        right_end = min(len(tokens), match_token_idx + context_size + 1)
                        right_tokens = tokens[match_token_idx+1:right_end]
                        right_context = " ".join(right_tokens) if right_tokens else ""
                        
                        # Build reference
                        reference = f"{book} {chapter_num}:{verse_num}"
                        
                        results.append({
                            'reference': reference,
                            'left_context': left_context,
                            'keyword': match_token,
                            'right_context': right_context,
                            'book': book,
                            'chapter': chapter_num,
                            'verse': verse_num
                        })
    
    # Convert results to DataFrame
    if results:
        df = pd.DataFrame(results)
        # Reorder columns for display
        display_cols = ['reference', 'left_context', 'keyword', 'right_context', 
                        'book', 'chapter', 'verse']
        df = df[display_cols]
    else:
        # Create empty DataFrame with expected columns
        df = pd.DataFrame(columns=['reference', 'left_context', 'keyword', 'right_context', 
                                 'book', 'chapter', 'verse'])
    
    return df


def find_all_occurrences(bible_dict: Dict[str, Any], term: str, 
                        ignore_case: bool = True, whole_word: bool = True,
                        books: Optional[List[str]] = None) -> List[VerseRef]:
    """
    Find all occurrences of a term in the biblical text.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        term: Term to search for (word or phrase)
        ignore_case: Whether to ignore case when matching
        whole_word: Whether to match whole words only
        books: Optional list of books to include in the search
        
    Returns:
        List of VerseRef namedtuples with book, chapter, verse, and text
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> refs = find_all_occurrences(bible, "in the beginning")
        >>> for ref in refs:
        ...     print(f"{ref.book} {ref.chapter}:{ref.verse}")
    """
    results = []
    
    # Build search pattern
    if whole_word:
        if ignore_case:
            pattern = r'\b' + re.escape(term) + r'\b'
            matcher = lambda text: bool(re.search(pattern, text, re.IGNORECASE))
        else:
            pattern = r'\b' + re.escape(term) + r'\b'
            matcher = lambda text: bool(re.search(pattern, text))
    else:
        if ignore_case:
            matcher = lambda text: term.lower() in text.lower()
        else:
            matcher = lambda text: term in text
    
    # Determine which books to include
    if books:
        filtered_books = {book: bible_dict[book] for book in books if book in bible_dict}
    else:
        filtered_books = bible_dict
    
    # Search through the text
    for book, chapters in filtered_books.items():
        for chapter_num, verses in chapters.items():
            for verse_num, verse_text in verses.items():
                # Check if verse contains the term
                if matcher(verse_text):
                    # Create VerseRef object
                    verse_ref = VerseRef(
                        book=book,
                        chapter=chapter_num,
                        verse=verse_num,
                        text=verse_text
                    )
                    results.append(verse_ref)
    
    return results


def find_verses_containing_all(bible_dict: Dict[str, Any], terms: List[str], 
                             max_distance: Optional[int] = None,
                             ignore_case: bool = True,
                             books: Optional[List[str]] = None) -> List[VerseRef]:
    """
    Find verses containing all specified terms.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        terms: List of terms to search for
        max_distance: Optional maximum word distance between terms
        ignore_case: Whether to ignore case when matching
        books: Optional list of books to include in the search
        
    Returns:
        List of VerseRef namedtuples with book, chapter, verse, and text
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> refs = find_verses_containing_all(bible, ["love", "neighbor"])
        >>> for ref in refs:
        ...     print(f"{ref.book} {ref.chapter}:{ref.verse}")
    """
    if not terms:
        return []
    
    results = []
    
    # Prepare search terms
    if ignore_case:
        terms = [term.lower() for term in terms]
        contains_all = lambda text: all(term in text.lower() for term in terms)
    else:
        contains_all = lambda text: all(term in text for term in terms)
    
    # Determine which books to include
    if books:
        filtered_books = {book: bible_dict[book] for book in books if book in bible_dict}
    else:
        filtered_books = bible_dict
    
    # Search through the text
    for book, chapters in filtered_books.items():
        for chapter_num, verses in chapters.items():
            for verse_num, verse_text in verses.items():
                # Check if all terms are present
                if contains_all(verse_text):
                    # Check distance constraint if specified
                    if max_distance is not None:
                        # Tokenize the verse
                        tokens = word_tokenize(verse_text.lower() if ignore_case else verse_text)
                        
                        # Find positions of all terms
                        term_positions = {}
                        for term in terms:
                            positions = []
                            for i, token in enumerate(tokens):
                                if term in token.lower() if ignore_case else term in token:
                                    positions.append(i)
                            term_positions[term] = positions
                        
                        # Check if terms are within max_distance of each other
                        if all(term_positions[term] for term in terms):
                            # Check each combination of term positions
                            all_close = False
                            
                            # Get all position combinations
                            positions_list = [term_positions[term] for term in terms]
                            
                            from itertools import product
                            for positions in product(*positions_list):
                                min_pos = min(positions)
                                max_pos = max(positions)
                                
                                if max_pos - min_pos <= max_distance:
                                    all_close = True
                                    break
                            
                            if not all_close:
                                continue
                    
                    # Create VerseRef object
                    verse_ref = VerseRef(
                        book=book,
                        chapter=chapter_num,
                        verse=verse_num,
                        text=verse_text
                    )
                    results.append(verse_ref)
    
    return results
