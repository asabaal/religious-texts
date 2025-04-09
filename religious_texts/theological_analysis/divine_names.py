"""
Divine Name Analysis Module

This module provides functions for analyzing the usage patterns of divine names
in biblical texts. It supports analysis of names like God, Lord, Yahweh, Jesus,
Christ, and various divine titles and attributes.
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union, Any, Tuple, Set

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize

# Standard divine names
DIVINE_NAMES = {
    # Hebrew divine names
    'god': {'god', 'elohim', 'el', 'eloah'},
    'lord': {'lord', 'adonai', 'adon'},
    'yahweh': {'yahweh', 'jehovah', 'yah'},
    
    # Combined forms
    'lord_god': {'lord god', 'lord thy god', 'the lord god', 'the lord thy god'},
    
    # Christian divine names
    'jesus': {'jesus', 'jesus christ', 'christ jesus'},
    'christ': {'christ', 'messiah', 'anointed', 'anointed one'},
    'holy_spirit': {'holy spirit', 'holy ghost', 'spirit of god', 'spirit of the lord'}
}

# Divine titles and attributes
DIVINE_TITLES = {
    # Generic titles
    'king': {'king', 'king of kings', 'sovereign'},
    'father': {'father', 'abba', 'heavenly father'},
    'creator': {'creator', 'maker', 'one who made'},
    'savior': {'savior', 'saviour', 'redeemer', 'deliverer'},
    'judge': {'judge', 'righteous judge'},
    
    # Old Testament titles
    'most_high': {'most high', 'most high god', 'el elyon'},
    'almighty': {'almighty', 'shaddai', 'el shaddai', 'god almighty'},
    'shepherd': {'shepherd', 'my shepherd'},
    'rock': {'rock', 'my rock', 'rock of salvation'},
    'fortress': {'fortress', 'stronghold', 'strong tower'},
    
    # New Testament titles
    'lamb': {'lamb', 'lamb of god'},
    'word': {'word', 'the word', 'logos'},
    'son_of_god': {'son of god', 'only begotten'},
    'son_of_man': {'son of man'},
    'alpha_omega': {'alpha and omega', 'beginning and end', 'first and last'}
}


def divine_name_usage(bible_dict: Dict[str, Any], book: Optional[str] = None,
                    chapter: Optional[int] = None, verse: Optional[int] = None,
                    names: Optional[List[str]] = None,
                    ignore_case: bool = True) -> pd.DataFrame:
    """
    Analyze the usage of divine names in biblical texts.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        book: Optional book name to filter by
        chapter: Optional chapter number to filter by
        verse: Optional verse number to filter by
        names: Optional list of specific divine names to analyze (from DIVINE_NAMES keys)
        ignore_case: Whether to ignore case when matching names
        
    Returns:
        DataFrame with divine name usage statistics
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Analyze divine names in Genesis
        >>> names_df = divine_name_usage(bible, book="Genesis")
        >>> # Count by name
        >>> names_df.groupby('divine_name')['divine_name'].count()
    """
    results = []
    
    # Determine which names to include
    if names:
        names_to_check = {name: DIVINE_NAMES[name] for name in names if name in DIVINE_NAMES}
    else:
        names_to_check = DIVINE_NAMES
    
    # Prepare regex patterns for each name set
    name_patterns = {}
    for name_key, name_set in names_to_check.items():
        patterns = []
        for name in name_set:
            # Escape special regex characters and convert to word boundary pattern
            pattern = r'\b' + re.escape(name) + r'\b'
            patterns.append(pattern)
        
        # Combine all patterns for this name key
        name_patterns[name_key] = re.compile('|'.join(patterns), re.IGNORECASE if ignore_case else 0)
    
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
                # Check each divine name pattern
                for name_key, pattern in name_patterns.items():
                    matches = pattern.finditer(verse_text)
                    
                    for match in matches:
                        # Get matched text
                        matched_text = match.group(0)
                        
                        # Add to results
                        results.append({
                            'book': book_name,
                            'chapter': chapter_num,
                            'verse': verse_num,
                            'text': verse_text,
                            'divine_name': name_key,
                            'matched_text': matched_text,
                            'reference': f"{book_name} {chapter_num}:{verse_num}"
                        })
    
    # Convert results to DataFrame
    if results:
        df = pd.DataFrame(results)
    else:
        # Create empty DataFrame with expected columns
        df = pd.DataFrame(columns=['book', 'chapter', 'verse', 'text', 
                                 'divine_name', 'matched_text', 'reference'])
    
    return df


def divine_name_distribution(bible_dict: Dict[str, Any], names: Optional[List[str]] = None,
                           unit: str = 'book', normalize: bool = False,
                           books: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Analyze the distribution of divine names across books, chapters, or verses.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        names: Optional list of specific divine names to analyze (from DIVINE_NAMES keys)
        unit: Unit for distribution analysis ('book', 'chapter', or 'verse')
        normalize: Whether to normalize counts by text length
        books: Optional list of books to include
        
    Returns:
        DataFrame with divine name distribution across specified units
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Compare 'god' vs 'lord' usage across books
        >>> dist = divine_name_distribution(bible, names=['god', 'lord'])
    """
    # Get divine name usage data
    usage_df = divine_name_usage(bible_dict, book=books, names=names)
    
    if usage_df.empty:
        return pd.DataFrame()
    
    # Process based on unit type
    if unit == 'book':
        # Count occurrences by book and divine name
        counts = usage_df.groupby(['book', 'divine_name']).size().reset_index(name='count')
        
        # Pivot to get book rows and divine name columns
        pivot_df = counts.pivot(index='book', columns='divine_name', values='count').fillna(0)
        
        # Calculate total words per book for normalization if needed
        if normalize:
            book_word_counts = {}
            for book_name, chapters in bible_dict.items():
                if books and book_name not in books:
                    continue
                
                # Combine all text in the book
                book_text = ' '.join(
                    verse_text 
                    for chapter_verses in chapters.values() 
                    for verse_text in chapter_verses.values()
                )
                
                # Count words
                tokens = word_tokenize(book_text)
                book_word_counts[book_name] = len(tokens)
            
            # Normalize counts by book length
            for name in pivot_df.columns:
                pivot_df[f"{name}_normalized"] = pivot_df[name] / pd.Series(book_word_counts) * 1000
            
            # Sort by total normalized usage
            normalized_cols = [col for col in pivot_df.columns if col.endswith('_normalized')]
            pivot_df['total_normalized'] = pivot_df[normalized_cols].sum(axis=1)
            pivot_df = pivot_df.sort_values('total_normalized', ascending=False)
        
        else:
            # Add total column and sort by it
            pivot_df['total'] = pivot_df.sum(axis=1)
            pivot_df = pivot_df.sort_values('total', ascending=False)
        
        result_df = pivot_df
    
    elif unit == 'chapter':
        # Count occurrences by book, chapter, and divine name
        counts = usage_df.groupby(['book', 'chapter', 'divine_name']).size().reset_index(name='count')
        
        # Create chapter labels
        counts['chapter_label'] = counts['book'] + ' ' + counts['chapter'].astype(str)
        
        # Pivot to get chapter rows and divine name columns
        pivot_df = counts.pivot(index='chapter_label', columns='divine_name', values='count').fillna(0)
        
        # Calculate total words per chapter for normalization if needed
        if normalize:
            chapter_word_counts = {}
            for book_name, chapters in bible_dict.items():
                if books and book_name not in books:
                    continue
                
                for chapter_num, verses in chapters.items():
                    # Combine all text in the chapter
                    chapter_text = ' '.join(verse_text for verse_text in verses.values())
                    
                    # Count words
                    tokens = word_tokenize(chapter_text)
                    chapter_word_counts[f"{book_name} {chapter_num}"] = len(tokens)
            
            # Normalize counts by chapter length
            for name in pivot_df.columns:
                pivot_df[f"{name}_normalized"] = pivot_df[name] / pd.Series(chapter_word_counts) * 1000
            
            # Sort by total normalized usage
            normalized_cols = [col for col in pivot_df.columns if col.endswith('_normalized')]
            pivot_df['total_normalized'] = pivot_df[normalized_cols].sum(axis=1)
            pivot_df = pivot_df.sort_values('total_normalized', ascending=False)
        
        else:
            # Add total column and sort by it
            pivot_df['total'] = pivot_df.sum(axis=1)
            pivot_df = pivot_df.sort_values('total', ascending=False)
        
        result_df = pivot_df
    
    elif unit == 'verse':
        # Count occurrences by reference and divine name
        counts = usage_df.groupby(['reference', 'divine_name']).size().reset_index(name='count')
        
        # Pivot to get verse rows and divine name columns
        pivot_df = counts.pivot(index='reference', columns='divine_name', values='count').fillna(0)
        
        # Add total column and sort by it
        pivot_df['total'] = pivot_df.sum(axis=1)
        pivot_df = pivot_df.sort_values('total', ascending=False)
        
        result_df = pivot_df
    
    else:
        raise ValueError(f"Unknown unit '{unit}'. Choose from 'book', 'chapter', or 'verse'.")
    
    return result_df


def divine_name_context_analysis(bible_dict: Dict[str, Any], 
                               name: str, context_words: int = 5,
                               books: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Analyze the context around divine names to identify patterns of usage.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        name: Divine name to analyze (from DIVINE_NAMES keys)
        context_words: Number of words before/after the name to include in context
        books: Optional list of books to include
        
    Returns:
        DataFrame with context analysis results
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Analyze context around "lord" usage
        >>> context = divine_name_context_analysis(bible, name="lord")
        >>> # Find most common words following "lord"
        >>> common_following = context.groupby('following_word')['following_word'].count().sort_values(ascending=False)
    """
    # Validate name
    if name not in DIVINE_NAMES:
        raise ValueError(f"Unknown divine name '{name}'. Choose from {list(DIVINE_NAMES.keys())}")
    
    # Get all occurrences of the divine name
    usage_df = divine_name_usage(bible_dict, book=books, names=[name])
    
    if usage_df.empty:
        return pd.DataFrame()
    
    results = []
    
    # Process each occurrence
    for _, row in usage_df.iterrows():
        verse_text = row['text']
        matched_text = row['matched_text']
        
        # Tokenize text
        tokens = word_tokenize(verse_text)
        
        # Find the position of the matched text
        name_positions = []
        for i, token in enumerate(tokens):
            if token.lower() == matched_text.lower() or \
               (i < len(tokens) - 1 and (token.lower() + ' ' + tokens[i+1].lower()) == matched_text.lower()):
                name_positions.append(i)
        
        for pos in name_positions:
            # Extract context before
            start_pos = max(0, pos - context_words)
            before_context = ' '.join(tokens[start_pos:pos])
            
            # Extract context after
            end_pos = min(len(tokens), pos + context_words + 1)
            after_context = ' '.join(tokens[pos+1:end_pos])
            
            # Identify preceding and following words
            preceding_word = tokens[pos-1] if pos > 0 else ''
            following_word = tokens[pos+1] if pos < len(tokens) - 1 else ''
            
            # Add to results
            results.append({
                'book': row['book'],
                'chapter': row['chapter'],
                'verse': row['verse'],
                'reference': row['reference'],
                'divine_name': name,
                'matched_text': matched_text,
                'before_context': before_context,
                'after_context': after_context,
                'preceding_word': preceding_word,
                'following_word': following_word
            })
    
    # Convert results to DataFrame
    if results:
        df = pd.DataFrame(results)
    else:
        # Create empty DataFrame with expected columns
        df = pd.DataFrame(columns=['book', 'chapter', 'verse', 'reference', 'divine_name',
                                 'matched_text', 'before_context', 'after_context',
                                 'preceding_word', 'following_word'])
    
    return df


def divine_title_analysis(bible_dict: Dict[str, Any], book: Optional[str] = None,
                        chapter: Optional[int] = None, verse: Optional[int] = None,
                        titles: Optional[List[str]] = None,
                        ignore_case: bool = True) -> pd.DataFrame:
    """
    Analyze the usage of divine titles and attributes in biblical texts.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        book: Optional book name to filter by
        chapter: Optional chapter number to filter by
        verse: Optional verse number to filter by
        titles: Optional list of specific divine titles to analyze (from DIVINE_TITLES keys)
        ignore_case: Whether to ignore case when matching titles
        
    Returns:
        DataFrame with divine title usage statistics
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Analyze divine titles in Psalms
        >>> titles_df = divine_title_analysis(bible, book="Psalms")
        >>> # Count by title
        >>> titles_df.groupby('title_category')['title_category'].count()
    """
    results = []
    
    # Determine which titles to include
    if titles:
        titles_to_check = {title: DIVINE_TITLES[title] for title in titles if title in DIVINE_TITLES}
    else:
        titles_to_check = DIVINE_TITLES
    
    # Prepare regex patterns for each title set
    title_patterns = {}
    for title_key, title_set in titles_to_check.items():
        patterns = []
        for title in title_set:
            # Escape special regex characters and convert to word boundary pattern
            pattern = r'\b' + re.escape(title) + r'\b'
            patterns.append(pattern)
        
        # Combine all patterns for this title key
        title_patterns[title_key] = re.compile('|'.join(patterns), re.IGNORECASE if ignore_case else 0)
    
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
                # Check each divine title pattern
                for title_key, pattern in title_patterns.items():
                    matches = pattern.finditer(verse_text)
                    
                    for match in matches:
                        # Get matched text
                        matched_text = match.group(0)
                        
                        # Add to results
                        results.append({
                            'book': book_name,
                            'chapter': chapter_num,
                            'verse': verse_num,
                            'text': verse_text,
                            'title_category': title_key,
                            'matched_text': matched_text,
                            'reference': f"{book_name} {chapter_num}:{verse_num}"
                        })
    
    # Convert results to DataFrame
    if results:
        df = pd.DataFrame(results)
    else:
        # Create empty DataFrame with expected columns
        df = pd.DataFrame(columns=['book', 'chapter', 'verse', 'text', 
                                 'title_category', 'matched_text', 'reference'])
    
    return df


def divine_name_to_title_association(bible_dict: Dict[str, Any],
                                   names: Optional[List[str]] = None,
                                   titles: Optional[List[str]] = None,
                                   books: Optional[List[str]] = None,
                                   proximity: int = 20) -> pd.DataFrame:
    """
    Analyze associations between divine names and titles within the same context.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        names: Optional list of specific divine names to analyze
        titles: Optional list of specific divine titles to analyze
        books: Optional list of books to include
        proximity: Maximum number of words between name and title to count as association
        
    Returns:
        DataFrame with name-title association statistics
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Analyze how "lord" is associated with different titles
        >>> associations = divine_name_to_title_association(bible, names=['lord'])
    """
    # Get divine name and title occurrences
    names_df = divine_name_usage(bible_dict, book=books, names=names)
    titles_df = divine_title_analysis(bible_dict, book=books, titles=titles)
    
    if names_df.empty or titles_df.empty:
        return pd.DataFrame()
    
    associations = []
    
    # For each verse with divine names
    verse_groups = names_df.groupby(['book', 'chapter', 'verse'])
    
    for (book, chapter, verse), name_group in verse_groups:
        # Get titles in the same verse
        title_group = titles_df[(titles_df['book'] == book) & 
                               (titles_df['chapter'] == chapter) &
                               (titles_df['verse'] == verse)]
        
        if title_group.empty:
            continue
        
        verse_text = name_group['text'].iloc[0]
        tokens = word_tokenize(verse_text)
        
        # Find positions of divine names and titles
        name_positions = {}
        for _, name_row in name_group.iterrows():
            name = name_row['divine_name']
            matched_text = name_row['matched_text']
            
            # Find all positions
            for i, token in enumerate(tokens):
                if token.lower() == matched_text.lower() or \
                   (i < len(tokens) - 1 and (token.lower() + ' ' + tokens[i+1].lower()) == matched_text.lower()):
                    if name not in name_positions:
                        name_positions[name] = []
                    name_positions[name].append(i)
        
        title_positions = {}
        for _, title_row in title_group.iterrows():
            title = title_row['title_category']
            matched_text = title_row['matched_text']
            
            # Find all positions
            for i, token in enumerate(tokens):
                if token.lower() == matched_text.lower() or \
                   (i < len(tokens) - 1 and (token.lower() + ' ' + tokens[i+1].lower()) == matched_text.lower()):
                    if title not in title_positions:
                        title_positions[title] = []
                    title_positions[title].append(i)
        
        # Check proximity between names and titles
        for name, name_pos_list in name_positions.items():
            for title, title_pos_list in title_positions.items():
                # Find minimum distance between any name and title position
                min_distance = float('inf')
                for name_pos in name_pos_list:
                    for title_pos in title_pos_list:
                        distance = abs(name_pos - title_pos)
                        min_distance = min(min_distance, distance)
                
                # If within proximity threshold, record association
                if min_distance <= proximity:
                    associations.append({
                        'book': book,
                        'chapter': chapter,
                        'verse': verse,
                        'reference': f"{book} {chapter}:{verse}",
                        'divine_name': name,
                        'title_category': title,
                        'distance': min_distance,
                        'text': verse_text
                    })
    
    # Convert associations to DataFrame
    if associations:
        df = pd.DataFrame(associations)
    else:
        # Create empty DataFrame with expected columns
        df = pd.DataFrame(columns=['book', 'chapter', 'verse', 'reference',
                                 'divine_name', 'title_category', 'distance', 'text'])
    
    return df
