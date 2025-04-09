"""
Authority Language Analysis Module

This module provides functions for analyzing authority claims and patterns
in biblical texts, including commands, delegated authority, and hierarchical
language.
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union, Any, Tuple, Set

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize

# Authority language markers
AUTHORITY_TERMS = {
    # Command verbs
    'command': {'command', 'charge', 'order', 'instruct', 'direct', 'decree', 'ordain'},
    'prohibition': {'forbid', 'prohibit', 'thou shalt not', 'do not', 'shall not', 'must not'},
    'obligation': {'must', 'shall', 'ought', 'obliged', 'bound', 'required', 'necessary'},
    
    # Authority language
    'declare': {'declare', 'proclaim', 'pronounce', 'announce', 'say', 'speak', 'utter'},
    'authority': {'authority', 'power', 'right', 'dominion', 'rule', 'throne', 'scepter'},
    'judgment': {'judge', 'judgment', 'condemn', 'sentence', 'punishment', 'reward'},
    
    # Delegation language
    'delegate': {'appoint', 'assign', 'commission', 'delegate', 'authorize', 'empower'},
    'send': {'send', 'sent', 'go', 'commissioned', 'ambassador', 'messenger', 'apostle'},
    'receive': {'receive', 'given', 'granted', 'bestowed', 'conferred', 'imparted'}
}

# Authority figures
AUTHORITY_FIGURES = {
    # Divine authority
    'god': {'god', 'lord', 'almighty', 'creator', 'most high'},
    'jesus': {'jesus', 'christ', 'messiah', 'son of man', 'son of god'},
    'holy_spirit': {'holy spirit', 'holy ghost', 'spirit of god', 'spirit of the lord'},
    
    # Human authority
    'prophets': {
        'moses', 'isaiah', 'jeremiah', 'ezekiel', 'daniel', 'hosea', 'joel', 'amos',
        'obadiah', 'jonah', 'micah', 'nahum', 'habakkuk', 'zephaniah', 'haggai',
        'zechariah', 'malachi', 'elijah', 'elisha', 'samuel', 'nathan'
    },
    'kings': {
        'king', 'ruler', 'sovereign', 'david', 'solomon', 'saul', 'hezekiah', 'josiah',
        'pharaoh', 'caesar', 'herod'
    },
    'priests': {
        'priest', 'high priest', 'aaron', 'levite', 'cohen', 'zadok', 'abiathar',
        'caiaphas', 'religious leaders', 'scribes', 'pharisees'
    },
    'apostles': {
        'apostle', 'disciple', 'peter', 'john', 'james', 'andrew', 'philip', 'thomas',
        'matthew', 'paul', 'timothy', 'titus'
    }
}

# Modal verbs and imperative patterns
MODALS = {'shall', 'must', 'will', 'should', 'ought', 'may', 'can', 'could', 'might'}
IMPERATIVE_PATTERNS = [
    r'^([A-Z][a-z]+)',  # Word at beginning of sentence
    r'\. ([A-Z][a-z]+)',  # Word after period
    r'\! ([A-Z][a-z]+)',  # Word after exclamation
    r'\? ([A-Z][a-z]+)',  # Word after question
]


def identify_authority_claims(bible_dict: Dict[str, Any], book: Optional[str] = None,
                            chapter: Optional[int] = None, verse: Optional[int] = None,
                            categories: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Identify passages containing authority claims in biblical texts.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        book: Optional book name to filter by
        chapter: Optional chapter number to filter by
        verse: Optional verse number to filter by
        categories: Optional list of authority term categories to include
        
    Returns:
        DataFrame with identified authority claims
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Identify authority claims in Exodus
        >>> authority_df = identify_authority_claims(bible, book="Exodus")
    """
    results = []
    
    # Determine which categories to include
    if categories:
        categories_to_check = {cat: AUTHORITY_TERMS[cat] for cat in categories if cat in AUTHORITY_TERMS}
    else:
        categories_to_check = AUTHORITY_TERMS
    
    # Combine all terms into a single set for efficient checking
    all_terms = set()
    for term_set in categories_to_check.values():
        all_terms.update(term_set)
    
    # Prepare regex pattern for all authority terms
    term_patterns = []
    for term in all_terms:
        # Escape special regex characters and convert to word boundary pattern
        pattern = r'\b' + re.escape(term) + r'\b'
        term_patterns.append(pattern)
    
    # Combine all patterns
    authority_pattern = re.compile('|'.join(term_patterns), re.IGNORECASE)
    
    # Compile patterns for modal verbs and imperatives
    modal_pattern = re.compile(r'\b(' + '|'.join(MODALS) + r')\b', re.IGNORECASE)
    imperative_patterns = [re.compile(pattern) for pattern in IMPERATIVE_PATTERNS]
    
    # Authority figure detection
    # Combine all authority figures into a single dict for efficient checking
    authority_figures = {}
    for category, figures in AUTHORITY_FIGURES.items():
        for figure in figures:
            authority_figures[figure] = category
    
    # Prepare regex pattern for all authority figures
    figure_patterns = []
    for figure in authority_figures:
        # Escape special regex characters and convert to word boundary pattern
        pattern = r'\b' + re.escape(figure) + r'\b'
        figure_patterns.append(pattern)
    
    # Combine all patterns
    figure_pattern = re.compile('|'.join(figure_patterns), re.IGNORECASE)
    
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
                # Look for authority terms
                authority_matches = authority_pattern.findall(verse_text)
                
                # Look for modal verbs
                modal_matches = modal_pattern.findall(verse_text)
                
                # Look for potential imperatives
                imperative_matches = []
                for pattern in imperative_patterns:
                    matches = pattern.findall(verse_text)
                    imperative_matches.extend(matches)
                
                # Look for authority figures
                figure_matches = figure_pattern.findall(verse_text)
                
                # Skip if no authority indicators found
                if not (authority_matches or modal_matches or (len(imperative_matches) > 0 and figure_matches)):
                    continue
                
                # Find which authority categories are present
                found_categories = {}
                for cat, terms in categories_to_check.items():
                    cat_matches = [match for match in authority_matches 
                                  if match.lower() in terms or 
                                  any(term in verse_text.lower() for term in terms)]
                    if cat_matches:
                        found_categories[cat] = cat_matches
                
                # Identify authority figures
                identified_figures = []
                figure_categories = set()
                for match in figure_matches:
                    for figure, category in authority_figures.items():
                        if match.lower() == figure.lower() or figure.lower() in match.lower():
                            identified_figures.append(match)
                            figure_categories.add(category)
                            break
                
                # Calculate confidence score
                # More authority terms, modal verbs, and authority figures increase confidence
                confidence = (len(authority_matches) * 0.5 + 
                            len(modal_matches) * 0.3 + 
                            len(imperative_matches) * 0.2 + 
                            len(identified_figures) * 0.5)
                
                # Normalize to 0-1
                confidence = min(1.0, confidence / 5)
                
                # Prepare result
                result = {
                    'book': book_name,
                    'chapter': chapter_num,
                    'verse': verse_num,
                    'reference': f"{book_name} {chapter_num}:{verse_num}",
                    'text': verse_text,
                    'authority_term_count': len(authority_matches),
                    'modal_verb_count': len(modal_matches),
                    'imperative_count': len(imperative_matches),
                    'authority_figure_count': len(identified_figures),
                    'confidence': confidence,
                    'authority_categories': ', '.join(found_categories.keys()) if found_categories else None,
                    'authority_terms': ', '.join(authority_matches) if authority_matches else None,
                    'modal_verbs': ', '.join(modal_matches) if modal_matches else None,
                    'authority_figures': ', '.join(identified_figures) if identified_figures else None,
                    'figure_categories': ', '.join(figure_categories) if figure_categories else None
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
                                 'authority_term_count', 'modal_verb_count', 'imperative_count',
                                 'authority_figure_count', 'confidence', 'authority_categories',
                                 'authority_terms', 'modal_verbs', 'authority_figures',
                                 'figure_categories'])
    
    return df


def authority_language_analysis(bible_dict: Dict[str, Any], book: Optional[str] = None,
                              unit: str = 'book', normalize: bool = True) -> pd.DataFrame:
    """
    Analyze the distribution of authority language across books or chapters.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        book: Optional book name to filter by
        unit: Unit for analysis ('book' or 'chapter')
        normalize: Whether to normalize counts by text length
        
    Returns:
        DataFrame with authority language distribution
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Analyze authority language across all books
        >>> authority_dist = authority_language_analysis(bible)
    """
    # Combine all authority terms for efficient checking
    all_terms = {}
    for category, terms in AUTHORITY_TERMS.items():
        for term in terms:
            all_terms[term] = category
    
    # Prepare regex pattern for all authority terms
    term_patterns = []
    for term in all_terms:
        # Escape special regex characters and convert to word boundary pattern
        pattern = r'\b' + re.escape(term) + r'\b'
        term_patterns.append(pattern)
    
    # Combine all patterns
    authority_pattern = re.compile('|'.join(term_patterns), re.IGNORECASE)
    
    # Determine which books to include
    if book:
        if book not in bible_dict:
            return pd.DataFrame()
        books_to_check = {book: bible_dict[book]}
    else:
        books_to_check = bible_dict
    
    results = []
    
    # Process based on unit
    if unit == 'book':
        for book_name, chapters in books_to_check.items():
            # Initialize category counts
            category_counts = {category: 0 for category in AUTHORITY_TERMS}
            
            # Combine all text in the book
            book_text = ' '.join(
                verse_text 
                for chapter_verses in chapters.values() 
                for verse_text in chapter_verses.values()
            )
            
            # Count total words for normalization
            total_words = len(word_tokenize(book_text))
            
            # Find all authority terms
            matches = authority_pattern.findall(book_text.lower())
            
            # Count by category
            for match in matches:
                match_lower = match.lower()
                # Find which category this term belongs to
                for term, category in all_terms.items():
                    if match_lower == term.lower() or term.lower() in match_lower:
                        category_counts[category] += 1
                        break
            
            # Create result
            result = {
                'book': book_name,
                'total_authority_terms': sum(category_counts.values()),
                'total_words': total_words
            }
            
            # Add category counts
            for category, count in category_counts.items():
                result[category] = count
                if normalize and total_words > 0:
                    result[f"{category}_normalized"] = count / total_words * 1000
            
            # Calculate overall authority density
            if normalize and total_words > 0:
                result['authority_density'] = result['total_authority_terms'] / total_words * 1000
            
            results.append(result)
    
    elif unit == 'chapter':
        for book_name, chapters in books_to_check.items():
            for chapter_num, verses in chapters.items():
                # Initialize category counts
                category_counts = {category: 0 for category in AUTHORITY_TERMS}
                
                # Combine all text in the chapter
                chapter_text = ' '.join(verse_text for verse_text in verses.values())
                
                # Count total words for normalization
                total_words = len(word_tokenize(chapter_text))
                
                # Find all authority terms
                matches = authority_pattern.findall(chapter_text.lower())
                
                # Count by category
                for match in matches:
                    match_lower = match.lower()
                    # Find which category this term belongs to
                    for term, category in all_terms.items():
                        if match_lower == term.lower() or term.lower() in match_lower:
                            category_counts[category] += 1
                            break
                
                # Create result
                result = {
                    'book': book_name,
                    'chapter': chapter_num,
                    'reference': f"{book_name} {chapter_num}",
                    'total_authority_terms': sum(category_counts.values()),
                    'total_words': total_words
                }
                
                # Add category counts
                for category, count in category_counts.items():
                    result[category] = count
                    if normalize and total_words > 0:
                        result[f"{category}_normalized"] = count / total_words * 1000
                
                # Calculate overall authority density
                if normalize and total_words > 0:
                    result['authority_density'] = result['total_authority_terms'] / total_words * 1000
                
                results.append(result)
    
    else:
        raise ValueError(f"Unknown unit '{unit}'. Choose from 'book' or 'chapter'.")
    
    # Convert results to DataFrame
    if results:
        df = pd.DataFrame(results)
        
        # Sort by authority density (descending)
        if normalize:
            df = df.sort_values('authority_density', ascending=False)
        else:
            df = df.sort_values('total_authority_terms', ascending=False)
    else:
        # Create empty DataFrame with expected columns
        base_cols = ['book']
        if unit == 'chapter':
            base_cols.extend(['chapter', 'reference'])
        
        data_cols = ['total_authority_terms', 'total_words']
        if normalize:
            data_cols.append('authority_density')
        
        category_cols = list(AUTHORITY_TERMS.keys())
        if normalize:
            category_cols.extend([f"{cat}_normalized" for cat in AUTHORITY_TERMS])
        
        df = pd.DataFrame(columns=base_cols + data_cols + category_cols)
    
    return df


def delegation_pattern_analysis(bible_dict: Dict[str, Any], book: Optional[str] = None,
                              chapter: Optional[int] = None, verse: Optional[int] = None) -> pd.DataFrame:
    """
    Analyze patterns of authority delegation in biblical texts.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        book: Optional book name to filter by
        chapter: Optional chapter number to filter by
        verse: Optional verse number to filter by
        
    Returns:
        DataFrame with delegation pattern analysis
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Analyze delegation patterns in Matthew
        >>> delegation_df = delegation_pattern_analysis(bible, book="Matthew")
    """
    results = []
    
    # Delegation-specific terms
    delegation_terms = AUTHORITY_TERMS.get('delegate', set()) | AUTHORITY_TERMS.get('send', set()) | AUTHORITY_TERMS.get('receive', set())
    
    # Additional delegation patterns
    delegation_patterns = {
        'commission': {'commission', 'go ye', 'send forth', 'sent them', 'gave them power', 'gave them authority'},
        'instruction': {'teach them', 'command them', 'tell them', 'instruct them', 'show them'},
        'transfer': {'lay hands on', 'put my spirit upon', 'filled with', 'anointed', 'pour out'},
        'representation': {'in my name', 'as I was sent', 'on my behalf', 'ambassador', 'represent'}
    }
    
    # Combine all delegation terms and patterns
    all_delegation_terms = delegation_terms.copy()
    for terms in delegation_patterns.values():
        all_delegation_terms.update(terms)
    
    # Prepare regex pattern for all delegation terms
    term_patterns = []
    for term in all_delegation_terms:
        # Escape special regex characters and convert to word boundary pattern
        pattern = r'\b' + re.escape(term) + r'\b'
        term_patterns.append(pattern)
    
    # Combine all patterns
    delegation_pattern = re.compile('|'.join(term_patterns), re.IGNORECASE)
    
    # Identify authority giver and receiver pattern
    # This regex looks for patterns like "X gave Y power" or "X sent Y"
    giver_receiver_pattern = re.compile(r'([A-Za-z\s]+)\s+(send|sent|appoint|appointed|gave|give|commission|commissioned|delegate|delegated|authorize|authorized|empower|empowered)\s+([A-Za-z\s]+)', re.IGNORECASE)
    
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
                # Look for delegation terms
                delegation_matches = delegation_pattern.findall(verse_text)
                
                # Skip if no delegation terms found
                if not delegation_matches:
                    continue
                
                # Look for giver-receiver patterns
                giver_receiver_matches = giver_receiver_pattern.findall(verse_text)
                
                # Identify delegation patterns
                found_patterns = {}
                for pattern_name, pattern_terms in delegation_patterns.items():
                    pattern_matches = []
                    for term in pattern_terms:
                        if term.lower() in verse_text.lower():
                            pattern_matches.append(term)
                    
                    if pattern_matches:
                        found_patterns[pattern_name] = pattern_matches
                
                # Identify potential authority giver and receiver
                giver = None
                receiver = None
                action = None
                
                if giver_receiver_matches:
                    giver_match, action_match, receiver_match = giver_receiver_matches[0]
                    giver = giver_match.strip()
                    receiver = receiver_match.strip()
                    action = action_match.strip()
                
                # Categorize delegation type
                delegation_type = 'unknown'
                if found_patterns:
                    if 'commission' in found_patterns:
                        delegation_type = 'mission'
                    elif 'instruction' in found_patterns:
                        delegation_type = 'teaching'
                    elif 'transfer' in found_patterns:
                        delegation_type = 'empowerment'
                    elif 'representation' in found_patterns:
                        delegation_type = 'representation'
                
                # Categorize authority figures if possible
                giver_category = 'unknown'
                for category, figures in AUTHORITY_FIGURES.items():
                    if giver and any(figure.lower() in giver.lower() for figure in figures):
                        giver_category = category
                        break
                
                receiver_category = 'unknown'
                for category, figures in AUTHORITY_FIGURES.items():
                    if receiver and any(figure.lower() in receiver.lower() for figure in figures):
                        receiver_category = category
                        break
                
                # Add to results
                result = {
                    'book': book_name,
                    'chapter': chapter_num,
                    'verse': verse_num,
                    'reference': f"{book_name} {chapter_num}:{verse_num}",
                    'text': verse_text,
                    'delegation_term_count': len(delegation_matches),
                    'delegation_type': delegation_type,
                    'delegation_patterns': ', '.join(found_patterns.keys()) if found_patterns else None,
                    'authority_giver': giver,
                    'authority_receiver': receiver,
                    'delegation_action': action,
                    'giver_category': giver_category,
                    'receiver_category': receiver_category,
                    'matched_terms': ', '.join(delegation_matches)
                }
                
                results.append(result)
    
    # Convert results to DataFrame
    if results:
        df = pd.DataFrame(results)
        
        # Sort by delegation term count (descending)
        df = df.sort_values('delegation_term_count', ascending=False)
    else:
        # Create empty DataFrame with expected columns
        df = pd.DataFrame(columns=['book', 'chapter', 'verse', 'reference', 'text',
                                 'delegation_term_count', 'delegation_type', 'delegation_patterns',
                                 'authority_giver', 'authority_receiver', 'delegation_action',
                                 'giver_category', 'receiver_category', 'matched_terms'])
    
    return df


def command_distribution(bible_dict: Dict[str, Any], book: Optional[str] = None,
                       imperative_only: bool = False) -> pd.DataFrame:
    """
    Analyze the distribution of commands across biblical books.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        book: Optional book name to filter by
        imperative_only: Whether to only count grammatical imperatives (stricter definition)
        
    Returns:
        DataFrame with command distribution analysis
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Analyze commands in the New Testament
        >>> commands_df = command_distribution(bible, book=["Matthew", "Mark", "Luke", "John", 
        ...                                           "Acts", "Romans", "1 Corinthians", 
        ...                                           "2 Corinthians", "Galatians", "Ephesians", 
        ...                                           "Philippians", "Colossians", "1 Thessalonians", 
        ...                                           "2 Thessalonians", "1 Timothy", "2 Timothy", 
        ...                                           "Titus", "Philemon", "Hebrews", "James", 
        ...                                           "1 Peter", "2 Peter", "1 John", "2 John", 
        ...                                           "3 John", "Jude", "Revelation"])
    """
    results = []
    
    # Command verbs and terms
    command_terms = AUTHORITY_TERMS.get('command', set())
    prohibition_terms = AUTHORITY_TERMS.get('prohibition', set())
    obligation_terms = AUTHORITY_TERMS.get('obligation', set())
    
    # Combine all command-related terms
    all_command_terms = command_terms | prohibition_terms | obligation_terms
    
    # Prepare regex pattern for all command terms
    term_patterns = []
    for term in all_command_terms:
        # Escape special regex characters and convert to word boundary pattern
        pattern = r'\b' + re.escape(term) + r'\b'
        term_patterns.append(pattern)
    
    # Combine all patterns
    command_pattern = re.compile('|'.join(term_patterns), re.IGNORECASE)
    
    # Prepare regex pattern for imperative commands
    # This is a simplified approach - full parsing would be more accurate
    imperative_pattern = re.compile(r'\. ([A-Z][a-z]+)\b')  # Word after period starting with capital
    
    # Modal verbs for obligation
    modal_pattern = re.compile(r'\b(shall|must|ought)\b', re.IGNORECASE)
    
    # Negative commands
    negative_pattern = re.compile(r'\b(not|no|never)\b', re.IGNORECASE)
    
    # Determine which books to include
    if book:
        if isinstance(book, str):
            if book not in bible_dict:
                return pd.DataFrame()
            books_to_check = {book: bible_dict[book]}
        else:  # List of books
            books_to_check = {b: bible_dict[b] for b in book if b in bible_dict}
    else:
        books_to_check = bible_dict
    
    # Process each book
    for book_name, chapters in books_to_check.items():
        # Initialize counts
        command_count = 0
        imperative_count = 0
        prohibition_count = 0
        obligation_count = 0
        
        # Count total verses and words for normalization
        total_verses = 0
        total_words = 0
        
        # Process each verse
        for chapter_num, verses in chapters.items():
            for verse_num, verse_text in verses.items():
                total_verses += 1
                verse_words = len(word_tokenize(verse_text))
                total_words += verse_words
                
                # Count different command types
                command_terms = command_pattern.findall(verse_text)
                imperatives = imperative_pattern.findall(verse_text)
                modals = modal_pattern.findall(verse_text)
                negatives = negative_pattern.findall(verse_text)
                
                if imperative_only:
                    # Only count grammatical imperatives
                    if imperatives:
                        command_count += len(imperatives)
                        imperative_count += len(imperatives)
                else:
                    # Count all forms of commands
                    if command_terms:
                        command_count += len(command_terms)
                    
                    if imperatives:
                        command_count += len(imperatives)
                        imperative_count += len(imperatives)
                    
                    if modals:
                        command_count += len(modals)
                        obligation_count += len(modals)
                
                # Count prohibitions (command terms + negatives)
                if negatives and (command_terms or imperatives or modals):
                    prohibition_count += len(negatives)
        
        # Add to results
        result = {
            'book': book_name,
            'total_commands': command_count,
            'imperative_commands': imperative_count,
            'prohibitions': prohibition_count,
            'obligations': obligation_count,
            'total_verses': total_verses,
            'total_words': total_words,
            'commands_per_verse': command_count / total_verses if total_verses > 0 else 0,
            'commands_per_1000_words': command_count / total_words * 1000 if total_words > 0 else 0
        }
        
        results.append(result)
    
    # Convert results to DataFrame
    if results:
        df = pd.DataFrame(results)
        
        # Sort by commands per verse (descending)
        df = df.sort_values('commands_per_verse', ascending=False)
    else:
        # Create empty DataFrame with expected columns
        df = pd.DataFrame(columns=['book', 'total_commands', 'imperative_commands',
                                 'prohibitions', 'obligations', 'total_verses',
                                 'total_words', 'commands_per_verse', 'commands_per_1000_words'])
    
    return df
