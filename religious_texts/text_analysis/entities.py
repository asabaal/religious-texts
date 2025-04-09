"""
Named Entity Recognition Module

This module provides functions for identifying people, places, and other entities
in biblical texts.
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union, Any, Tuple, Set

import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize

# Try to import optional dependencies
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False


# Common biblical people names
BIBLICAL_PEOPLE = {
    # Old Testament
    'adam', 'eve', 'cain', 'abel', 'noah', 'abraham', 'sarah', 'isaac', 'rebekah', 
    'jacob', 'leah', 'rachel', 'joseph', 'moses', 'aaron', 'joshua', 'samuel', 'saul', 
    'david', 'solomon', 'elijah', 'elisha', 'isaiah', 'jeremiah', 'ezekiel', 'daniel',
    'hosea', 'joel', 'amos', 'jonah', 'micah', 'nahum', 'habakkuk', 'zephaniah', 
    'haggai', 'zechariah', 'malachi', 'job', 'ruth', 'esther', 'ezra', 'nehemiah',
    
    # New Testament
    'jesus', 'christ', 'mary', 'joseph', 'john', 'peter', 'andrew', 'james', 
    'philip', 'bartholomew', 'matthew', 'thomas', 'thaddaeus', 'simon', 'judas', 
    'paul', 'timothy', 'titus', 'barnabas', 'luke', 'mark', 'stephen', 'philip',
    'priscilla', 'aquila', 'apollos', 'silas', 'lydia', 'martha', 'lazarus',
    'nicodemus', 'pilate', 'herod', 'caesar'
}

# Common biblical place names
BIBLICAL_PLACES = {
    # Old Testament
    'eden', 'ararat', 'babel', 'ur', 'canaan', 'egypt', 'sinai', 'jericho', 
    'bethel', 'bethlehem', 'jerusalem', 'judah', 'israel', 'samaria', 'nazareth', 
    'galilee', 'jordan', 'babylon', 'assyria', 'nineveh', 'damascus', 'tyre', 
    'sidon', 'moab', 'edom', 'philistia', 'shinar', 'goshen', 'haran', 'moriah',
    'carmel', 'lebanon', 'megiddo', 'hebron', 'beersheba', 'shechem', 'gilgal',
    
    # New Testament
    'nazareth', 'capernaum', 'bethany', 'jericho', 'samaria', 'decapolis',
    'caesarea', 'philippi', 'emmaus', 'cana', 'corinth', 'ephesus', 'athens', 
    'rome', 'antioch', 'galatia', 'macedonia', 'achaia', 'crete', 'cyprus', 'malta',
    'patmos', 'thessalonica', 'berea', 'miletus', 'troas'
}


def _extract_sentences_with_refs(bible_dict: Dict[str, Any], book: Optional[str] = None, 
                               chapter: Optional[int] = None, verse: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Helper function to extract sentences with their references.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        book: Optional book name to filter by
        chapter: Optional chapter number to filter by
        verse: Optional verse number to filter by
        
    Returns:
        List of dictionaries with sentence text and reference information
    """
    sentences = []
    
    # Determine which books to include
    if book:
        if book not in bible_dict:
            return []
        books_to_check = {book: bible_dict[book]}
    else:
        books_to_check = bible_dict
    
    # Extract sentences with references
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
                # Split into sentences
                verse_sentences = sent_tokenize(verse_text)
                
                for sentence in verse_sentences:
                    sentences.append({
                        'book': book_name,
                        'chapter': chapter_num,
                        'verse': verse_num,
                        'text': sentence,
                        'reference': f"{book_name} {chapter_num}:{verse_num}"
                    })
    
    return sentences


def extract_entities(bible_dict: Dict[str, Any], book: Optional[str] = None,
                   chapter: Optional[int] = None, verse: Optional[int] = None,
                   method: str = 'lexicon', entity_types: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Extract named entities from biblical texts.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        book: Optional book name to filter by
        chapter: Optional chapter number to filter by
        verse: Optional verse number to filter by
        method: Entity extraction method ('lexicon' or 'spacy')
        entity_types: Types of entities to extract (e.g., ['PERSON', 'GPE', 'LOC'])
                     Only used with 'spacy' method. If None, all types are included.
        
    Returns:
        DataFrame with extracted entities and their references
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Extract entities from Genesis
        >>> entities = extract_entities(bible, book="Genesis")
        >>> # Count entity occurrences
        >>> entity_counts = entities.groupby('entity')['entity'].count().sort_values(ascending=False)
    """
    if method == 'lexicon':
        # Use simple dictionary-based approach
        results = []
        
        # Extract sentences with references
        sentences = _extract_sentences_with_refs(bible_dict, book, chapter, verse)
        
        for sentence_info in sentences:
            sentence_text = sentence_info['text']
            
            # Tokenize text
            tokens = word_tokenize(sentence_text.lower())
            
            # Look for people and places
            for token in tokens:
                entity_type = None
                
                if token in BIBLICAL_PEOPLE:
                    entity_type = 'PERSON'
                elif token in BIBLICAL_PLACES:
                    entity_type = 'PLACE'
                
                if entity_type:
                    # Find original capitalization in text
                    original_token = None
                    for word in re.findall(r'\b\w+\b', sentence_text):
                        if word.lower() == token:
                            original_token = word
                            break
                    
                    if original_token:
                        results.append({
                            'entity': original_token,
                            'entity_type': entity_type,
                            'book': sentence_info['book'],
                            'chapter': sentence_info['chapter'],
                            'verse': sentence_info['verse'],
                            'text': sentence_text,
                            'reference': sentence_info['reference']
                        })
    
    elif method == 'spacy':
        if not HAS_SPACY:
            raise ImportError("spaCy is required for this method. Please install it with 'pip install spacy' and download a model with 'python -m spacy download en_core_web_sm'.")
        
        # Load spaCy model
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise ImportError("spaCy model not found. Please download it with 'python -m spacy download en_core_web_sm'.")
        
        results = []
        
        # Extract sentences with references
        sentences = _extract_sentences_with_refs(bible_dict, book, chapter, verse)
        
        for sentence_info in sentences:
            sentence_text = sentence_info['text']
            
            # Process with spaCy
            doc = nlp(sentence_text)
            
            # Extract entities
            for ent in doc.ents:
                # Filter by entity type if specified
                if entity_types and ent.label_ not in entity_types:
                    continue
                
                results.append({
                    'entity': ent.text,
                    'entity_type': ent.label_,
                    'book': sentence_info['book'],
                    'chapter': sentence_info['chapter'],
                    'verse': sentence_info['verse'],
                    'text': sentence_text,
                    'reference': sentence_info['reference']
                })
    
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'lexicon' or 'spacy'.")
    
    # Convert results to DataFrame
    if results:
        df = pd.DataFrame(results)
    else:
        # Create empty DataFrame with expected columns
        df = pd.DataFrame(columns=['entity', 'entity_type', 'book', 'chapter', 'verse', 'text', 'reference'])
    
    return df


def identify_people(bible_dict: Dict[str, Any], book: Optional[str] = None,
                  chapter: Optional[int] = None, verse: Optional[int] = None,
                  method: str = 'lexicon', custom_names: Optional[Set[str]] = None) -> pd.DataFrame:
    """
    Identify people mentioned in biblical texts.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        book: Optional book name to filter by
        chapter: Optional chapter number to filter by
        verse: Optional verse number to filter by
        method: Entity extraction method ('lexicon' or 'spacy')
        custom_names: Optional set of additional person names to identify
        
    Returns:
        DataFrame with identified people and their references
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> people = identify_people(bible, book="Matthew")
        >>> # Count references to each person
        >>> person_counts = people.groupby('entity')['entity'].count().sort_values(ascending=False)
    """
    # Combine default and custom name sets if provided
    person_names = BIBLICAL_PEOPLE
    if custom_names:
        person_names = BIBLICAL_PEOPLE.union(custom_names)
    
    if method == 'lexicon':
        # Use simple dictionary-based approach
        results = []
        
        # Extract sentences with references
        sentences = _extract_sentences_with_refs(bible_dict, book, chapter, verse)
        
        for sentence_info in sentences:
            sentence_text = sentence_info['text']
            
            # Tokenize text
            tokens = word_tokenize(sentence_text.lower())
            
            # Look for people
            for token in tokens:
                if token in person_names:
                    # Find original capitalization in text
                    original_token = None
                    for word in re.findall(r'\b\w+\b', sentence_text):
                        if word.lower() == token:
                            original_token = word
                            break
                    
                    if original_token:
                        results.append({
                            'entity': original_token,
                            'entity_type': 'PERSON',
                            'book': sentence_info['book'],
                            'chapter': sentence_info['chapter'],
                            'verse': sentence_info['verse'],
                            'text': sentence_text,
                            'reference': sentence_info['reference']
                        })
    
    elif method == 'spacy':
        if not HAS_SPACY:
            raise ImportError("spaCy is required for this method. Please install it with 'pip install spacy' and download a model with 'python -m spacy download en_core_web_sm'.")
        
        # Load spaCy model
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise ImportError("spaCy model not found. Please download it with 'python -m spacy download en_core_web_sm'.")
        
        results = []
        
        # Extract sentences with references
        sentences = _extract_sentences_with_refs(bible_dict, book, chapter, verse)
        
        for sentence_info in sentences:
            sentence_text = sentence_info['text']
            
            # Process with spaCy
            doc = nlp(sentence_text)
            
            # Extract person entities
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    results.append({
                        'entity': ent.text,
                        'entity_type': 'PERSON',
                        'book': sentence_info['book'],
                        'chapter': sentence_info['chapter'],
                        'verse': sentence_info['verse'],
                        'text': sentence_text,
                        'reference': sentence_info['reference']
                    })
    
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'lexicon' or 'spacy'.")
    
    # Convert results to DataFrame
    if results:
        df = pd.DataFrame(results)
    else:
        # Create empty DataFrame with expected columns
        df = pd.DataFrame(columns=['entity', 'entity_type', 'book', 'chapter', 'verse', 'text', 'reference'])
    
    return df


def identify_places(bible_dict: Dict[str, Any], book: Optional[str] = None,
                  chapter: Optional[int] = None, verse: Optional[int] = None,
                  method: str = 'lexicon', custom_places: Optional[Set[str]] = None) -> pd.DataFrame:
    """
    Identify places mentioned in biblical texts.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        book: Optional book name to filter by
        chapter: Optional chapter number to filter by
        verse: Optional verse number to filter by
        method: Entity extraction method ('lexicon' or 'spacy')
        custom_places: Optional set of additional place names to identify
        
    Returns:
        DataFrame with identified places and their references
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> places = identify_places(bible, book="Acts")
        >>> # Count references to each place
        >>> place_counts = places.groupby('entity')['entity'].count().sort_values(ascending=False)
    """
    # Combine default and custom place sets if provided
    place_names = BIBLICAL_PLACES
    if custom_places:
        place_names = BIBLICAL_PLACES.union(custom_places)
    
    if method == 'lexicon':
        # Use simple dictionary-based approach
        results = []
        
        # Extract sentences with references
        sentences = _extract_sentences_with_refs(bible_dict, book, chapter, verse)
        
        for sentence_info in sentences:
            sentence_text = sentence_info['text']
            
            # Tokenize text
            tokens = word_tokenize(sentence_text.lower())
            
            # Look for places
            for token in tokens:
                if token in place_names:
                    # Find original capitalization in text
                    original_token = None
                    for word in re.findall(r'\b\w+\b', sentence_text):
                        if word.lower() == token:
                            original_token = word
                            break
                    
                    if original_token:
                        results.append({
                            'entity': original_token,
                            'entity_type': 'PLACE',
                            'book': sentence_info['book'],
                            'chapter': sentence_info['chapter'],
                            'verse': sentence_info['verse'],
                            'text': sentence_text,
                            'reference': sentence_info['reference']
                        })
    
    elif method == 'spacy':
        if not HAS_SPACY:
            raise ImportError("spaCy is required for this method. Please install it with 'pip install spacy' and download a model with 'python -m spacy download en_core_web_sm'.")
        
        # Load spaCy model
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise ImportError("spaCy model not found. Please download it with 'python -m spacy download en_core_web_sm'.")
        
        results = []
        
        # Extract sentences with references
        sentences = _extract_sentences_with_refs(bible_dict, book, chapter, verse)
        
        for sentence_info in sentences:
            sentence_text = sentence_info['text']
            
            # Process with spaCy
            doc = nlp(sentence_text)
            
            # Extract place entities (GPE=GeoPoliticalEntity, LOC=Location)
            for ent in doc.ents:
                if ent.label_ in ('GPE', 'LOC'):
                    results.append({
                        'entity': ent.text,
                        'entity_type': 'PLACE',
                        'book': sentence_info['book'],
                        'chapter': sentence_info['chapter'],
                        'verse': sentence_info['verse'],
                        'text': sentence_text,
                        'reference': sentence_info['reference']
                    })
    
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'lexicon' or 'spacy'.")
    
    # Convert results to DataFrame
    if results:
        df = pd.DataFrame(results)
    else:
        # Create empty DataFrame with expected columns
        df = pd.DataFrame(columns=['entity', 'entity_type', 'book', 'chapter', 'verse', 'text', 'reference'])
    
    return df


def entity_frequency(bible_dict: Dict[str, Any], entity_type: str = 'PERSON',
                   book: Optional[str] = None, top_n: int = 20,
                   method: str = 'lexicon') -> pd.DataFrame:
    """
    Calculate the frequency of entities (people, places) in biblical texts.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        entity_type: Type of entity to analyze ('PERSON' or 'PLACE')
        book: Optional book name to filter by
        top_n: Number of top entities to include
        method: Entity extraction method ('lexicon' or 'spacy')
        
    Returns:
        DataFrame with entity frequencies
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Get top people mentioned in the Gospels
        >>> top_people = entity_frequency(bible, entity_type='PERSON', 
        ...                           book=['Matthew', 'Mark', 'Luke', 'John'])
    """
    # Extract entities based on type
    if entity_type == 'PERSON':
        entities_df = identify_people(bible_dict, book=book, method=method)
    elif entity_type == 'PLACE':
        entities_df = identify_places(bible_dict, book=book, method=method)
    else:
        entities_df = extract_entities(bible_dict, book=book, method=method)
        entities_df = entities_df[entities_df['entity_type'] == entity_type]
    
    if entities_df.empty:
        return pd.DataFrame(columns=['entity', 'count', 'entity_type'])
    
    # Count entity occurrences
    entity_counts = entities_df.groupby('entity')['entity'].count()
    
    # Create frequency DataFrame
    freq_df = pd.DataFrame({
        'entity': entity_counts.index,
        'count': entity_counts.values,
        'entity_type': entity_type
    })
    
    # Sort by frequency (descending)
    freq_df = freq_df.sort_values('count', ascending=False)
    
    # Limit to top_n
    if top_n > 0:
        freq_df = freq_df.head(top_n)
    
    return freq_df


def entity_co_occurrence(bible_dict: Dict[str, Any], entity_type: str = 'PERSON',
                       unit: str = 'verse', min_count: int = 2,
                       book: Optional[str] = None, method: str = 'lexicon') -> pd.DataFrame:
    """
    Calculate co-occurrence between entities (people, places) in biblical texts.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        entity_type: Type of entity to analyze ('PERSON' or 'PLACE')
        unit: Unit for co-occurrence calculation ('verse', 'sentence', or 'chapter')
        min_count: Minimum co-occurrence count to include
        book: Optional book name to filter by
        method: Entity extraction method ('lexicon' or 'spacy')
        
    Returns:
        DataFrame with entity co-occurrence counts
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Find which people appear together in the Gospels
        >>> person_cooccur = entity_co_occurrence(bible, entity_type='PERSON', 
        ...                                     book=['Matthew', 'Mark', 'Luke', 'John'])
    """
    # Extract entities based on type
    if entity_type == 'PERSON':
        entities_df = identify_people(bible_dict, book=book, method=method)
    elif entity_type == 'PLACE':
        entities_df = identify_places(bible_dict, book=book, method=method)
    else:
        entities_df = extract_entities(bible_dict, book=book, method=method)
        entities_df = entities_df[entities_df['entity_type'] == entity_type]
    
    if entities_df.empty:
        return pd.DataFrame(columns=['entity1', 'entity2', 'count'])
    
    # Group entities by unit
    if unit == 'verse':
        grouped = entities_df.groupby(['book', 'chapter', 'verse'])
    elif unit == 'sentence':
        # For sentence level, use the full text as grouper (each sentence is unique)
        grouped = entities_df.groupby(['book', 'chapter', 'verse', 'text'])
    elif unit == 'chapter':
        grouped = entities_df.groupby(['book', 'chapter'])
    else:
        raise ValueError(f"Unknown unit '{unit}'. Choose from 'verse', 'sentence', or 'chapter'.")
    
    # Calculate co-occurrences
    cooccurrences = defaultdict(int)
    
    for _, group in grouped:
        # Get unique entities in this unit
        entities = group['entity'].unique()
        
        # Skip if there's only one entity
        if len(entities) < 2:
            continue
        
        # Count all pairs
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                if entity1 < entity2:  # Consistent ordering
                    cooccurrences[(entity1, entity2)] += 1
                else:
                    cooccurrences[(entity2, entity1)] += 1
    
    # Convert to DataFrame
    results = []
    for (entity1, entity2), count in cooccurrences.items():
        if count >= min_count:
            results.append({
                'entity1': entity1,
                'entity2': entity2,
                'count': count
            })
    
    # Create DataFrame
    if results:
        cooccur_df = pd.DataFrame(results)
        cooccur_df = cooccur_df.sort_values('count', ascending=False)
    else:
        cooccur_df = pd.DataFrame(columns=['entity1', 'entity2', 'count'])
    
    return cooccur_df
