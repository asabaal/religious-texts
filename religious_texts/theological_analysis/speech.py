"""
Speech Attribution Analysis Module

This module provides functions for identifying and analyzing speech attribution
in biblical texts, focusing on identifying who is speaking and analyzing patterns
in different speakers' speech.
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union, Any, Tuple, Set

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize

# Speech attribution markers
SPEECH_MARKERS = {
    'said': ['said', 'saith', 'spoke', 'spake', 'answered', 'replied', 'commanded', 'asked'],
    'quotation': ['"', "'", """, """, "'", "'"]
}

# Common speakers
SPEAKERS = {
    # Divine speakers
    'god': {'god', 'the lord', 'lord', 'lord god', 'the lord god', 'the almighty'},
    'jesus': {'jesus', 'christ', 'jesus christ', 'the son', 'son of man', 'son of god'},
    'holy_spirit': {'holy spirit', 'holy ghost', 'the spirit', 'spirit of god', 'spirit of the lord'},
    
    # Human speaker categories
    'prophets': {
        'moses', 'isaiah', 'jeremiah', 'ezekiel', 'daniel', 'hosea', 'joel', 'amos',
        'obadiah', 'jonah', 'micah', 'nahum', 'habakkuk', 'zephaniah', 'haggai',
        'zechariah', 'malachi', 'elijah', 'elisha', 'samuel', 'nathan'
    },
    'apostles': {
        'peter', 'john', 'james', 'andrew', 'philip', 'bartholomew', 'matthew',
        'thomas', 'james the less', 'simon', 'judas', 'paul', 'matthias'
    },
    'kings': {
        'david', 'solomon', 'saul', 'rehoboam', 'jeroboam', 'abijam', 'nadab',
        'asa', 'baasha', 'elah', 'zimri', 'omri', 'ahab', 'ahaziah', 'jehoram',
        'jehu', 'jehoahaz', 'jehoash', 'jeroboam ii', 'zechariah', 'shallum',
        'menahem', 'pekahiah', 'pekah', 'hoshea', 'jehoshaphat', 'joram',
        'ahaziah', 'athaliah', 'joash', 'amaziah', 'uzziah', 'jotham',
        'ahaz', 'hezekiah', 'manasseh', 'amon', 'josiah', 'jehoahaz',
        'jehoiakim', 'jehoiachin', 'zedekiah'
    }
}


def identify_speech_segments(bible_dict: Dict[str, Any], book: Optional[str] = None,
                           chapter: Optional[int] = None, verse: Optional[int] = None) -> pd.DataFrame:
    """
    Identify speech segments in biblical texts with potential speakers.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        book: Optional book name to filter by
        chapter: Optional chapter number to filter by
        verse: Optional verse number to filter by
        
    Returns:
        DataFrame with identified speech segments and potential speakers
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Identify speech in Genesis
        >>> speech_df = identify_speech_segments(bible, book="Genesis")
    """
    results = []
    
    # Compile regex patterns
    # Pattern for speech attribution with quotation marks
    quote_pattern = re.compile(r'([\w\s]+)\s+(said|saith|spake|spoke|answered|replied|asked|commanded)[,:]?\s*[""]([^""]+)[""]')
    
    # Pattern for speech attribution without quotation marks (more challenging)
    said_pattern = re.compile(r'([\w\s]+)\s+(said|saith|spake|spoke|answered|replied|asked|commanded)[,:]?\s+([^\.!?;]+[\.!?;])')
    
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
                # Look for speech with quotation marks
                for match in quote_pattern.finditer(verse_text):
                    potential_speaker = match.group(1).strip()
                    speech_marker = match.group(2)
                    speech_content = match.group(3).strip()
                    
                    results.append({
                        'book': book_name,
                        'chapter': chapter_num,
                        'verse': verse_num,
                        'reference': f"{book_name} {chapter_num}:{verse_num}",
                        'potential_speaker': potential_speaker,
                        'speech_marker': speech_marker,
                        'speech_content': speech_content,
                        'confidence': 'high',
                        'full_text': verse_text
                    })
                
                # Look for speech without quotation marks
                for match in said_pattern.finditer(verse_text):
                    # Skip if already matched with quotes
                    if any(result['speech_content'] in match.group(3) for result in results
                          if result['book'] == book_name and 
                          result['chapter'] == chapter_num and
                          result['verse'] == verse_num):
                        continue
                    
                    potential_speaker = match.group(1).strip()
                    speech_marker = match.group(2)
                    speech_content = match.group(3).strip()
                    
                    results.append({
                        'book': book_name,
                        'chapter': chapter_num,
                        'verse': verse_num,
                        'reference': f"{book_name} {chapter_num}:{verse_num}",
                        'potential_speaker': potential_speaker,
                        'speech_marker': speech_marker,
                        'speech_content': speech_content,
                        'confidence': 'medium',
                        'full_text': verse_text
                    })
    
    # Convert results to DataFrame
    if results:
        df = pd.DataFrame(results)
    else:
        # Create empty DataFrame with expected columns
        df = pd.DataFrame(columns=['book', 'chapter', 'verse', 'reference',
                                 'potential_speaker', 'speech_marker', 
                                 'speech_content', 'confidence', 'full_text'])
    
    return df


def _normalize_speaker(speaker: str) -> str:
    """
    Helper function to normalize speaker names to canonical categories.
    
    Args:
        speaker: Raw speaker name from text
        
    Returns:
        Normalized speaker category
    """
    speaker_lower = speaker.lower()
    
    # Check for divine speakers
    for category, names in SPEAKERS.items():
        if speaker_lower in names or any(name in speaker_lower for name in names):
            return category
    
    # Check for speaker groups
    for group, members in SPEAKERS.items():
        if any(member in speaker_lower for member in members):
            return group
    
    # If not matched, return original
    return speaker


def speech_attribution_analysis(bible_dict: Dict[str, Any], book: Optional[str] = None,
                              normalized: bool = True) -> pd.DataFrame:
    """
    Analyze speech attribution patterns in biblical texts.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        book: Optional book name to filter by
        normalized: Whether to normalize speaker names to canonical categories
        
    Returns:
        DataFrame with speech attribution analysis results
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Analyze speech in the Gospels
        >>> speech_analysis = speech_attribution_analysis(bible, book=["Matthew", "Mark", "Luke", "John"])
        >>> # Count speeches by speaker
        >>> speaker_counts = speech_analysis.groupby('speaker')['speaker'].count().sort_values(ascending=False)
    """
    # Get speech segments
    speech_df = identify_speech_segments(bible_dict, book=book)
    
    if speech_df.empty:
        return pd.DataFrame()
    
    # Normalize speakers if requested
    if normalized:
        speech_df['speaker'] = speech_df['potential_speaker'].apply(_normalize_speaker)
    else:
        speech_df['speaker'] = speech_df['potential_speaker']
    
    # Calculate speech length
    speech_df['speech_word_count'] = speech_df['speech_content'].apply(lambda x: len(word_tokenize(x)))
    
    # Summarize by book and speaker
    book_speaker_counts = speech_df.groupby(['book', 'speaker']).agg({
        'speech_content': 'count',
        'speech_word_count': 'sum'
    }).reset_index()
    
    book_speaker_counts.columns = ['book', 'speaker', 'speech_count', 'total_words']
    book_speaker_counts['avg_speech_length'] = book_speaker_counts['total_words'] / book_speaker_counts['speech_count']
    
    return book_speaker_counts


def speech_act_distribution(speech_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the distribution of different speech acts in attributed speech.
    
    Args:
        speech_df: DataFrame with speech segments (from identify_speech_segments)
        
    Returns:
        DataFrame with speech act analysis results
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> speech_segments = identify_speech_segments(bible, book="Genesis")
        >>> speech_acts = speech_act_distribution(speech_segments)
    """
    if speech_df.empty:
        return pd.DataFrame()
    
    # Initialize speech act categories
    speech_acts = {
        'command': ['command', 'order', 'tell', 'must', 'shall', 'will', 'go', 'do', 'obey'],
        'question': ['?', 'who', 'what', 'where', 'when', 'why', 'how', 'which'],
        'promise': ['promise', 'covenant', 'vow', 'swear', 'surely', 'verily', 'indeed', 'truly'],
        'blessing': ['bless', 'blessed', 'blessing'],
        'curse': ['curse', 'cursed', 'woe'],
        'prophecy': ['shall come to pass', 'will come', 'in that day', 'days are coming'],
        'praise': ['praise', 'glory', 'glorify', 'thank', 'worship', 'sing'],
        'lament': ['woe', 'alas', 'mourn', 'weep', 'grieve']
    }
    
    # Compile regex patterns for each speech act
    act_patterns = {}
    for act, keywords in speech_acts.items():
        patterns = []
        for keyword in keywords:
            # Simple word boundary pattern
            patterns.append(r'\b' + re.escape(keyword) + r'\b')
        
        # Combine patterns for this act
        act_patterns[act] = re.compile('|'.join(patterns), re.IGNORECASE)
    
    # Function to identify speech acts in a speech segment
    def identify_acts(speech_content):
        acts = []
        for act, pattern in act_patterns.items():
            if pattern.search(speech_content):
                acts.append(act)
        
        return acts if acts else ['other']
    
    # Add speech act analysis to each segment
    results = []
    
    for _, row in speech_df.iterrows():
        speech_content = row['speech_content']
        acts = identify_acts(speech_content)
        
        for act in acts:
            result = {
                'book': row['book'],
                'chapter': row['chapter'],
                'verse': row['verse'],
                'reference': row['reference'],
                'speaker': row['potential_speaker'],
                'speech_act': act,
                'speech_content': speech_content
            }
            
            results.append(result)
    
    # Convert results to DataFrame
    if results:
        df = pd.DataFrame(results)
        
        # Summarize by book, speaker, and speech act
        summary = df.groupby(['book', 'speaker', 'speech_act']).size().reset_index(name='count')
        
        # Pivot to get speaker rows and speech act columns
        pivot_df = summary.pivot_table(
            index=['book', 'speaker'],
            columns='speech_act',
            values='count',
            fill_value=0
        ).reset_index()
        
        return pivot_df
    else:
        # Create empty DataFrame with expected columns
        return pd.DataFrame(columns=['book', 'speaker'] + list(speech_acts.keys()) + ['other'])


def speech_comparison_by_speaker(speech_df: pd.DataFrame, 
                               speakers: List[str],
                               use_normalized: bool = True) -> Dict[str, Any]:
    """
    Compare speech patterns between different speakers.
    
    Args:
        speech_df: DataFrame with speech segments (from identify_speech_segments)
        speakers: List of speakers to compare
        use_normalized: Whether to use normalized speaker names
        
    Returns:
        Dictionary with comparative analysis results
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> speech_segments = identify_speech_segments(bible)
        >>> comparison = speech_comparison_by_speaker(speech_segments, ['god', 'jesus'])
    """
    if speech_df.empty:
        return {}
    
    # Ensure speaker column is available
    if use_normalized and 'speaker' not in speech_df.columns:
        speech_df['speaker'] = speech_df['potential_speaker'].apply(_normalize_speaker)
    elif not use_normalized and 'speaker' not in speech_df.columns:
        speech_df['speaker'] = speech_df['potential_speaker']
    
    # Filter to selected speakers
    speaker_col = 'speaker' if 'speaker' in speech_df.columns else 'potential_speaker'
    filtered_df = speech_df[speech_df[speaker_col].isin(speakers)]
    
    if filtered_df.empty:
        return {}
    
    # Calculate speech length
    if 'speech_word_count' not in filtered_df.columns:
        filtered_df['speech_word_count'] = filtered_df['speech_content'].apply(lambda x: len(word_tokenize(x)))
    
    # Initialize comparison results
    comparison = {}
    
    # 1. Basic speech statistics
    speaker_stats = filtered_df.groupby(speaker_col).agg({
        'speech_content': 'count',
        'speech_word_count': ['sum', 'mean', 'median', 'max']
    })
    
    speaker_stats.columns = ['speech_count', 'total_words', 'avg_length', 'median_length', 'max_length']
    comparison['basic_stats'] = speaker_stats.reset_index()
    
    # 2. Most frequent words by speaker
    speaker_words = {}
    
    for speaker in speakers:
        speaker_speech = filtered_df[filtered_df[speaker_col] == speaker]['speech_content']
        
        if speaker_speech.empty:
            speaker_words[speaker] = {}
            continue
        
        # Tokenize and count words
        all_words = []
        for speech in speaker_speech:
            all_words.extend([word.lower() for word in word_tokenize(speech)])
        
        # Count words
        word_counts = Counter(all_words)
        
        # Filter common stopwords
        stopwords = {'the', 'and', 'of', 'to', 'a', 'in', 'that', 'is', 'i', 'it', 'for', 
                    'you', 'be', 'with', 'as', 'this', 'by', 'on', 'not', 'are', 'from', 
                    'have', 'will', 'he', 'they', 'at', 'or', 'his', 'my', 'an', 'but', 
                    'if', 'we', 'me', 'your', 'their', 'so', 'which', 'him', 'our', 'was'}
        
        filtered_counts = {word: count for word, count in word_counts.items() 
                         if word not in stopwords and len(word) > 1}
        
        # Get top words
        top_words = dict(sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)[:20])
        speaker_words[speaker] = top_words
    
    comparison['top_words'] = speaker_words
    
    # 3. Speech acts by speaker
    speech_acts = speech_act_distribution(filtered_df)
    if not speech_acts.empty:
        speech_acts = speech_acts.set_index('speaker')
        comparison['speech_acts'] = speech_acts.to_dict(orient='index')
    
    # 4. Books where each speaker appears
    speaker_books = {}
    for speaker in speakers:
        speaker_df = filtered_df[filtered_df[speaker_col] == speaker]
        if not speaker_df.empty:
            book_counts = speaker_df['book'].value_counts().to_dict()
            speaker_books[speaker] = book_counts
    
    comparison['books'] = speaker_books
    
    return comparison
