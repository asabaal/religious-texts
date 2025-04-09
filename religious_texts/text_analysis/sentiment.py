"""
Sentiment Analysis Module

This module provides functions for analyzing sentiment and emotions in biblical texts.
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union, Any, Tuple, Set

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize

# Try to import optional dependencies
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False


# Basic positive and negative word lexicons (can be extended)
POSITIVE_WORDS = {
    'love', 'joy', 'peace', 'hope', 'faith', 'good', 'righteous', 'blessed',
    'holy', 'happy', 'glad', 'praise', 'mercy', 'grace', 'salvation', 'heaven',
    'gentle', 'kind', 'comfort', 'strength', 'courage', 'wisdom', 'light',
    'true', 'truth', 'pure', 'honor', 'glory', 'abundance', 'rejoice',
    'forgive', 'forgiveness', 'life', 'prosperity', 'blessing', 'delight'
}

NEGATIVE_WORDS = {
    'sin', 'evil', 'wicked', 'death', 'darkness', 'hell', 'fear', 'anger',
    'hate', 'wrath', 'judgment', 'curse', 'punishment', 'sorrow', 'pain',
    'suffering', 'affliction', 'trouble', 'enemy', 'war', 'blood', 'destroy',
    'destruction', 'corrupt', 'false', 'deceit', 'grief', 'torment', 'weep',
    'weeping', 'mourn', 'mourning', 'crime', 'transgression', 'guilt',
    'condemnation', 'perish', 'disaster', 'calamity', 'famine', 'plague',
    'disease', 'misery', 'shame', 'terror', 'horror', 'cruel', 'jealous',
    'greed', 'bitter', 'betray', 'doom', 'dread', 'envy', 'strife', 'lust'
}

# Basic emotion categories with associated words
EMOTION_LEXICON = {
    'joy': {'joy', 'happy', 'glad', 'rejoice', 'delight', 'pleasure', 'merry', 'cheerful'},
    'sadness': {'sad', 'sorrow', 'grief', 'mourn', 'weep', 'lament', 'distress', 'despair'},
    'anger': {'anger', 'wrath', 'fury', 'rage', 'indignation', 'vengeance', 'fierce'},
    'fear': {'fear', 'afraid', 'terror', 'dread', 'horror', 'alarm', 'panic', 'trembling'},
    'love': {'love', 'beloved', 'loving', 'affection', 'fond', 'care', 'cherish'},
    'hate': {'hate', 'hatred', 'despise', 'abhor', 'loathe', 'detest', 'scorn'},
    'surprise': {'amaze', 'astonish', 'wonder', 'marvel', 'awe', 'astound', 'startled'},
    'disgust': {'abomination', 'disgust', 'loathsome', 'revulsion', 'abhor', 'repulsive', 'vile'},
    'shame': {'shame', 'ashamed', 'embarrass', 'humiliate', 'disgrace', 'dishonor'},
    'peace': {'peace', 'calm', 'tranquil', 'quiet', 'rest', 'serene', 'still'},
    'hope': {'hope', 'expect', 'anticipate', 'trust', 'confidence', 'faith', 'assurance'}
}


def _extract_sentences(bible_dict: Dict[str, Any], book: Optional[str] = None, 
                      chapter: Optional[int] = None, verse: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Helper function to extract sentences with their references from a Bible dictionary.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        book: Optional book name to filter by
        chapter: Optional chapter number to filter by
        verse: Optional verse number to filter by
        
    Returns:
        List of dictionaries, each containing a sentence and its reference
    """
    sentences = []
    
    # Determine which books to include
    if book:
        if book not in bible_dict:
            return []
        books_to_check = {book: bible_dict[book]}
    else:
        books_to_check = bible_dict
    
    # Extract sentences based on filters
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
                # Split verse into sentences
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


def sentiment_analysis(bible_dict: Dict[str, Any], book: Optional[str] = None,
                      chapter: Optional[int] = None, verse: Optional[int] = None,
                      method: str = 'lexicon', unit: str = 'verse') -> pd.DataFrame:
    """
    Perform sentiment analysis on biblical texts.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        book: Optional book name to filter by
        chapter: Optional chapter number to filter by
        verse: Optional verse number to filter by
        method: Sentiment analysis method ('lexicon', 'textblob', or 'vader')
        unit: Unit of analysis ('verse', 'sentence', 'chapter', or 'book')
        
    Returns:
        DataFrame with sentiment scores for each unit
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Analyze sentiment in Psalms
        >>> sentiment = sentiment_analysis(bible, book="Psalms")
        >>> # Get average sentiment by chapter
        >>> chapter_avg = sentiment.groupby('chapter')['sentiment_score'].mean()
    """
    # Define sentiment analysis function based on method
    if method == 'lexicon':
        def get_sentiment(text):
            tokens = [token.lower() for token in word_tokenize(text)]
            pos_count = sum(1 for token in tokens if token in POSITIVE_WORDS)
            neg_count = sum(1 for token in tokens if token in NEGATIVE_WORDS)
            total_count = pos_count + neg_count
            
            if total_count == 0:
                return 0.0  # Neutral
            
            # Return sentiment score between -1 and 1
            return (pos_count - neg_count) / total_count
    
    elif method == 'textblob':
        if not HAS_TEXTBLOB:
            raise ImportError("TextBlob is required for this method. Please install it with 'pip install textblob'.")
        
        def get_sentiment(text):
            return TextBlob(text).sentiment.polarity
    
    elif method == 'vader':
        if not HAS_VADER:
            raise ImportError("NLTK's VADER is required for this method. Please install it with 'pip install nltk' and run 'nltk.download(\"vader_lexicon\")'.")
        
        # Initialize VADER analyzer
        sia = SentimentIntensityAnalyzer()
        
        def get_sentiment(text):
            scores = sia.polarity_scores(text)
            return scores['compound']  # Compound score between -1 and 1
    
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'lexicon', 'textblob', or 'vader'.")
    
    # Process based on unit type
    if unit == 'verse':
        results = []
        
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
                    # Calculate sentiment
                    sentiment_score = get_sentiment(verse_text)
                    
                    results.append({
                        'book': book_name,
                        'chapter': chapter_num,
                        'verse': verse_num,
                        'text': verse_text,
                        'sentiment_score': sentiment_score,
                        'sentiment_category': 'positive' if sentiment_score > 0 else 
                                             'negative' if sentiment_score < 0 else 'neutral',
                        'reference': f"{book_name} {chapter_num}:{verse_num}"
                    })
    
    elif unit == 'sentence':
        # Extract sentences
        sentences = _extract_sentences(bible_dict, book, chapter, verse)
        
        results = []
        for sentence_info in sentences:
            # Calculate sentiment
            sentiment_score = get_sentiment(sentence_info['text'])
            
            sentence_info['sentiment_score'] = sentiment_score
            sentence_info['sentiment_category'] = 'positive' if sentiment_score > 0 else 
                                                'negative' if sentiment_score < 0 else 'neutral'
            results.append(sentence_info)
    
    elif unit == 'chapter':
        results = []
        
        # Determine which books to include
        if book:
            if book not in bible_dict:
                return pd.DataFrame()
            books_to_check = {book: bible_dict[book]}
        else:
            books_to_check = bible_dict
        
        # Process each chapter
        for book_name, chapters in books_to_check.items():
            # Filter by chapter
            if chapter:
                if chapter not in chapters:
                    continue
                chapters_to_check = {chapter: chapters[chapter]}
            else:
                chapters_to_check = chapters
            
            for chapter_num, verses in chapters_to_check.items():
                # Combine all verses in the chapter
                chapter_text = ' '.join(verse_text for verse_text in verses.values())
                
                # Calculate sentiment
                sentiment_score = get_sentiment(chapter_text)
                
                results.append({
                    'book': book_name,
                    'chapter': chapter_num,
                    'text': chapter_text[:100] + '...' if len(chapter_text) > 100 else chapter_text,
                    'sentiment_score': sentiment_score,
                    'sentiment_category': 'positive' if sentiment_score > 0 else 
                                         'negative' if sentiment_score < 0 else 'neutral',
                    'reference': f"{book_name} {chapter_num}"
                })
    
    elif unit == 'book':
        results = []
        
        # Determine which books to include
        if book:
            if book not in bible_dict:
                return pd.DataFrame()
            books_to_check = {book: bible_dict[book]}
        else:
            books_to_check = bible_dict
        
        # Process each book
        for book_name, chapters in books_to_check.items():
            # Combine all chapters and verses
            book_text = ' '.join(
                verse_text 
                for chapter_verses in chapters.values() 
                for verse_text in chapter_verses.values()
            )
            
            # Calculate sentiment
            sentiment_score = get_sentiment(book_text)
            
            results.append({
                'book': book_name,
                'text': book_text[:100] + '...' if len(book_text) > 100 else book_text,
                'sentiment_score': sentiment_score,
                'sentiment_category': 'positive' if sentiment_score > 0 else 
                                     'negative' if sentiment_score < 0 else 'neutral',
                'reference': book_name
            })
    
    else:
        raise ValueError(f"Unknown unit '{unit}'. Choose from 'verse', 'sentence', 'chapter', or 'book'.")
    
    # Convert results to DataFrame
    if results:
        df = pd.DataFrame(results)
    else:
        # Create empty DataFrame with expected columns
        df = pd.DataFrame(columns=['book', 'chapter', 'verse', 'text', 
                                 'sentiment_score', 'sentiment_category', 'reference'])
    
    return df


def emotion_detection(bible_dict: Dict[str, Any], book: Optional[str] = None,
                     chapter: Optional[int] = None, verse: Optional[int] = None,
                     custom_lexicon: Optional[Dict[str, Set[str]]] = None,
                     unit: str = 'verse') -> pd.DataFrame:
    """
    Detect emotions in biblical texts based on lexical analysis.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        book: Optional book name to filter by
        chapter: Optional chapter number to filter by
        verse: Optional verse number to filter by
        custom_lexicon: Optional custom emotion lexicon to use instead of default
        unit: Unit of analysis ('verse', 'sentence', 'chapter', or 'book')
        
    Returns:
        DataFrame with emotion scores for each unit
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Analyze emotions in Psalms
        >>> emotions = emotion_detection(bible, book="Psalms")
        >>> # Get most common emotions
        >>> emotion_counts = emotions[EMOTION_LEXICON.keys()].sum()
        >>> emotion_counts.sort_values(ascending=False)
    """
    # Use custom lexicon if provided, otherwise use default
    emotion_lex = custom_lexicon if custom_lexicon else EMOTION_LEXICON
    emotion_categories = list(emotion_lex.keys())
    
    # Process based on unit type
    if unit == 'verse':
        results = []
        
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
                    # Tokenize text
                    tokens = [token.lower() for token in word_tokenize(verse_text)]
                    token_set = set(tokens)
                    
                    # Count emotions
                    emotion_counts = {}
                    dominant_emotion = None
                    max_count = 0
                    
                    for emotion, word_set in emotion_lex.items():
                        # Count matching words
                        matches = token_set.intersection(word_set)
                        count = len(matches)
                        emotion_counts[emotion] = count
                        
                        # Track dominant emotion
                        if count > max_count:
                            max_count = count
                            dominant_emotion = emotion
                    
                    # If no emotions detected, set dominant to 'neutral'
                    if max_count == 0:
                        dominant_emotion = 'neutral'
                    
                    # Create result
                    result = {
                        'book': book_name,
                        'chapter': chapter_num,
                        'verse': verse_num,
                        'text': verse_text,
                        'dominant_emotion': dominant_emotion,
                        'emotion_count': max_count,
                        'reference': f"{book_name} {chapter_num}:{verse_num}"
                    }
                    
                    # Add individual emotion counts
                    for emotion, count in emotion_counts.items():
                        result[emotion] = count
                    
                    results.append(result)
    
    elif unit == 'sentence':
        # Extract sentences
        sentences = _extract_sentences(bible_dict, book, chapter, verse)
        
        results = []
        for sentence_info in sentences:
            # Tokenize text
            tokens = [token.lower() for token in word_tokenize(sentence_info['text'])]
            token_set = set(tokens)
            
            # Count emotions
            emotion_counts = {}
            dominant_emotion = None
            max_count = 0
            
            for emotion, word_set in emotion_lex.items():
                # Count matching words
                matches = token_set.intersection(word_set)
                count = len(matches)
                emotion_counts[emotion] = count
                
                # Track dominant emotion
                if count > max_count:
                    max_count = count
                    dominant_emotion = emotion
            
            # If no emotions detected, set dominant to 'neutral'
            if max_count == 0:
                dominant_emotion = 'neutral'
            
            # Create result
            result = sentence_info.copy()
            result['dominant_emotion'] = dominant_emotion
            result['emotion_count'] = max_count
            
            # Add individual emotion counts
            for emotion, count in emotion_counts.items():
                result[emotion] = count
            
            results.append(result)
    
    elif unit in ('chapter', 'book'):
        results = []
        
        # Determine which books to include
        if book:
            if book not in bible_dict:
                return pd.DataFrame()
            books_to_check = {book: bible_dict[book]}
        else:
            books_to_check = bible_dict
        
        for book_name, chapters in books_to_check.items():
            if unit == 'chapter':
                # Filter by chapter
                if chapter:
                    if chapter not in chapters:
                        continue
                    chapters_to_check = {chapter: chapters[chapter]}
                else:
                    chapters_to_check = chapters
                
                # Process each chapter
                for chapter_num, verses in chapters_to_check.items():
                    # Combine all verses in the chapter
                    chapter_text = ' '.join(verse_text for verse_text in verses.values())
                    
                    # Tokenize text
                    tokens = [token.lower() for token in word_tokenize(chapter_text)]
                    token_set = set(tokens)
                    
                    # Count emotions
                    emotion_counts = {}
                    dominant_emotion = None
                    max_count = 0
                    
                    for emotion, word_set in emotion_lex.items():
                        # Count matching words
                        matches = token_set.intersection(word_set)
                        count = len(matches)
                        emotion_counts[emotion] = count
                        
                        # Track dominant emotion
                        if count > max_count:
                            max_count = count
                            dominant_emotion = emotion
                    
                    # If no emotions detected, set dominant to 'neutral'
                    if max_count == 0:
                        dominant_emotion = 'neutral'
                    
                    # Create result
                    result = {
                        'book': book_name,
                        'chapter': chapter_num,
                        'text': chapter_text[:100] + '...' if len(chapter_text) > 100 else chapter_text,
                        'dominant_emotion': dominant_emotion,
                        'emotion_count': max_count,
                        'reference': f"{book_name} {chapter_num}"
                    }
                    
                    # Add individual emotion counts
                    for emotion, count in emotion_counts.items():
                        result[emotion] = count
                    
                    results.append(result)
            
            else:  # unit == 'book'
                # Combine all chapters and verses
                book_text = ' '.join(
                    verse_text 
                    for chapter_verses in chapters.values() 
                    for verse_text in chapter_verses.values()
                )
                
                # Tokenize text
                tokens = [token.lower() for token in word_tokenize(book_text)]
                token_set = set(tokens)
                
                # Count emotions
                emotion_counts = {}
                dominant_emotion = None
                max_count = 0
                
                for emotion, word_set in emotion_lex.items():
                    # Count matching words
                    matches = token_set.intersection(word_set)
                    count = len(matches)
                    emotion_counts[emotion] = count
                    
                    # Normalize by text length for fair comparison between books
                    normalized_count = count / len(tokens) if tokens else 0
                    emotion_counts[f"{emotion}_normalized"] = normalized_count
                    
                    # Track dominant emotion
                    if count > max_count:
                        max_count = count
                        dominant_emotion = emotion
                
                # If no emotions detected, set dominant to 'neutral'
                if max_count == 0:
                    dominant_emotion = 'neutral'
                
                # Create result
                result = {
                    'book': book_name,
                    'text': book_text[:100] + '...' if len(book_text) > 100 else book_text,
                    'dominant_emotion': dominant_emotion,
                    'emotion_count': max_count,
                    'reference': book_name
                }
                
                # Add individual emotion counts
                for emotion, count in emotion_counts.items():
                    result[emotion] = count
                
                results.append(result)
    
    else:
        raise ValueError(f"Unknown unit '{unit}'. Choose from 'verse', 'sentence', 'chapter', or 'book'.")
    
    # Convert results to DataFrame
    if results:
        df = pd.DataFrame(results)
    else:
        # Create empty DataFrame with expected columns
        cols = ['book', 'reference', 'dominant_emotion', 'emotion_count'] + emotion_categories
        df = pd.DataFrame(columns=cols)
    
    return df


def subjectivity_analysis(bible_dict: Dict[str, Any], book: Optional[str] = None,
                         chapter: Optional[int] = None, verse: Optional[int] = None,
                         unit: str = 'verse') -> pd.DataFrame:
    """
    Analyze subjectivity (objective vs. subjective) in biblical texts.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        book: Optional book name to filter by
        chapter: Optional chapter number to filter by
        verse: Optional verse number to filter by
        unit: Unit of analysis ('verse', 'sentence', 'chapter', or 'book')
        
    Returns:
        DataFrame with subjectivity scores for each unit
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> subj = subjectivity_analysis(bible, book="John")
    """
    if not HAS_TEXTBLOB:
        raise ImportError("TextBlob is required for subjectivity analysis. "
                         "Please install it with 'pip install textblob'.")
    
    # Process based on unit type
    if unit == 'verse':
        results = []
        
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
                    # Calculate subjectivity
                    subjectivity = TextBlob(verse_text).sentiment.subjectivity
                    
                    # Categorize subjectivity
                    if subjectivity < 0.33:
                        category = 'objective'
                    elif subjectivity > 0.66:
                        category = 'subjective'
                    else:
                        category = 'neutral'
                    
                    results.append({
                        'book': book_name,
                        'chapter': chapter_num,
                        'verse': verse_num,
                        'text': verse_text,
                        'subjectivity_score': subjectivity,
                        'subjectivity_category': category,
                        'reference': f"{book_name} {chapter_num}:{verse_num}"
                    })
    
    elif unit == 'sentence':
        # Extract sentences
        sentences = _extract_sentences(bible_dict, book, chapter, verse)
        
        results = []
        for sentence_info in sentences:
            # Calculate subjectivity
            subjectivity = TextBlob(sentence_info['text']).sentiment.subjectivity
            
            # Categorize subjectivity
            if subjectivity < 0.33:
                category = 'objective'
            elif subjectivity > 0.66:
                category = 'subjective'
            else:
                category = 'neutral'
            
            result = sentence_info.copy()
            result['subjectivity_score'] = subjectivity
            result['subjectivity_category'] = category
            
            results.append(result)
    
    elif unit in ('chapter', 'book'):
        results = []
        
        # Determine which books to include
        if book:
            if book not in bible_dict:
                return pd.DataFrame()
            books_to_check = {book: bible_dict[book]}
        else:
            books_to_check = bible_dict
        
        for book_name, chapters in books_to_check.items():
            if unit == 'chapter':
                # Filter by chapter
                if chapter:
                    if chapter not in chapters:
                        continue
                    chapters_to_check = {chapter: chapters[chapter]}
                else:
                    chapters_to_check = chapters
                
                # Process each chapter
                for chapter_num, verses in chapters_to_check.items():
                    # Combine all verses in the chapter
                    chapter_text = ' '.join(verse_text for verse_text in verses.values())
                    
                    # Calculate subjectivity
                    subjectivity = TextBlob(chapter_text).sentiment.subjectivity
                    
                    # Categorize subjectivity
                    if subjectivity < 0.33:
                        category = 'objective'
                    elif subjectivity > 0.66:
                        category = 'subjective'
                    else:
                        category = 'neutral'
                    
                    results.append({
                        'book': book_name,
                        'chapter': chapter_num,
                        'text': chapter_text[:100] + '...' if len(chapter_text) > 100 else chapter_text,
                        'subjectivity_score': subjectivity,
                        'subjectivity_category': category,
                        'reference': f"{book_name} {chapter_num}"
                    })
            
            else:  # unit == 'book'
                # Combine all chapters and verses
                book_text = ' '.join(
                    verse_text 
                    for chapter_verses in chapters.values() 
                    for verse_text in chapter_verses.values()
                )
                
                # Calculate subjectivity
                subjectivity = TextBlob(book_text).sentiment.subjectivity
                
                # Categorize subjectivity
                if subjectivity < 0.33:
                    category = 'objective'
                elif subjectivity > 0.66:
                    category = 'subjective'
                else:
                    category = 'neutral'
                
                results.append({
                    'book': book_name,
                    'text': book_text[:100] + '...' if len(book_text) > 100 else book_text,
                    'subjectivity_score': subjectivity,
                    'subjectivity_category': category,
                    'reference': book_name
                })
    
    else:
        raise ValueError(f"Unknown unit '{unit}'. Choose from 'verse', 'sentence', 'chapter', or 'book'.")
    
    # Convert results to DataFrame
    if results:
        df = pd.DataFrame(results)
    else:
        # Create empty DataFrame with expected columns
        df = pd.DataFrame(columns=['book', 'chapter', 'verse', 'text', 
                                 'subjectivity_score', 'subjectivity_category', 'reference'])
    
    return df
