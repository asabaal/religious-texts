"""
Sentiment Analysis Module

This module provides functions for analyzing the sentiment and emotional content
of biblical passages, which can help understand the tone, emotional appeals, and
rhetorical strategies used in different texts.
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union, Any, Tuple, Set

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Define emotion and sentiment lexicons
POSITIVE_LEXICON = {
    "joy": ["joy", "rejoice", "glad", "delight", "happy", "blessed", "praise", "exalt", 
           "thank", "gratitude", "celebrate", "honor", "glory", "peace"],
    "love": ["love", "beloved", "compassion", "mercy", "kindness", "grace", "gentle",
            "comfort", "care", "affection", "charity", "devoted"],
    "hope": ["hope", "promise", "faith", "trust", "believe", "confidence", "assurance",
            "salvation", "redemption", "renewal", "restore", "covenant"],
    "moral_approval": ["righteous", "just", "holy", "pure", "good", "worthy", "noble",
                     "true", "honest", "faithful", "virtue", "integrity", "honor"]
}

NEGATIVE_LEXICON = {
    "fear": ["fear", "afraid", "terror", "dread", "anxiety", "alarm", "panic", "horror",
            "tremble", "fright", "scared", "apprehension", "worry"],
    "anger": ["anger", "wrath", "fury", "rage", "indignation", "vengeance", "judgment", 
             "rebuke", "condemn", "punish", "curse", "destroy"],
    "sorrow": ["sorrow", "grief", "mourn", "weep", "lament", "sad", "distress", "affliction",
              "suffering", "pain", "anguish", "broken", "tear"],
    "moral_disapproval": ["sin", "evil", "wicked", "corrupt", "abomination", "unclean", "impure",
                        "transgression", "iniquity", "wrong", "guilt", "shame", "disgrace"]
}

def analyze_passage_sentiment(text: str, detailed: bool = False) -> Dict[str, Any]:
    """
    Analyze the sentiment and emotional content of a biblical passage.
    
    Args:
        text: The text to analyze
        detailed: Whether to return detailed emotion analysis
        
    Returns:
        Dictionary with sentiment analysis results
        
    Example:
        >>> # Analyze sentiment of Psalm 23
        >>> psalm23 = "The LORD is my shepherd; I shall not want..."
        >>> sentiment = analyze_passage_sentiment(psalm23, detailed=True)
        >>> print(sentiment["compound_score"])
    """
    # Initialize NLTK's VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Get basic sentiment scores
    scores = sia.polarity_scores(text)
    
    # Initialize result
    result = {
        "compound_score": scores["compound"],
        "positive_score": scores["pos"],
        "negative_score": scores["neg"],
        "neutral_score": scores["neu"],
        "overall_sentiment": "positive" if scores["compound"] >= 0.05 else 
                           "negative" if scores["compound"] <= -0.05 else "neutral"
    }
    
    # Return basic analysis if not detailed
    if not detailed:
        return result
    
    # Tokenize text
    tokens = [token.lower() for token in word_tokenize(text)]
    
    # Analyze emotions using lexicons
    emotions = {}
    
    # Count positive emotions
    for emotion, terms in POSITIVE_LEXICON.items():
        count = sum(tokens.count(term) for term in terms)
        emotions[emotion] = count
    
    # Count negative emotions
    for emotion, terms in NEGATIVE_LEXICON.items():
        count = sum(tokens.count(term) for term in terms)
        emotions[emotion] = count
    
    # Calculate dominant emotions
    if emotions:
        max_emotion = max(emotions.items(), key=lambda x: x[1])
        
        if max_emotion[1] > 0:
            dominant_emotion = max_emotion[0]
        else:
            dominant_emotion = "neutral"
    else:
        dominant_emotion = "neutral"
    
    # Get sentence-level analysis
    sentences = sent_tokenize(text)
    sentence_analysis = []
    
    for sent in sentences:
        sent_scores = sia.polarity_scores(sent)
        
        # Create TextBlob for subjectivity
        blob = TextBlob(sent)
        
        sent_emotions = {}
        sent_tokens = [token.lower() for token in word_tokenize(sent)]
        
        # Count emotions in sentence
        for category in POSITIVE_LEXICON:
            terms = POSITIVE_LEXICON[category]
            count = sum(sent_tokens.count(term) for term in terms)
            sent_emotions[category] = count
        
        for category in NEGATIVE_LEXICON:
            terms = NEGATIVE_LEXICON[category]
            count = sum(sent_tokens.count(term) for term in terms)
            sent_emotions[category] = count
        
        # Add to sentence analysis
        sentence_analysis.append({
            "text": sent,
            "compound_score": sent_scores["compound"],
            "positive_score": sent_scores["pos"],
            "negative_score": sent_scores["neg"],
            "neutral_score": sent_scores["neu"],
            "subjectivity": blob.sentiment.subjectivity,
            "emotions": sent_emotions
        })
    
    # Add detailed results
    result["emotions"] = emotions
    result["dominant_emotion"] = dominant_emotion
    result["sentence_analysis"] = sentence_analysis
    result["subjectivity"] = sum(s["subjectivity"] for s in sentence_analysis) / len(sentence_analysis) if sentence_analysis else 0
    
    return result

def compare_passage_sentiments(passages: List[Dict[str, str]]) -> pd.DataFrame:
    """
    Compare sentiment and emotional content across multiple passages.
    
    Args:
        passages: List of dictionaries with passage information
        
    Returns:
        DataFrame with comparative sentiment analysis
        
    Example:
        >>> # Compare sentiments across beatitudes and woes
        >>> passages = [
        ...     {"label": "Beatitudes", "text": "Blessed are the poor in spirit..."},
        ...     {"label": "Woes", "text": "Woe to you, scribes and Pharisees..."}
        ... ]
        >>> comparison = compare_passage_sentiments(passages)
    """
    # Initialize results
    results = []
    
    # Process each passage
    for passage in passages:
        label = passage.get("label", "Unnamed")
        text = passage.get("text", "")
        reference = passage.get("reference", "")
        
        if not text:
            continue
        
        # Get sentiment analysis
        sentiment = analyze_passage_sentiment(text, detailed=True)
        
        # Extract key metrics
        result = {
            "label": label,
            "reference": reference,
            "text_sample": text[:100] + "..." if len(text) > 100 else text,
            "compound_score": sentiment["compound_score"],
            "positive_score": sentiment["positive_score"],
            "negative_score": sentiment["negative_score"],
            "subjectivity": sentiment["subjectivity"],
            "dominant_emotion": sentiment["dominant_emotion"]
        }
        
        # Add emotion counts
        for emotion, count in sentiment["emotions"].items():
            result[f"emotion_{emotion}"] = count
        
        results.append(result)
    
    # Convert to DataFrame
    if results:
        df = pd.DataFrame(results)
        
        # Sort by compound score (descending)
        df = df.sort_values("compound_score", ascending=False)
    else:
        # Create empty DataFrame with expected columns
        columns = ["label", "reference", "text_sample", "compound_score", 
                  "positive_score", "negative_score", "subjectivity", "dominant_emotion"]
        df = pd.DataFrame(columns=columns)
    
    return df

def analyze_sentiment_patterns(bible_dict: Dict[str, Any], 
                             target_pattern: Optional[Dict[str, List[str]]] = None,
                             books: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Analyze sentiment patterns across biblical texts, optionally filtering by specific patterns.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        target_pattern: Optional dictionary mapping concepts to related terms for targeting analysis
        books: Optional list of books to include
        
    Returns:
        DataFrame with sentiment pattern analysis
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Analyze sentiment patterns in Jesus's teachings
        >>> patterns = {"teaching": ["said", "teach", "taught", "speaks", "spoke"]}
        >>> sentiment_patterns = analyze_sentiment_patterns(
        ...     bible,
        ...     target_pattern=patterns,
        ...     books=["Matthew", "Mark", "Luke", "John"]
        ... )
    """
    results = []
    
    # Determine which books to include
    if books:
        book_subset = {book: bible_dict[book] for book in books if book in bible_dict}
    else:
        book_subset = bible_dict
    
    # Process each book
    for book_name, chapters in book_subset.items():
        for chapter_num, verses in chapters.items():
            for verse_num, verse_text in verses.items():
                # Skip empty verses
                if not verse_text:
                    continue
                
                # Check if verse matches target pattern
                pattern_match = False
                matched_terms = []
                
                if target_pattern:
                    verse_lower = verse_text.lower()
                    
                    for concept, terms in target_pattern.items():
                        for term in terms:
                            if term.lower() in verse_lower:
                                pattern_match = True
                                matched_terms.append(term)
                
                # If no pattern specified or pattern matched, analyze sentiment
                if not target_pattern or pattern_match:
                    # Get sentiment analysis
                    sentiment = analyze_passage_sentiment(verse_text)
                    
                    # Add to results
                    results.append({
                        "book": book_name,
                        "chapter": chapter_num,
                        "verse": verse_num,
                        "reference": f"{book_name} {chapter_num}:{verse_num}",
                        "text": verse_text,
                        "compound_score": sentiment["compound_score"],
                        "sentiment": sentiment["overall_sentiment"],
                        "pattern_match": pattern_match,
                        "matched_terms": ", ".join(matched_terms) if matched_terms else None
                    })
    
    # Convert to DataFrame
    if results:
        df = pd.DataFrame(results)
    else:
        # Create empty DataFrame with expected columns
        columns = ["book", "chapter", "verse", "reference", "text", 
                  "compound_score", "sentiment", "pattern_match", "matched_terms"]
        df = pd.DataFrame(columns=columns)
    
    return df

def analyze_sentiment_by_speaker(bible_dict: Dict[str, Any],
                               speakers: Dict[str, List[str]],
                               books: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Analyze sentiment patterns in text attributed to different speakers.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        speakers: Dictionary mapping speaker labels to speech attribution terms
        books: Optional list of books to include
        
    Returns:
        DataFrame with sentiment analysis by speaker
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Compare sentiment in Jesus vs. Pharisees speech
        >>> speakers = {
        ...     "Jesus": ["Jesus said", "Jesus answered", "He said to them"],
        ...     "Pharisees": ["Pharisees said", "they answered", "they asked him"]
        ... }
        >>> speaker_sentiment = analyze_sentiment_by_speaker(
        ...     bible,
        ...     speakers=speakers,
        ...     books=["Matthew", "Mark", "Luke", "John"]
        ... )
    """
    from religious_texts.theological_analysis.speech import extract_speech_by_speaker
    
    results = []
    
    # Extract speech for each speaker
    for speaker_label, attribution_terms in speakers.items():
        # Get speech attributed to this speaker
        speech_df = extract_speech_by_speaker(bible_dict, attribution_terms, books=books)
        
        if speech_df.empty:
            continue
        
        # Process each speech segment
        for _, row in speech_df.iterrows():
            # Get sentiment analysis
            sentiment = analyze_passage_sentiment(row["speech_text"])
            
            # Add to results
            results.append({
                "speaker": speaker_label,
                "book": row["book"],
                "chapter": row["chapter"],
                "verse": row["verse"],
                "reference": row["reference"],
                "speech_text": row["speech_text"],
                "compound_score": sentiment["compound_score"],
                "positive_score": sentiment["positive_score"],
                "negative_score": sentiment["negative_score"],
                "sentiment": sentiment["overall_sentiment"]
            })
    
    # Convert to DataFrame
    if results:
        df = pd.DataFrame(results)
    else:
        # Create empty DataFrame with expected columns
        columns = ["speaker", "book", "chapter", "verse", "reference", "speech_text",
                  "compound_score", "positive_score", "negative_score", "sentiment"]
        df = pd.DataFrame(columns=columns)
    
    return df

def get_sentiment_summary_by_book(bible_dict: Dict[str, Any],
                                books: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Generate a summary of sentiment patterns across biblical books.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        books: Optional list of books to include
        
    Returns:
        DataFrame with sentiment summary by book
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Get sentiment summary for all Gospels
        >>> summary = get_sentiment_summary_by_book(
        ...     bible,
        ...     books=["Matthew", "Mark", "Luke", "John"]
        ... )
    """
    results = []
    
    # Determine which books to include
    if books:
        book_subset = {book: bible_dict[book] for book in books if book in bible_dict}
    else:
        book_subset = bible_dict
    
    # Process each book
    for book_name, chapters in book_subset.items():
        # Combine all text in the book
        book_text = ""
        
        for chapter_verses in chapters.values():
            for verse_text in chapter_verses.values():
                if verse_text:
                    book_text += verse_text + " "
        
        # Skip empty books
        if not book_text:
            continue
        
        # Get sentiment analysis
        sentiment = analyze_passage_sentiment(book_text, detailed=True)
        
        # Count verses by sentiment
        verse_sentiments = []
        total_verses = 0
        
        for chapter_num, verses in chapters.items():
            for verse_num, verse_text in verses.items():
                if verse_text:
                    total_verses += 1
                    verse_sentiment = analyze_passage_sentiment(verse_text)
                    verse_sentiments.append(verse_sentiment["overall_sentiment"])
        
        # Count by sentiment category
        sentiment_counts = Counter(verse_sentiments)
        
        positive_percent = sentiment_counts.get("positive", 0) / total_verses * 100 if total_verses else 0
        negative_percent = sentiment_counts.get("negative", 0) / total_verses * 100 if total_verses else 0
        neutral_percent = sentiment_counts.get("neutral", 0) / total_verses * 100 if total_verses else 0
        
        # Add dominant emotions
        emotions = sentiment.get("emotions", {})
        top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        top_emotions = [f"{emotion} ({count})" for emotion, count in top_emotions[:3] if count > 0]
        
        # Add to results
        results.append({
            "book": book_name,
            "total_verses": total_verses,
            "compound_score": sentiment["compound_score"],
            "positive_score": sentiment["positive_score"],
            "negative_score": sentiment["negative_score"],
            "neutral_score": sentiment["neutral_score"],
            "overall_sentiment": sentiment["overall_sentiment"],
            "positive_verses_percent": positive_percent,
            "negative_verses_percent": negative_percent,
            "neutral_verses_percent": neutral_percent,
            "dominant_emotions": ", ".join(top_emotions) if top_emotions else "None detected"
        })
    
    # Convert to DataFrame
    if results:
        df = pd.DataFrame(results)
        
        # Sort alphabetically by book name
        df = df.sort_values("book")
    else:
        # Create empty DataFrame with expected columns
        columns = ["book", "total_verses", "compound_score", "positive_score", 
                  "negative_score", "neutral_score", "overall_sentiment",
                  "positive_verses_percent", "negative_verses_percent", 
                  "neutral_verses_percent", "dominant_emotions"]
        df = pd.DataFrame(columns=columns)
    
    return df
