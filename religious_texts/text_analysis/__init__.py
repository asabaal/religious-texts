"""
Text Analysis Module

This module provides functions for analyzing biblical texts including:
- Word and phrase frequency analysis
- Concordance generation
- Co-occurrence pattern detection
- Sentiment analysis
- Named entity recognition
"""

from religious_texts.text_analysis.frequency import (
    word_frequency,
    phrase_frequency,
    frequency_distribution,
    relative_frequency,
    comparative_frequency
)

from religious_texts.text_analysis.concordance import (
    generate_concordance,
    keyword_in_context,
    find_all_occurrences
)

from religious_texts.text_analysis.cooccurrence import (
    word_cooccurrence,
    concept_cooccurrence,
    proximity_analysis
)

from religious_texts.text_analysis.sentiment import (
    sentiment_analysis,
    emotion_detection,
    subjectivity_analysis
)

from religious_texts.text_analysis.entities import (
    extract_entities,
    identify_people,
    identify_places,
    entity_frequency
)

__all__ = [
    # Frequency analysis
    'word_frequency',
    'phrase_frequency',
    'frequency_distribution',
    'relative_frequency',
    'comparative_frequency',
    
    # Concordance
    'generate_concordance',
    'keyword_in_context',
    'find_all_occurrences',
    
    # Co-occurrence
    'word_cooccurrence',
    'concept_cooccurrence',
    'proximity_analysis',
    
    # Sentiment
    'sentiment_analysis',
    'emotion_detection',
    'subjectivity_analysis',
    
    # Entities
    'extract_entities',
    'identify_people',
    'identify_places',
    'entity_frequency'
]
