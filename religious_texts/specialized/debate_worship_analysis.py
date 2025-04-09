"""
Specialized Worship Analysis Module for Theological Debates

This module provides advanced functions for analyzing worship language in theological debates,
specifically focusing on the debate between David Wood and Alex O'Connor regarding
worship terminology and divine attribution.
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union, Any, Tuple, Set

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize

from religious_texts.specialized.worship_analysis import (
    extract_worship_instances,
    analyze_proskuneo_usage,
    compare_proskuneo_latreo,
    WORSHIP_TERMS,
    WORSHIP_RECIPIENTS
)

# Define specific theological debate claims about worship
DEBATE_CLAIMS = {
    "proskuneo_exclusivity": {
        "claim": "Proskuneo is a worship term exclusively used for God in Jewish tradition",
        "counter": "Proskuneo was used for humans and even objects in Jewish tradition",
        "related_terms": ["worship", "bow down", "prostrate", "homage"],
        "key_passages": ["Matthew 2:11", "Matthew 4:10", "Matthew 28:17", "John 9:38", "Revelation 22:8-9"]
    },
    "latreo_exclusivity": {
        "claim": "Latreo (service/worship) is exclusively used for God in Jewish tradition",
        "counter": "Latreo has broader usage patterns than suggested",
        "related_terms": ["serve", "service", "religious service", "sacred service"],
        "key_passages": ["Matthew 4:10", "Luke 2:37", "Acts 7:7", "Romans 1:9", "Philippians 3:3"]
    },
    "jesus_worship_significance": {
        "claim": "Jesus receiving proskuneo indicates his divinity given Jewish worship prohibitions",
        "counter": "Proskuneo to Jesus doesn't necessarily indicate divine worship",
        "related_terms": ["worship", "bow down", "homage", "reverence"],
        "key_passages": ["Matthew 28:9", "Matthew 28:17", "Luke 24:52", "John 9:38", "Hebrews 1:6"]
    },
    "worship_distinction": {
        "claim": "The distinction between proskuneo and latreo shows different levels of worship",
        "counter": "The terms overlap significantly and don't support such a clear distinction",
        "related_terms": ["worship", "serve", "reverence", "homage", "service"],
        "key_passages": ["Matthew 4:10", "John 4:20-24", "Romans 12:1", "Revelation 7:15", "Revelation 22:3"]
    }
}

def analyze_debate_claim(bible_dict: Dict[str, Any], claim_key: str) -> Dict[str, Any]:
    """
    Analyze biblical evidence for a specific debate claim about worship.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        claim_key: Key of the claim to analyze from DEBATE_CLAIMS
        
    Returns:
        Dictionary with analysis results for the claim
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Analyze evidence for the proskuneo exclusivity claim
        >>> analysis = analyze_debate_claim(bible, "proskuneo_exclusivity")
    """
    # Validate claim key
    if claim_key not in DEBATE_CLAIMS:
        raise ValueError(f"Invalid claim key: {claim_key}. Choose from {list(DEBATE_CLAIMS.keys())}")
    
    # Get claim details
    claim_details = DEBATE_CLAIMS[claim_key]
    claim_text = claim_details["claim"]
    counter_text = claim_details["counter"]
    related_terms = claim_details["related_terms"]
    key_passages = claim_details["key_passages"]
    
    # Parse key passages
    passage_refs = []
    for ref in key_passages:
        parts = ref.split()
        if len(parts) < 2:
            continue
            
        book = " ".join(parts[:-1])
        chapter_verse = parts[-1].split(":")
        
        if len(chapter_verse) < 2:
            continue
        
        # Handle verse ranges (e.g., 8-9)
        verse_parts = chapter_verse[1].split("-")
        if len(verse_parts) == 1:
            try:
                chapter = int(chapter_verse[0])
                verse = int(verse_parts[0])
                passage_refs.append((book, chapter, verse))
            except ValueError:
                continue
        else:
            try:
                chapter = int(chapter_verse[0])
                start_verse = int(verse_parts[0])
                end_verse = int(verse_parts[1])
                for verse in range(start_verse, end_verse + 1):
                    passage_refs.append((book, chapter, verse))
            except ValueError:
                continue
    
    # Extract passage texts
    passages = []
    for book, chapter, verse in passage_refs:
        if book in bible_dict and chapter in bible_dict[book] and verse in bible_dict[book][chapter]:
            passages.append({
                "reference": f"{book} {chapter}:{verse}",
                "text": bible_dict[book][chapter][verse]
            })
    
    # Analyze key terms in passages
    term_occurrences = defaultdict(list)
    for passage in passages:
        text = passage["text"].lower()
        for term in related_terms:
            pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            if pattern.search(text):
                term_occurrences[term].append(passage["reference"])
    
    # Determine if passage supports claim or counter
    # For this, we need more sophisticated analysis based on the claim type
    passage_support = {}
    
    if claim_key == "proskuneo_exclusivity":
        # For proskuneo exclusivity, check if the recipient is God, Jesus, or other
        for passage in passages:
            reference = passage["reference"]
            text = passage["text"]
            
            # Check for worship terms
            has_worship_term = any(re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE) 
                                 for term in related_terms)
            
            if not has_worship_term:
                passage_support[reference] = "neutral"
                continue
            
            # Check for recipient
            has_god = any(re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE) 
                        for term in WORSHIP_RECIPIENTS["god_terms"])
            
            has_jesus = any(re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE) 
                          for term in WORSHIP_RECIPIENTS["jesus_terms"])
            
            has_other = any(re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE) 
                          for term in WORSHIP_RECIPIENTS["other_recipients"])
            
            if has_other:
                # Proskuneo to non-divine entities counters exclusivity claim
                passage_support[reference] = "counter"
            elif has_jesus:
                # Proskuneo to Jesus is ambiguous for this claim
                # (supports claim if Jesus is God, counters if not)
                passage_support[reference] = "ambiguous"
            elif has_god:
                # Proskuneo to God supports exclusivity claim
                passage_support[reference] = "claim"
            else:
                # Unable to determine recipient
                passage_support[reference] = "neutral"
    
    elif claim_key == "latreo_exclusivity":
        # Similar logic for latreo exclusivity
        for passage in passages:
            reference = passage["reference"]
            text = passage["text"]
            
            # Check for service terms
            has_service_term = any(re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE) 
                                 for term in related_terms)
            
            if not has_service_term:
                passage_support[reference] = "neutral"
                continue
            
            # Check for recipient
            has_god = any(re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE) 
                        for term in WORSHIP_RECIPIENTS["god_terms"])
            
            has_jesus = any(re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE) 
                          for term in WORSHIP_RECIPIENTS["jesus_terms"])
            
            has_other = any(re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE) 
                          for term in WORSHIP_RECIPIENTS["other_recipients"])
            
            if has_other:
                # Latreo to non-divine entities counters exclusivity claim
                passage_support[reference] = "counter"
            elif has_jesus:
                # Latreo to Jesus is relevant for divinity claims
                passage_support[reference] = "claim"
            elif has_god:
                # Latreo to God supports exclusivity claim but doesn't disprove counter
                passage_support[reference] = "claim"
            else:
                # Unable to determine recipient
                passage_support[reference] = "neutral"
    
    elif claim_key == "jesus_worship_significance":
        # For Jesus worship significance, check if Jesus receives worship
        for passage in passages:
            reference = passage["reference"]
            text = passage["text"]
            
            # Check for worship terms
            has_worship_term = any(re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE) 
                                 for term in related_terms)
            
            if not has_worship_term:
                passage_support[reference] = "neutral"
                continue
            
            # Check for Jesus as recipient
            has_jesus = any(re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE) 
                          for term in WORSHIP_RECIPIENTS["jesus_terms"])
            
            # Check context for divine indicators
            divine_indicators = ['god', 'lord', 'divine', 'deity', 'creator', 'worship', 'son of god']
            has_divine_context = any(re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE) 
                                    for term in divine_indicators)
            
            if has_jesus and has_worship_term:
                if has_divine_context:
                    # Jesus receives worship in divine context
                    passage_support[reference] = "claim"
                else:
                    # Jesus receives worship without clear divine context
                    passage_support[reference] = "ambiguous"
            else:
                passage_support[reference] = "neutral"
    
    elif claim_key == "worship_distinction":
        # For worship distinction, check if both terms appear or are distinguished
        for passage in passages:
            reference = passage["reference"]
            text = passage["text"]
            
            # Check for proskuneo terms
            has_proskuneo = any(re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE) 
                              for term in WORSHIP_TERMS["proskuneo"])
            
            # Check for latreo terms
            has_latreo = any(re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE) 
                           for term in WORSHIP_TERMS["latreo"])
            
            if has_proskuneo and has_latreo:
                # Both terms appear together, relevant for distinction
                passage_support[reference] = "relevant"
            elif has_proskuneo or has_latreo:
                # Only one term appears, less relevant for distinction
                passage_support[reference] = "partial"
            else:
                passage_support[reference] = "neutral"
    
    # Calculate support statistics
    support_counts = Counter(passage_support.values())
    
    # Prepare analysis results
    results = {
        "claim": claim_text,
        "counter": counter_text,
        "related_terms": related_terms,
        "key_passages": [{
            "reference": passage["reference"],
            "text": passage["text"],
            "support": passage_support.get(passage["reference"], "neutral")
        } for passage in passages],
        "term_occurrences": dict(term_occurrences),
        "support_statistics": support_counts,
        "total_passages": len(passages)
    }
    
    return results

def analyze_debate_worship_terms(bible_dict: Dict[str, Any], *,
                              proskuneo_passages: Optional[List[str]] = None,
                              latreo_passages: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of worship terms relevant to theological debates.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        proskuneo_passages: Optional list of specific proskuneo passages to analyze
        latreo_passages: Optional list of specific latreo passages to analyze
        
    Returns:
        Dictionary with comprehensive worship term analysis
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Analyze specific passages relevant to the debate
        >>> proskuneo_refs = ["Matthew 2:11", "Matthew 28:17", "John 9:38"]
        >>> latreo_refs = ["Matthew 4:10", "Romans 1:9", "Revelation 22:3"]
        >>> analysis = analyze_debate_worship_terms(
        ...     bible,
        ...     proskuneo_passages=proskuneo_refs,
        ...     latreo_passages=latreo_refs
        ... )
    """
    # Default passages if none provided
    if proskuneo_passages is None:
        proskuneo_passages = [
            "Matthew 2:11", "Matthew 8:2", "Matthew 9:18", "Matthew 14:33",
            "Matthew 15:25", "Matthew 20:20", "Matthew 28:9", "Matthew 28:17",
            "Mark 5:6", "Luke 24:52", "John 9:38", "Hebrews 1:6",
            "Revelation 19:10", "Revelation 22:8-9"
        ]
    
    if latreo_passages is None:
        latreo_passages = [
            "Matthew 4:10", "Luke 1:74", "Luke 2:37", "Luke 4:8",
            "Acts 7:7", "Acts 24:14", "Romans 1:9", "Romans 1:25",
            "Philippians 3:3", "2 Timothy 1:3", "Hebrews 8:5", "Hebrews 9:14",
            "Hebrews 12:28", "Hebrews 13:10", "Revelation 7:15", "Revelation 22:3"
        ]
    
    # Combine and parse all references
    all_refs = proskuneo_passages + latreo_passages
    proskuneo_parsed = []
    latreo_parsed = []
    
    for ref in all_refs:
        parts = ref.split()
        if len(parts) < 2:
            continue
            
        book = " ".join(parts[:-1])
        chapter_verse = parts[-1].split(":")
        
        if len(chapter_verse) < 2:
            continue
        
        # Handle verse ranges (e.g., 8-9)
        verse_parts = chapter_verse[1].split("-")
        if len(verse_parts) == 1:
            try:
                chapter = int(chapter_verse[0])
                verse = int(verse_parts[0])
                ref_tuple = (book, chapter, verse)
                if ref in proskuneo_passages:
                    proskuneo_parsed.append(ref_tuple)
                if ref in latreo_passages:
                    latreo_parsed.append(ref_tuple)
            except ValueError:
                continue
        else:
            try:
                chapter = int(chapter_verse[0])
                start_verse = int(verse_parts[0])
                end_verse = int(verse_parts[1])
                for verse in range(start_verse, end_verse + 1):
                    ref_tuple = (book, chapter, verse)
                    if ref in proskuneo_passages:
                        proskuneo_parsed.append(ref_tuple)
                    if ref in latreo_passages:
                        latreo_parsed.append(ref_tuple)
            except ValueError:
                continue
    
    # Extract passages
    proskuneo_texts = []
    for book, chapter, verse in proskuneo_parsed:
        if book in bible_dict and chapter in bible_dict[book] and verse in bible_dict[book][chapter]:
            proskuneo_texts.append({
                "reference": f"{book} {chapter}:{verse}",
                "text": bible_dict[book][chapter][verse]
            })
    
    latreo_texts = []
    for book, chapter, verse in latreo_parsed:
        if book in bible_dict and chapter in bible_dict[book] and verse in bible_dict[book][chapter]:
            latreo_texts.append({
                "reference": f"{book} {chapter}:{verse}",
                "text": bible_dict[book][chapter][verse]
            })
    
    # Analyze recipients for each term
    proskuneo_recipients = {
        "god": 0,
        "jesus": 0,
        "other": 0,
        "unknown": 0
    }
    
    latreo_recipients = {
        "god": 0,
        "jesus": 0,
        "other": 0,
        "unknown": 0
    }
    
    # Process proskuneo passages
    for passage in proskuneo_texts:
        text = passage["text"]
        
        # Check for recipients
        has_god = any(re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE) 
                     for term in WORSHIP_RECIPIENTS["god_terms"])
        
        has_jesus = any(re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE) 
                       for term in WORSHIP_RECIPIENTS["jesus_terms"])
        
        has_other = any(re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE) 
                       for term in WORSHIP_RECIPIENTS["other_recipients"])
        
        if has_god and not has_jesus:
            proskuneo_recipients["god"] += 1
        elif has_jesus and not has_god:
            proskuneo_recipients["jesus"] += 1
        elif has_other:
            proskuneo_recipients["other"] += 1
        else:
            proskuneo_recipients["unknown"] += 1
    
    # Process latreo passages
    for passage in latreo_texts:
        text = passage["text"]
        
        # Check for recipients
        has_god = any(re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE) 
                     for term in WORSHIP_RECIPIENTS["god_terms"])
        
        has_jesus = any(re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE) 
                       for term in WORSHIP_RECIPIENTS["jesus_terms"])
        
        has_other = any(re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE) 
                       for term in WORSHIP_RECIPIENTS["other_recipients"])
        
        if has_god and not has_jesus:
            latreo_recipients["god"] += 1
        elif has_jesus and not has_god:
            latreo_recipients["jesus"] += 1
        elif has_other:
            latreo_recipients["other"] += 1
        else:
            latreo_recipients["unknown"] += 1
    
    # Compile New Testament statistics
    nt_books = [
        "Matthew", "Mark", "Luke", "John", "Acts", "Romans", "1 Corinthians", 
        "2 Corinthians", "Galatians", "Ephesians", "Philippians", "Colossians", 
        "1 Thessalonians", "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", 
        "Philemon", "Hebrews", "James", "1 Peter", "2 Peter", "1 John", "2 John", 
        "3 John", "Jude", "Revelation"
    ]
    
    # Get full NT proskuneo and latreo data
    full_comparison = compare_proskuneo_latreo(bible_dict, books=nt_books)
    
    # Prepare final analysis
    results = {
        "proskuneo_analysis": {
            "passages": proskuneo_texts,
            "recipient_distribution": proskuneo_recipients,
            "total_passages": len(proskuneo_texts)
        },
        "latreo_analysis": {
            "passages": latreo_texts,
            "recipient_distribution": latreo_recipients,
            "total_passages": len(latreo_texts)
        },
        "comparison": {
            "proskuneo_total": full_comparison["proskuneo_instances"],
            "latreo_total": full_comparison["latreo_instances"],
            "proskuneo_recipients": full_comparison["proskuneo_recipients"]["counts"],
            "latreo_recipients": full_comparison["latreo_recipients"]["counts"],
            "overlap_count": full_comparison["overlap_count"],
            "proskuneo_jesus_count": full_comparison["proskuneo_jesus_count"],
            "latreo_jesus_count": full_comparison["latreo_jesus_count"]
        },
        "debate_significance": {
            "proskuneo_exclusivity": proskuneo_recipients["other"] == 0,
            "latreo_exclusivity": latreo_recipients["other"] == 0,
            "jesus_proskuneo": proskuneo_recipients["jesus"] > 0,
            "jesus_latreo": latreo_recipients["jesus"] > 0
        }
    }
    
    return results

def create_worship_evidence_matrix(bible_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a matrix of worship language evidence for theological debates.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        
    Returns:
        DataFrame with evidence matrix for worship claims
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> evidence_df = create_worship_evidence_matrix(bible)
    """
    # Define the debate claims to analyze
    claims = list(DEBATE_CLAIMS.keys())
    
    # Analyze each claim
    claim_analyses = {}
    for claim in claims:
        claim_analyses[claim] = analyze_debate_claim(bible_dict, claim)
    
    # Define evidence categories
    evidence_categories = ["strong_support", "moderate_support", "neutral", "moderate_counter", "strong_counter"]
    
    # Initialize evidence matrix
    matrix_data = []
    
    for claim, analysis in claim_analyses.items():
        # Determine support for each claim based on passage analysis
        support = analysis["support_statistics"]
        
        # Calculate evidence strength (simplified logic)
        if claim in ["proskuneo_exclusivity", "latreo_exclusivity"]:
            claim_count = support.get("claim", 0)
            counter_count = support.get("counter", 0)
            
            if counter_count > 0:
                # Any counter evidence weakens exclusivity claims
                evidence = "strong_counter" if counter_count > 2 else "moderate_counter"
            elif claim_count > 5:
                evidence = "strong_support"
            elif claim_count > 2:
                evidence = "moderate_support"
            else:
                evidence = "neutral"
        
        elif claim == "jesus_worship_significance":
            claim_count = support.get("claim", 0)
            ambiguous_count = support.get("ambiguous", 0)
            
            if claim_count > 3:
                evidence = "strong_support"
            elif claim_count > 1:
                evidence = "moderate_support"
            elif ambiguous_count > 3:
                evidence = "moderate_counter"
            else:
                evidence = "neutral"
        
        elif claim == "worship_distinction":
            relevant_count = support.get("relevant", 0)
            partial_count = support.get("partial", 0)
            
            if relevant_count > 3:
                evidence = "strong_support"
            elif relevant_count > 1:
                evidence = "moderate_support"
            elif partial_count > 5:
                evidence = "moderate_counter"
            else:
                evidence = "neutral"
        
        # Create matrix row
        row = {
            "claim": DEBATE_CLAIMS[claim]["claim"],
            "evidence_category": evidence,
            "support_count": support.get("claim", 0) + support.get("relevant", 0),
            "counter_count": support.get("counter", 0),
            "neutral_count": support.get("neutral", 0) + support.get("ambiguous", 0) + support.get("partial", 0),
            "total_passages": analysis["total_passages"],
            "claim_key": claim
        }
        
        matrix_data.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(matrix_data)
    
    return df

def analyze_debate_passages_context(bible_dict: Dict[str, Any], 
                                 passage_refs: List[str], 
                                 context_verses: int = 2) -> Dict[str, Any]:
    """
    Analyze the context of specific passages cited in theological debates.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        passage_refs: List of verse references to analyze
        context_verses: Number of verses before/after to include
        
    Returns:
        Dictionary with contextual analysis of debate passages
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> debate_passages = ["Matthew 28:17", "John 9:38", "Hebrews 1:6"]
        >>> context_analysis = analyze_debate_passages_context(bible, debate_passages)
    """
    # Parse references
    parsed_refs = []
    
    for ref in passage_refs:
        parts = ref.split()
        if len(parts) < 2:
            continue
            
        book = " ".join(parts[:-1])
        chapter_verse = parts[-1].split(":")
        
        if len(chapter_verse) != 2:
            continue
            
        try:
            chapter = int(chapter_verse[0])
            verse = int(chapter_verse[1])
            parsed_refs.append((book, chapter, verse))
        except ValueError:
            continue
    
    # Initialize results
    results = {
        "passages": []
    }
    
    # Process each passage
    for book, chapter, verse in parsed_refs:
        # Skip if book/chapter/verse not in bible_dict
        if book not in bible_dict or chapter not in bible_dict[book] or verse not in bible_dict[book][chapter]:
            continue
        
        # Get the verse text
        verse_text = bible_dict[book][chapter][verse]
        reference = f"{book} {chapter}:{verse}"
        
        # Get preceding context
        preceding_verses = []
        for v in range(max(1, verse - context_verses), verse):
            if v in bible_dict[book][chapter]:
                preceding_verses.append({
                    "reference": f"{book} {chapter}:{v}",
                    "text": bible_dict[book][chapter][v]
                })
        
        # Get following context
        following_verses = []
        max_verse = max(bible_dict[book][chapter].keys())
        for v in range(verse + 1, min(max_verse + 1, verse + context_verses + 1)):
            if v in bible_dict[book][chapter]:
                following_verses.append({
                    "reference": f"{book} {chapter}:{v}",
                    "text": bible_dict[book][chapter][v]
                })
        
        # Check for worship terms
        worship_terms = []
        for category, terms in WORSHIP_TERMS.items():
            category_terms = []
            for term in terms:
                pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
                if pattern.search(verse_text):
                    category_terms.append(term)
            
            if category_terms:
                worship_terms.append({
                    "category": category,
                    "terms": category_terms
                })
        
        # Check for recipients
        recipients = []
        for category, terms in WORSHIP_RECIPIENTS.items():
            category_terms = []
            for term in terms:
                pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
                if pattern.search(verse_text):
                    category_terms.append(term)
            
            if category_terms:
                recipients.append({
                    "category": category,
                    "terms": category_terms
                })
        
        # Analyze context for themes
        context_text = " ".join([v["text"] for v in preceding_verses + following_verses])
        
        # Check for theological themes in context
        themes = []
        theological_themes = {
            "divinity": ["god", "divine", "glory", "power", "authority", "heaven", "creator", "lord"],
            "worship": ["worship", "bow", "honor", "praise", "glorify", "reverence", "adore"],
            "messianic": ["messiah", "christ", "son of god", "king", "prophet", "savior", "kingdom"],
            "salvation": ["save", "redeem", "deliver", "forgive", "sin", "sacrifice", "blood"]
        }
        
        for theme, keywords in theological_themes.items():
            theme_matches = []
            for keyword in keywords:
                pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                if pattern.search(context_text):
                    theme_matches.append(keyword)
            
            if theme_matches:
                themes.append({
                    "theme": theme,
                    "keywords": theme_matches
                })
        
        # Add to results
        results["passages"].append({
            "reference": reference,
            "text": verse_text,
            "preceding_context": preceding_verses,
            "following_context": following_verses,
            "worship_terms": worship_terms,
            "recipients": recipients,
            "themes": themes
        })
    
    # Add summary statistics
    results["total_passages"] = len(results["passages"])
    
    worship_categories = {category: 0 for category in WORSHIP_TERMS}
    recipient_categories = {category: 0 for category in WORSHIP_RECIPIENTS}
    theme_counts = defaultdict(int)
    
    for passage in results["passages"]:
        for worship in passage["worship_terms"]:
            worship_categories[worship["category"]] += 1
        
        for recipient in passage["recipients"]:
            recipient_categories[recipient["category"]] += 1
        
        for theme in passage["themes"]:
            theme_counts[theme["theme"]] += 1
    
    results["worship_category_counts"] = worship_categories
    results["recipient_category_counts"] = recipient_categories
    results["theme_counts"] = dict(theme_counts)
    
    return results
