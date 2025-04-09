"""
Context Analysis Module

This module provides functions for analyzing the context of biblical passages
quoted in debates, examining whether passages are used in a way consistent with
their full context.
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union, Any, Tuple, Set

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

def extract_passage_context(bible_dict: Dict[str, Any],
                          reference: str,
                          context_size: int = 3) -> Dict[str, Any]:
    """
    Extract the context surrounding a referenced biblical passage.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        reference: Verse reference (e.g., "John 3:16")
        context_size: Number of verses before and after to include
        
    Returns:
        Dictionary with verse and its context
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Get context for John 10:30
        >>> context = extract_passage_context(bible, "John 10:30", context_size=5)
        >>> print(context["preceding_context"])
        >>> print(context["verse_text"])
        >>> print(context["following_context"])
    """
    result = {
        "reference": reference,
        "verse_text": None,
        "preceding_context": [],
        "following_context": [],
        "chapter_context": None
    }
    
    # Parse reference
    parts = reference.split()
    if len(parts) < 2:
        return result
    
    # Handle multi-word book names (e.g., "1 Kings")
    book = " ".join(parts[:-1])
    chapter_verse = parts[-1].split(":")
    
    if len(chapter_verse) != 2:
        return result
    
    try:
        chapter = int(chapter_verse[0])
        verse = int(chapter_verse[1])
    except ValueError:
        return result
    
    # Check if book and chapter exist
    if book not in bible_dict or chapter not in bible_dict[book]:
        return result
    
    # Get verse text
    if verse in bible_dict[book][chapter]:
        result["verse_text"] = bible_dict[book][chapter][verse]
    else:
        return result
    
    # Get preceding context
    preceding = []
    for v in range(max(1, verse - context_size), verse):
        if v in bible_dict[book][chapter]:
            preceding.append({
                "verse": v,
                "text": bible_dict[book][chapter][v],
                "reference": f"{book} {chapter}:{v}"
            })
    result["preceding_context"] = preceding
    
    # Get following context
    following = []
    max_verse = max(bible_dict[book][chapter].keys())
    for v in range(verse + 1, min(max_verse + 1, verse + context_size + 1)):
        if v in bible_dict[book][chapter]:
            following.append({
                "verse": v,
                "text": bible_dict[book][chapter][v],
                "reference": f"{book} {chapter}:{v}"
            })
    result["following_context"] = following
    
    # Get full chapter context
    chapter_verses = []
    for v in sorted(bible_dict[book][chapter].keys()):
        chapter_verses.append({
            "verse": v,
            "text": bible_dict[book][chapter][v],
            "reference": f"{book} {chapter}:{v}"
        })
    result["chapter_context"] = chapter_verses
    
    return result

def analyze_quoted_context(bible_dict: Dict[str, Any],
                         quoted_reference: str,
                         quoted_interpretation: str,
                         context_size: int = 5) -> Dict[str, Any]:
    """
    Analyze whether a quoted passage is used consistently with its broader context.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        quoted_reference: Verse reference as quoted in debate
        quoted_interpretation: The interpretation or claim made about the verse
        context_size: Number of verses before and after to analyze
        
    Returns:
        Dictionary with context analysis results
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Analyze how John 10:30 is used in a debate
        >>> analysis = analyze_quoted_context(
        ...     bible,
        ...     "John 10:30",
        ...     "Jesus claims to be equal to God in essence and nature"
        ... )
        >>> print(analysis["contextual_consistency"])
    """
    # Get the context
    context = extract_passage_context(bible_dict, quoted_reference, context_size)
    
    if not context["verse_text"]:
        return {
            "reference": quoted_reference,
            "quoted_interpretation": quoted_interpretation,
            "valid_reference": False,
            "error": "Reference not found in the text"
        }
    
    # Get verse and combined context
    verse_text = context["verse_text"]
    
    preceding_text = " ".join([v["text"] for v in context["preceding_context"]])
    following_text = " ".join([v["text"] for v in context["following_context"]])
    
    # Tokenize and get key terms from the verse and interpretation
    stop_words = set(stopwords.words('english'))
    
    verse_tokens = [w.lower() for w in word_tokenize(verse_text) if w.lower() not in stop_words and w.isalnum()]
    verse_terms = Counter(verse_tokens)
    
    interp_tokens = [w.lower() for w in word_tokenize(quoted_interpretation) if w.lower() not in stop_words and w.isalnum()]
    interp_terms = Counter(interp_tokens)
    
    preceding_tokens = [w.lower() for w in word_tokenize(preceding_text) if w.lower() not in stop_words and w.isalnum()]
    preceding_terms = Counter(preceding_tokens)
    
    following_tokens = [w.lower() for w in word_tokenize(following_text) if w.lower() not in stop_words and w.isalnum()]
    following_terms = Counter(following_tokens)
    
    # Calculate term overlap between interpretation and various contexts
    verse_overlap = sum((interp_terms & verse_terms).values())
    preceding_overlap = sum((interp_terms & preceding_terms).values())
    following_overlap = sum((interp_terms & following_terms).values())
    
    # Normalize by interpretation length
    interp_term_count = sum(interp_terms.values())
    
    if interp_term_count > 0:
        verse_overlap_norm = verse_overlap / interp_term_count
        preceding_overlap_norm = preceding_overlap / interp_term_count
        following_overlap_norm = following_overlap / interp_term_count
    else:
        verse_overlap_norm = 0
        preceding_overlap_norm = 0
        following_overlap_norm = 0
    
    # Check for key terms from verse that aren't in the interpretation
    verse_unique = set(verse_terms.keys()) - set(interp_terms.keys())
    missing_key_terms = []
    
    # Simply identify the most frequent terms that are missing
    for term in sorted(verse_unique, key=lambda t: verse_terms[t], reverse=True)[:5]:
        if verse_terms[t] > 1:  # Only include terms that appear multiple times
            missing_key_terms.append(term)
    
    # Calculate contextual consistency score (0-10)
    # Higher weight on verse overlap, but also account for context
    contextual_score = (verse_overlap_norm * 5) + (preceding_overlap_norm * 2.5) + (following_overlap_norm * 2.5)
    contextual_score = min(10, contextual_score * 10)  # Scale to 0-10
    
    # Determine factors that could affect interpretation
    interpretation_factors = []
    
    # Check for specific genre markers
    if "parable" in preceding_text.lower() or "parable" in verse_text.lower():
        interpretation_factors.append("Parable context")
    
    if "answered" in preceding_text.lower() or "question" in preceding_text.lower():
        interpretation_factors.append("Response to question")
    
    if "scripture" in verse_text.lower() or "written" in verse_text.lower():
        interpretation_factors.append("Reference to earlier scripture")
    
    # Check for recurring themes in surrounding context
    context_combined = preceding_text + " " + following_text
    context_tokens = [w.lower() for w in word_tokenize(context_combined) if w.lower() not in stop_words and w.isalnum()]
    context_terms = Counter(context_tokens)
    
    # Identify recurring themes (terms that appear multiple times in context)
    recurring_themes = [term for term, count in context_terms.most_common(5) if count >= 3]
    
    # Check if any of these themes are missing from the interpretation
    missing_themes = [theme for theme in recurring_themes if theme not in interp_terms]
    
    # Prepare results
    result = {
        "reference": quoted_reference,
        "verse_text": verse_text,
        "quoted_interpretation": quoted_interpretation,
        "valid_reference": True,
        "contextual_consistency": contextual_score,
        "preceding_context_overlap": preceding_overlap_norm,
        "following_context_overlap": following_overlap_norm,
        "verse_overlap": verse_overlap_norm,
        "missing_key_terms": missing_key_terms,
        "interpretation_factors": interpretation_factors,
        "recurring_themes": recurring_themes,
        "missing_themes": missing_themes,
        "preceding_context": context["preceding_context"],
        "following_context": context["following_context"]
    }
    
    return result

def compare_passage_interpretations(bible_dict: Dict[str, Any],
                                  reference: str,
                                  interpretations: List[Dict[str, str]]) -> pd.DataFrame:
    """
    Compare multiple interpretations of the same passage based on contextual consistency.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        reference: Verse reference to analyze
        interpretations: List of dictionaries with interpretation details
        
    Returns:
        DataFrame comparing the interpretations
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Compare different interpretations of John 17:5
        >>> interpretations = [
        ...     {"label": "Literal Preexistence", "text": "Jesus existed literally before creation"},
        ...     {"label": "Ideal Preexistence", "text": "Jesus existed in God's plan before creation"}
        ... ]
        >>> comparison = compare_passage_interpretations(bible, "John 17:5", interpretations)
    """
    # Get the context
    context = extract_passage_context(bible_dict, reference, context_size=5)
    
    if not context["verse_text"]:
        # Create empty DataFrame with expected columns
        columns = ["reference", "interpretation_label", "interpretation_text",
                  "contextual_score", "verse_overlap", "context_overlap", 
                  "missing_terms", "supporting_context"]
        return pd.DataFrame(columns=columns)
    
    # Extract context text
    verse_text = context["verse_text"]
    preceding_text = " ".join([v["text"] for v in context["preceding_context"]])
    following_text = " ".join([v["text"] for v in context["following_context"]])
    full_context = preceding_text + " " + verse_text + " " + following_text
    
    # Process each interpretation
    results = []
    
    for interp in interpretations:
        label = interp.get("label", "")
        text = interp.get("text", "")
        
        # Skip empty interpretations
        if not text:
            continue
        
        # Analyze this interpretation
        analysis = analyze_quoted_context(bible_dict, reference, text)
        
        # Identify most supportive context verses
        supporting_verses = []
        
        # Tokenize the interpretation
        stop_words = set(stopwords.words('english'))
        interp_tokens = [w.lower() for w in word_tokenize(text) if w.lower() not in stop_words and w.isalnum()]
        interp_terms = Counter(interp_tokens)
        
        # Check preceding context
        for verse_data in context["preceding_context"]:
            verse_tokens = [w.lower() for w in word_tokenize(verse_data["text"]) 
                           if w.lower() not in stop_words and w.isalnum()]
            verse_terms = Counter(verse_tokens)
            
            overlap = sum((interp_terms & verse_terms).values())
            if overlap > 0:
                supporting_verses.append({
                    "reference": verse_data["reference"],
                    "text": verse_data["text"],
                    "overlap": overlap
                })
        
        # Check following context
        for verse_data in context["following_context"]:
            verse_tokens = [w.lower() for w in word_tokenize(verse_data["text"]) 
                           if w.lower() not in stop_words and w.isalnum()]
            verse_terms = Counter(verse_tokens)
            
            overlap = sum((interp_terms & verse_terms).values())
            if overlap > 0:
                supporting_verses.append({
                    "reference": verse_data["reference"],
                    "text": verse_data["text"],
                    "overlap": overlap
                })
        
        # Sort supporting verses by overlap
        supporting_verses.sort(key=lambda x: x["overlap"], reverse=True)
        
        # Add to results
        results.append({
            "reference": reference,
            "verse_text": verse_text,
            "interpretation_label": label,
            "interpretation_text": text,
            "contextual_score": analysis["contextual_consistency"],
            "verse_overlap": analysis["verse_overlap"],
            "context_overlap": (analysis["preceding_context_overlap"] + analysis["following_context_overlap"]) / 2,
            "missing_terms": ", ".join(analysis["missing_key_terms"]),
            "missing_themes": ", ".join(analysis["missing_themes"]),
            "supporting_context": supporting_verses[:3]  # Top 3 supporting verses
        })
    
    if results:
        return pd.DataFrame(results)
    else:
        # Create empty DataFrame with expected columns
        columns = ["reference", "verse_text", "interpretation_label", "interpretation_text",
                  "contextual_score", "verse_overlap", "context_overlap", 
                  "missing_terms", "missing_themes", "supporting_context"]
        return pd.DataFrame(columns=columns)

def analyze_cross_reference_consistency(bible_dict: Dict[str, Any],
                                      primary_reference: str,
                                      cross_references: List[str],
                                      interpretation: str) -> Dict[str, Any]:
    """
    Analyze how consistently an interpretation applies across multiple related passages.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        primary_reference: Main verse reference being interpreted
        cross_references: List of related verse references
        interpretation: The interpretation being evaluated
        
    Returns:
        Dictionary with cross-reference analysis results
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Check if trinitarian reading of John 10:30 is consistent with related verses
        >>> analysis = analyze_cross_reference_consistency(
        ...     bible,
        ...     "John 10:30",
        ...     ["John 17:11", "John 17:22"],
        ...     "Jesus claims to be equal to God in essence and nature"
        ... )
    """
    # Initialize results
    result = {
        "primary_reference": primary_reference,
        "interpretation": interpretation,
        "cross_references": [],
        "consistency_score": 0,
        "thematic_alignment": [],
        "contradictory_references": []
    }
    
    # Get primary reference details
    primary_analysis = analyze_quoted_context(bible_dict, primary_reference, interpretation)
    
    if not primary_analysis.get("valid_reference", False):
        result["error"] = "Primary reference not found"
        return result
    
    # Extract key terms from interpretation
    stop_words = set(stopwords.words('english'))
    interp_tokens = [w.lower() for w in word_tokenize(interpretation) 
                    if w.lower() not in stop_words and w.isalnum()]
    interp_terms = set(interp_tokens)
    
    # Process each cross-reference
    ref_analyses = []
    
    for ref in cross_references:
        # Get verse details
        verse_context = extract_passage_context(bible_dict, ref, context_size=2)
        
        if not verse_context["verse_text"]:
            continue
            
        verse_text = verse_context["verse_text"]
        
        # Get verse terms
        verse_tokens = [w.lower() for w in word_tokenize(verse_text) 
                       if w.lower() not in stop_words and w.isalnum()]
        verse_terms = set(verse_tokens)
        
        # Calculate overlap and opposition
        term_overlap = interp_terms.intersection(verse_terms)
        
        # Check for opposing terms/concepts (simplistic approach)
        opposing_pairs = [
            ({"equal", "same", "identical"}, {"different", "distinct", "separate"}),
            ({"divine", "god", "deity"}, {"human", "man", "created"}),
            ({"literal", "actual", "real"}, {"figurative", "symbolic", "metaphor"})
        ]
        
        opposition_score = 0
        opposing_terms = []
        
        for group1, group2 in opposing_pairs:
            if any(term in interp_terms for term in group1) and any(term in verse_terms for term in group2):
                opposition_score += 1
                opposing_terms.extend([term for term in verse_terms if term in group2])
            
            if any(term in interp_terms for term in group2) and any(term in verse_terms for term in group1):
                opposition_score += 1
                opposing_terms.extend([term for term in verse_terms if term in group1])
        
        # Calculate a consistency score (higher is more consistent)
        overlap_score = len(term_overlap) / len(interp_terms) if interp_terms else 0
        consistency = overlap_score * (1 - (opposition_score * 0.2))  # Reduce for opposition
        
        ref_analysis = {
            "reference": ref,
            "verse_text": verse_text,
            "consistency": consistency,
            "term_overlap": list(term_overlap),
            "opposing_terms": opposing_terms,
            "opposition_score": opposition_score
        }
        
        ref_analyses.append(ref_analysis)
    
    # Calculate overall consistency
    if ref_analyses:
        overall_consistency = sum(r["consistency"] for r in ref_analyses) / len(ref_analyses)
        result["consistency_score"] = overall_consistency
    
    # Sort references by consistency
    ref_analyses.sort(key=lambda x: x["consistency"], reverse=True)
    
    # Identify supportive and contradictory references
    supportive_refs = [r for r in ref_analyses if r["consistency"] > 0.5]
    contradictory_refs = [r for r in ref_analyses if r["opposition_score"] > 0]
    
    # Add to result
    result["cross_references"] = ref_analyses
    result["supportive_references"] = supportive_refs
    result["contradictory_references"] = contradictory_refs
    
    # Extract common themes across consistent references
    if supportive_refs:
        all_terms = []
        for ref in supportive_refs:
            all_terms.extend(ref["term_overlap"])
        
        term_counter = Counter(all_terms)
        result["thematic_alignment"] = [term for term, count in term_counter.most_common(5) if count > 1]
    
    return result

def check_genre_appropriate_interpretation(bible_dict: Dict[str, Any],
                                         reference: str,
                                         interpretation: str) -> Dict[str, Any]:
    """
    Check if an interpretation is appropriate for the literary genre of the passage.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        reference: Verse reference to analyze
        interpretation: The interpretation to evaluate
        
    Returns:
        Dictionary with genre analysis results
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Check if an interpretation respects the parable genre
        >>> analysis = check_genre_appropriate_interpretation(
        ...     bible,
        ...     "Luke 15:11",
        ...     "The father represents God, and the prodigal son represents sinners"
        ... )
    """
    # Define genre characteristics
    genres = {
        "narrative": {
            "markers": ["it came to pass", "in those days", "after this", "then"],
            "books": ["Genesis", "Exodus", "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel", 
                    "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah", 
                    "Esther", "Daniel", "Jonah", "Matthew", "Mark", "Luke", "John", "Acts"],
            "interpretation_approach": ["historical context", "character development", 
                                      "plot significance", "narrative arc"]
        },
        "poetry": {
            "markers": ["blessed is", "praise the lord", "o lord", "my soul"],
            "books": ["Job", "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon", "Lamentations"],
            "interpretation_approach": ["figurative language", "parallelism", "emotional expression", 
                                      "imagery", "metaphor"]
        },
        "prophecy": {
            "markers": ["thus says the lord", "the word of the lord came", "behold", "in that day", 
                       "declares the lord"],
            "books": ["Isaiah", "Jeremiah", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos", "Obadiah", 
                     "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi", 
                     "Revelation"],
            "interpretation_approach": ["historical fulfillment", "symbolic imagery", 
                                      "covenant language", "future orientation"]
        },
        "epistle": {
            "markers": ["grace and peace", "i write to you", "beloved", "brothers", "greet one another"],
            "books": ["Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians", 
                     "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians", 
                     "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews", "James", 
                     "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude"],
            "interpretation_approach": ["theological teaching", "practical application", 
                                      "logical argumentation", "cultural context"]
        },
        "parable": {
            "markers": ["parable", "kingdom of heaven is like", "there was a", "a certain man"],
            "books": [],  # Not book-specific but passage-specific
            "interpretation_approach": ["main point", "symbolic elements", "cultural background", 
                                      "unexpected twist"]
        },
        "apocalyptic": {
            "markers": ["vision", "saw", "behold", "like", "as it were"],
            "books": ["Daniel", "Ezekiel", "Zechariah", "Revelation"],
            "interpretation_approach": ["symbolic imagery", "cosmic conflict", 
                                      "divine sovereignty", "coded language"]
        },
        "law": {
            "markers": ["shall", "command", "thus shall you do", "keep"],
            "books": ["Exodus", "Leviticus", "Numbers", "Deuteronomy"],
            "interpretation_approach": ["covenant context", "ethical principles", 
                                      "historical application", "underlying purpose"]
        },
        "wisdom": {
            "markers": ["wisdom", "whoever", "better is", "fear of the lord"],
            "books": ["Proverbs", "Ecclesiastes", "Job", "James"],
            "interpretation_approach": ["general principle", "practical application", 
                                      "exception awareness", "life orientation"]
        }
    }
    
    # Initialize results
    result = {
        "reference": reference,
        "interpretation": interpretation,
        "detected_genre": None,
        "genre_markers_found": [],
        "genre_appropriate": None,
        "appropriate_approaches": [],
        "inappropriate_aspects": []
    }
    
    # Parse reference
    parts = reference.split()
    if len(parts) < 2:
        result["error"] = "Invalid reference format"
        return result
    
    # Handle multi-word book names (e.g., "1 Kings")
    book = " ".join(parts[:-1])
    chapter_verse = parts[-1].split(":")
    
    if len(chapter_verse) != 2:
        result["error"] = "Invalid reference format"
        return result
    
    try:
        chapter = int(chapter_verse[0])
        verse = int(chapter_verse[1])
    except ValueError:
        result["error"] = "Invalid reference format"
        return result
    
    # Check if book and chapter exist
    if book not in bible_dict or chapter not in bible_dict[book]:
        result["error"] = "Reference not found"
        return result
    
    # Get the verse and surrounding context
    context = extract_passage_context(bible_dict, reference, context_size=5)
    
    if not context["verse_text"]:
        result["error"] = "Verse not found"
        return result
    
    verse_text = context["verse_text"]
    context_text = verse_text + " " + \
                 " ".join([v["text"] for v in context["preceding_context"]]) + " " + \
                 " ".join([v["text"] for v in context["following_context"]])
    
    # Detect genre
    # 1. First check by book
    detected_genres = []
    for genre, properties in genres.items():
        if book in properties["books"]:
            detected_genres.append(genre)
    
    # 2. Then check for genre markers
    genre_markers = {}
    for genre, properties in genres.items():
        markers_found = []
        for marker in properties["markers"]:
            if marker.lower() in context_text.lower():
                markers_found.append(marker)
        
        if markers_found:
            genre_markers[genre] = markers_found
    
    # Determine most likely genre
    if genre_markers.get("parable"):
        # Parable markers override other genres if present
        primary_genre = "parable"
    elif genre_markers.get("apocalyptic") and book in genres["apocalyptic"]["books"]:
        # Apocalyptic needs both markers and right book
        primary_genre = "apocalyptic"
    elif len(detected_genres) == 1:
        # Single genre detected by book
        primary_genre = detected_genres[0]
    elif len(detected_genres) > 1:
        # Multiple genres possible, use markers to decide
        max_markers = 0
        max_genre = detected_genres[0]
        
        for genre in detected_genres:
            num_markers = len(genre_markers.get(genre, []))
            if num_markers > max_markers:
                max_markers = num_markers
                max_genre = genre
        
        primary_genre = max_genre
    elif genre_markers:
        # No genre detected by book, use markers
        max_markers = 0
        max_genre = list(genre_markers.keys())[0]
        
        for genre, markers in genre_markers.items():
            if len(markers) > max_markers:
                max_markers = len(markers)
                max_genre = genre
        
        primary_genre = max_genre
    else:
        # Default to narrative if no clear indicators
        primary_genre = "narrative"
    
    # Store detected genre info
    result["detected_genre"] = primary_genre
    result["genre_markers_found"] = genre_markers.get(primary_genre, [])
    result["all_detected_genres"] = detected_genres
    result["all_genre_markers"] = genre_markers
    
    # Check if interpretation is appropriate for genre
    appropriate_approaches = genres[primary_genre]["interpretation_approach"]
    
    # Tokenize interpretation and look for approach indicators
    tokens = word_tokenize(interpretation.lower())
    approach_found = False
    
    for approach in appropriate_approaches:
        approach_tokens = word_tokenize(approach.lower())
        if any(token in tokens for token in approach_tokens):
            approach_found = True
            break
    
    # Check for inappropriate approaches
    inappropriate_approaches = []
    
    if primary_genre == "parable" and "historical" in interpretation.lower():
        inappropriate_approaches.append("treating parable as historical narrative")
    
    if primary_genre == "apocalyptic" and "literal" in interpretation.lower():
        inappropriate_approaches.append("overly literal reading of apocalyptic imagery")
    
    if primary_genre == "poetry" and "doctrine" in interpretation.lower():
        inappropriate_approaches.append("deriving precise doctrine from poetic language")
    
    if primary_genre == "wisdom" and "universal" in interpretation.lower():
        inappropriate_approaches.append("treating wisdom literature as universal promises")
    
    # Determine overall appropriateness
    if inappropriate_approaches:
        result["genre_appropriate"] = False
    elif approach_found:
        result["genre_appropriate"] = True
    else:
        # Neutral result if no clear indicators
        result["genre_appropriate"] = None
    
    result["appropriate_approaches"] = appropriate_approaches
    result["inappropriate_aspects"] = inappropriate_approaches
    
    return result
