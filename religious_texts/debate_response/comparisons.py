"""
Interpretive Framework Comparison Module

This module provides functions for comparing different interpretive frameworks
applied to biblical texts, specifically designed to evaluate competing theological
interpretations such as those presented in debates.
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union, Any, Tuple, Set

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize

def compare_interpretations(interpretation1: Dict[str, Any], 
                           interpretation2: Dict[str, Any],
                           passages: List[str],
                           bible_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Compare two different interpretations of the same biblical passages.
    
    Args:
        interpretation1: Dictionary with details of the first interpretation
        interpretation2: Dictionary with details of the second interpretation
        passages: List of biblical passages to compare interpretations on
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        
    Returns:
        DataFrame with comparison results
        
    Example:
        >>> # Compare Trinitarian vs Unitarian interpretations
        >>> trinitarian = {
        ...     "name": "Trinitarian Reading",
        ...     "description": "Jesus is fully divine, equal with God",
        ...     "key_terms": ["deity", "worship", "divine", "God"],
        ...     "supporting_passages": ["John 1:1", "John 20:28", "Hebrews 1:8"]
        ... }
        >>> unitarian = {
        ...     "name": "Unitarian Reading",
        ...     "description": "Jesus is God's agent but not equal to God",
        ...     "key_terms": ["agent", "subordinate", "representative", "sent"],
        ...     "supporting_passages": ["John 14:28", "John 17:3", "1 Cor 8:6"]
        ... }
        >>> results = compare_interpretations(
        ...     trinitarian, unitarian, ["John 1:1-18", "John 10:30"], bible_dict
        ... )
    """
    # Process passages to get the text
    passage_texts = {}
    
    for ref in passages:
        # Handle range references (e.g., "John 1:1-18")
        if '-' in ref.split(':')[-1]:
            parts = ref.split()
            book = ' '.join(parts[:-1])
            chapter_verse = parts[-1].split(':')
            
            if len(chapter_verse) == 2:
                chapter = int(chapter_verse[0])
                verse_range = chapter_verse[1].split('-')
                
                if len(verse_range) == 2:
                    start_verse = int(verse_range[0])
                    end_verse = int(verse_range[1])
                    
                    if book in bible_dict and chapter in bible_dict[book]:
                        # Collect all verses in the range
                        verses_text = []
                        for verse in range(start_verse, end_verse + 1):
                            if verse in bible_dict[book][chapter]:
                                verses_text.append(bible_dict[book][chapter][verse])
                        
                        if verses_text:
                            passage_texts[ref] = ' '.join(verses_text)
        else:
            # Handle single verse references
            parts = ref.split()
            if len(parts) >= 2:
                book = ' '.join(parts[:-1])
                chapter_verse = parts[-1].split(':')
                
                if len(chapter_verse) == 2:
                    chapter = int(chapter_verse[0])
                    verse = int(chapter_verse[1])
                    
                    if book in bible_dict and chapter in bible_dict[book] and verse in bible_dict[book][chapter]:
                        passage_texts[ref] = bible_dict[book][chapter][verse]
    
    # Initialize results list
    results = []
    
    # Compare interpretations for each passage
    for passage, text in passage_texts.items():
        # Calculate scores for each interpretation
        interp1_score = 0
        interp1_terms = []
        interp2_score = 0
        interp2_terms = []
        
        # Check for key terms from each interpretation
        for term in interpretation1.get("key_terms", []):
            pattern = r'\b' + re.escape(term.lower()) + r'\b'
            matches = re.findall(pattern, text.lower())
            interp1_score += len(matches)
            interp1_terms.extend([term] * len(matches))
        
        for term in interpretation2.get("key_terms", []):
            pattern = r'\b' + re.escape(term.lower()) + r'\b'
            matches = re.findall(pattern, text.lower())
            interp2_score += len(matches)
            interp2_terms.extend([term] * len(matches))
        
        # Check for supporting passages
        interp1_support = 0
        for support_ref in interpretation1.get("supporting_passages", []):
            if passage == support_ref or (passage in passage_texts and support_ref in passage_texts):
                interp1_support += 1
        
        interp2_support = 0
        for support_ref in interpretation2.get("supporting_passages", []):
            if passage == support_ref or (passage in passage_texts and support_ref in passage_texts):
                interp2_support += 1
        
        # Determine which interpretation has stronger evidence for this passage
        stronger_interp = None
        if interp1_score > interp2_score:
            stronger_interp = interpretation1["name"]
        elif interp2_score > interp1_score:
            stronger_interp = interpretation2["name"]
        else:
            stronger_interp = "Equal"
        
        # Add to results
        results.append({
            "passage": passage,
            "text": text,
            f"{interpretation1['name']}_score": interp1_score,
            f"{interpretation1['name']}_terms": ", ".join(interp1_terms),
            f"{interpretation1['name']}_support": interp1_support,
            f"{interpretation2['name']}_score": interp2_score,
            f"{interpretation2['name']}_terms": ", ".join(interp2_terms),
            f"{interpretation2['name']}_support": interp2_support,
            "stronger_interpretation": stronger_interp
        })
    
    if results:
        return pd.DataFrame(results)
    else:
        # Return empty DataFrame with expected columns if no results
        columns = [
            "passage", "text", 
            f"{interpretation1['name']}_score", f"{interpretation1['name']}_terms", f"{interpretation1['name']}_support",
            f"{interpretation2['name']}_score", f"{interpretation2['name']}_terms", f"{interpretation2['name']}_support",
            "stronger_interpretation"
        ]
        return pd.DataFrame(columns=columns)

def interpretation_consistency(bible_dict: Dict[str, Any],
                              interpretation: Dict[str, Any],
                              passages: List[str]) -> Dict[str, Any]:
    """
    Evaluate the internal consistency of an interpretation across multiple passages.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        interpretation: Dictionary with interpretation details
        passages: List of biblical passages to check for consistency
        
    Returns:
        Dictionary with consistency evaluation results
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Check consistency of Trinitarian interpretation
        >>> trinitarian = {
        ...     "name": "Trinitarian Reading",
        ...     "description": "Jesus is fully divine, equal with God",
        ...     "key_terms": ["deity", "worship", "divine", "God"],
        ...     "supporting_passages": ["John 1:1", "John 20:28", "Hebrews 1:8"]
        ... }
        >>> results = interpretation_consistency(
        ...     bible, trinitarian, ["John 1:1", "John 17:3", "1 Timothy 2:5"]
        ... )
    """
    # Extract interpretation details
    interp_name = interpretation.get("name", "Unnamed Interpretation")
    interp_desc = interpretation.get("description", "")
    key_terms = interpretation.get("key_terms", [])
    supporting_passages = interpretation.get("supporting_passages", [])
    
    # Process passages to get the text
    passage_data = []
    
    for ref in passages:
        # Handle range references (e.g., "John 1:1-18")
        text = None
        
        if '-' in ref.split(':')[-1]:
            parts = ref.split()
            book = ' '.join(parts[:-1])
            chapter_verse = parts[-1].split(':')
            
            if len(chapter_verse) == 2:
                chapter = int(chapter_verse[0])
                verse_range = chapter_verse[1].split('-')
                
                if len(verse_range) == 2:
                    start_verse = int(verse_range[0])
                    end_verse = int(verse_range[1])
                    
                    if book in bible_dict and chapter in bible_dict[book]:
                        # Collect all verses in the range
                        verses_text = []
                        for verse in range(start_verse, end_verse + 1):
                            if verse in bible_dict[book][chapter]:
                                verses_text.append(bible_dict[book][chapter][verse])
                        
                        if verses_text:
                            text = ' '.join(verses_text)
        else:
            # Handle single verse references
            parts = ref.split()
            if len(parts) >= 2:
                book = ' '.join(parts[:-1])
                chapter_verse = parts[-1].split(':')
                
                if len(chapter_verse) == 2:
                    chapter = int(chapter_verse[0])
                    verse = int(chapter_verse[1])
                    
                    if book in bible_dict and chapter in bible_dict[book] and verse in bible_dict[book][chapter]:
                        text = bible_dict[book][chapter][verse]
        
        if text:
            # Calculate term occurrences for this passage
            term_matches = []
            
            for term in key_terms:
                pattern = r'\b' + re.escape(term.lower()) + r'\b'
                matches = re.findall(pattern, text.lower())
                if matches:
                    term_matches.append(term)
            
            # Is this a supporting passage?
            is_supporting = ref in supporting_passages
            
            # Store passage data
            passage_data.append({
                "reference": ref,
                "text": text,
                "matching_terms": term_matches,
                "term_count": len(term_matches),
                "is_supporting": is_supporting
            })
    
    # Calculate consistency metrics
    supporting_score = 0
    supporting_count = 0
    non_supporting_score = 0
    non_supporting_count = 0
    
    for data in passage_data:
        if data["is_supporting"]:
            supporting_score += data["term_count"]
            supporting_count += 1
        else:
            non_supporting_score += data["term_count"]
            non_supporting_count += 1
    
    avg_supporting_score = supporting_score / max(1, supporting_count)
    avg_non_supporting_score = non_supporting_score / max(1, non_supporting_count)
    
    # Calculate consistency ratio
    # Higher ratio means more consistent (supporting passages have more key terms)
    consistency_ratio = avg_supporting_score / max(0.1, avg_non_supporting_score)
    
    # Analyze term distribution across passages
    term_distribution = {}
    for term in key_terms:
        term_count = sum(1 for data in passage_data if term in data["matching_terms"])
        term_distribution[term] = term_count / len(passage_data)
    
    # Prepare results
    results = {
        "interpretation_name": interp_name,
        "interpretation_description": interp_desc,
        "passages_analyzed": len(passage_data),
        "supporting_passages_count": supporting_count,
        "non_supporting_passages_count": non_supporting_count,
        "avg_terms_in_supporting": avg_supporting_score,
        "avg_terms_in_non_supporting": avg_non_supporting_score,
        "consistency_ratio": consistency_ratio,
        "is_consistent": consistency_ratio > 1.5,  # Threshold for considering consistent
        "term_distribution": term_distribution,
        "passage_data": passage_data
    }
    
    return results

def alternative_readings(bible_dict: Dict[str, Any],
                         passage: str,
                         interpretations: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Analyze and compare multiple alternative readings of a biblical passage.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        passage: Biblical passage reference to analyze
        interpretations: List of dictionaries with interpretation details
        
    Returns:
        DataFrame with alternative reading analysis
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Analyze John 1:1 alternative readings
        >>> readings = [
        ...     {
        ...         "name": "Orthodox",
        ...         "translation": "the Word was God",
        ...         "key_terms": ["deity", "identity"],
        ...         "theological_framework": "Trinitarian"
        ...     },
        ...     {
        ...         "name": "NWT",
        ...         "translation": "the Word was a god",
        ...         "key_terms": ["qualitative", "subordinate"],
        ...         "theological_framework": "Non-trinitarian"
        ...     }
        ... ]
        >>> results = alternative_readings(bible, "John 1:1", readings)
    """
    # Extract passage text
    passage_text = None
    
    # Handle range references (e.g., "John 1:1-18")
    if '-' in passage.split(':')[-1]:
        parts = passage.split()
        book = ' '.join(parts[:-1])
        chapter_verse = parts[-1].split(':')
        
        if len(chapter_verse) == 2:
            chapter = int(chapter_verse[0])
            verse_range = chapter_verse[1].split('-')
            
            if len(verse_range) == 2:
                start_verse = int(verse_range[0])
                end_verse = int(verse_range[1])
                
                if book in bible_dict and chapter in bible_dict[book]:
                    # Collect all verses in the range
                    verses_text = []
                    for verse in range(start_verse, end_verse + 1):
                        if verse in bible_dict[book][chapter]:
                            verses_text.append(bible_dict[book][chapter][verse])
                    
                    if verses_text:
                        passage_text = ' '.join(verses_text)
    else:
        # Handle single verse references
        parts = passage.split()
        if len(parts) >= 2:
            book = ' '.join(parts[:-1])
            chapter_verse = parts[-1].split(':')
            
            if len(chapter_verse) == 2:
                chapter = int(chapter_verse[0])
                verse = int(chapter_verse[1])
                
                if book in bible_dict and chapter in bible_dict[book] and verse in bible_dict[book][chapter]:
                    passage_text = bible_dict[book][chapter][verse]
    
    if not passage_text:
        # Empty DataFrame with expected columns
        columns = ["reading_name", "translation", "theological_framework", 
                   "textual_basis", "interpretive_issues", "comparative_score"]
        return pd.DataFrame(columns=columns)
    
    # Get surrounding context for evaluating readings
    context_text = ""
    parts = passage.split()
    if len(parts) >= 2:
        book = ' '.join(parts[:-1])
        chapter_verse = parts[-1].split(':')
        
        if len(chapter_verse) == 2:
            chapter = int(chapter_verse[0])
            verse = int(chapter_verse[1])
            
            # Get 2 verses before and after
            for v in range(max(1, verse - 2), verse + 3):
                if v != verse and book in bible_dict and chapter in bible_dict[book] and v in bible_dict[book][chapter]:
                    context_text += bible_dict[book][chapter][v] + " "
    
    # Analyze each reading
    reading_data = []
    
    for i, reading in enumerate(interpretations):
        reading_name = reading.get("name", f"Reading {i+1}")
        translation = reading.get("translation", "")
        framework = reading.get("theological_framework", "")
        key_terms = reading.get("key_terms", [])
        
        # Calculate textual fit
        textual_fit = 0
        matching_terms = []
        
        for term in key_terms:
            pattern = r'\b' + re.escape(term.lower()) + r'\b'
            # Check both passage and context
            passage_matches = re.findall(pattern, passage_text.lower())
            context_matches = re.findall(pattern, context_text.lower())
            
            # Count matches (passage matches worth more)
            term_score = len(passage_matches) + 0.5 * len(context_matches)
            textual_fit += term_score
            
            if passage_matches or context_matches:
                matching_terms.append(term)
        
        # Identify potential interpretive issues
        issues = []
        
        # Check if translation matches passage
        if translation and translation.lower() not in passage_text.lower():
            issues.append("Translation differs from standard text")
        
        # Check for theological coherence with context
        if context_text and key_terms:
            context_matches = sum(1 for term in key_terms if term.lower() in context_text.lower())
            if context_matches == 0:
                issues.append("Key terms absent from surrounding context")
        
        reading_data.append({
            "reading_name": reading_name,
            "translation": translation,
            "theological_framework": framework,
            "textual_basis": ", ".join(matching_terms),
            "interpretive_issues": ", ".join(issues),
            "comparative_score": textual_fit
        })
    
    # Sort by comparative score
    reading_df = pd.DataFrame(reading_data)
    reading_df = reading_df.sort_values("comparative_score", ascending=False)
    
    return reading_df

def compare_translations(bible_dict: Dict[str, Any],
                        reference: str,
                        translations: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Compare different translations of a biblical passage and analyze differences.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        reference: Biblical passage reference to compare
        translations: Dictionary mapping translation names to text
        
    Returns:
        DataFrame with translation comparison results
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Compare translations of Romans 3:28
        >>> translations = {
        ...     "KJV": {"text": "Therefore we conclude that a man is justified by faith without the deeds of the law."},
        ...     "Luther": {"text": "So halten wir es nun, daß der Mensch gerecht werde ohne des Gesetzes Werke, allein durch den Glauben."}
        ... }
        >>> results = compare_translations(bible, "Romans 3:28", translations)
    """
    # Initialize results list
    results = []
    
    # Extract reference information
    parts = reference.split()
    if len(parts) < 2:
        return pd.DataFrame()
    
    book = ' '.join(parts[:-1])
    chapter_verse = parts[-1].split(':')
    
    if len(chapter_verse) != 2:
        return pd.DataFrame()
    
    # Extract key terms from all translations
    all_terms = []
    
    for trans_name, trans_data in translations.items():
        text = trans_data.get("text", "")
        if text:
            # Simple tokenization (could be improved for non-English)
            tokens = text.lower().split()
            # Remove common words and punctuation
            tokens = [token.strip('.,;:?!()[]"\'') for token in tokens if len(token) > 3]
            all_terms.extend(tokens)
    
    # Count term frequency
    term_counts = Counter(all_terms)
    
    # Get the most common terms (potential key theological terms)
    key_terms = [term for term, count in term_counts.most_common(10) if count > 1]
    
    # Analyze each translation
    for trans_name, trans_data in translations.items():
        text = trans_data.get("text", "")
        if not text:
            continue
        
        # Check for key terms
        term_presence = {}
        for term in key_terms:
            term_presence[term] = term.lower() in text.lower()
        
        # Analyze translation characteristics
        char_count = len(text)
        word_count = len(text.split())
        avg_word_length = sum(len(word) for word in text.split()) / max(1, word_count)
        
        # Check for unique terms in this translation
        tokens = set(text.lower().split())
        unique_terms = [token for token in tokens 
                       if sum(1 for t_name, t_data in translations.items() 
                             if t_name != trans_name and token in t_data.get("text", "").lower()) == 0]
        
        results.append({
            "translation": trans_name,
            "text": text,
            "char_count": char_count,
            "word_count": word_count,
            "avg_word_length": avg_word_length,
            "key_terms_present": sum(term_presence.values()),
            "unique_terms": ", ".join(unique_terms[:5])  # Show up to 5 unique terms
        })
    
    if results:
        df = pd.DataFrame(results)
        
        # Add comparative columns
        avg_char_count = df["char_count"].mean()
        avg_word_count = df["word_count"].mean()
        
        df["char_ratio"] = df["char_count"] / avg_char_count
        df["word_ratio"] = df["word_count"] / avg_word_count
        
        # Sort by word count (shorter to longer)
        df = df.sort_values("word_count")
        
        return df
    else:
        return pd.DataFrame()

def translation_bias_analysis(bible_dict: Dict[str, Any],
                             reference: str,
                             translations: Dict[str, Dict[str, Any]],
                             theological_terms: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Analyze potential theological bias in different translations of a passage.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        reference: Biblical passage reference to analyze
        translations: Dictionary mapping translation names to text
        theological_terms: Dictionary mapping theological positions to related terms
        
    Returns:
        Dictionary with bias analysis results
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Analyze Romans 9:5 translations for Christological bias
        >>> translations = {
        ...     "KJV": {"text": "...Christ came, who is over all, God blessed for ever."},
        ...     "NEB": {"text": "...Christ came. May God, supreme above all, be blessed for ever!"}
        ... }
        >>> theological_terms = {
        ...     "High Christology": ["God", "deity", "divine", "worship"],
        ...     "Low Christology": ["agent", "subordinate", "human", "separate"]
        ... }
        >>> results = translation_bias_analysis(
        ...     bible, "Romans 9:5", translations, theological_terms
        ... )
    """
    # Initialize results
    results = {
        "reference": reference,
        "translation_analysis": [],
        "theological_metrics": {},
        "bias_assessment": {}
    }
    
    # Extract reference information
    parts = reference.split()
    if len(parts) < 2:
        return results
    
    book = ' '.join(parts[:-1])
    chapter_verse = parts[-1].split(':')
    
    if len(chapter_verse) != 2:
        return results
    
    # Analyze each translation
    translation_scores = {}
    
    for trans_name, trans_data in translations.items():
        text = trans_data.get("text", "")
        if not text:
            continue
        
        # Calculate theological term occurrences
        theology_scores = {}
        matching_terms = {}
        
        for position, terms in theological_terms.items():
            position_score = 0
            position_matches = []
            
            for term in terms:
                pattern = r'\b' + re.escape(term.lower()) + r'\b'
                matches = re.findall(pattern, text.lower())
                
                if matches:
                    position_score += len(matches)
                    position_matches.extend([term] * len(matches))
            
            theology_scores[position] = position_score
            matching_terms[position] = list(set(position_matches))
        
        # Determine primary theological lean
        max_score = 0
        primary_position = None
        
        for position, score in theology_scores.items():
            if score > max_score:
                max_score = score
                primary_position = position
        
        # Calculate bias strength
        total_score = sum(theology_scores.values())
        bias_strength = max_score / total_score if total_score > 0 else 0
        
        # Only consider biased if there's a clear lean
        is_biased = bias_strength > 0.7 and max_score > 1
        
        # Store analysis for this translation
        translation_scores[trans_name] = {
            "text": text,
            "theology_scores": theology_scores,
            "matching_terms": matching_terms,
            "primary_position": primary_position,
            "bias_strength": bias_strength,
            "is_biased": is_biased
        }
        
        results["translation_analysis"].append({
            "translation": trans_name,
            "text": text,
            "primary_position": primary_position,
            "bias_strength": bias_strength,
            "is_biased": is_biased,
            "matching_terms": matching_terms
        })
    
    # Calculate overall metrics
    position_counts = {}
    for theological_position in theological_terms.keys():
        count = sum(1 for trans_name, data in translation_scores.items() 
                   if data["primary_position"] == theological_position and data["is_biased"])
        position_counts[theological_position] = count
    
    results["theological_metrics"]["position_counts"] = position_counts
    
    # Assess bias across translations
    total_translations = len(translations)
    biased_translations = sum(1 for data in translation_scores.values() if data["is_biased"])
    bias_percentage = (biased_translations / total_translations) * 100 if total_translations > 0 else 0
    
    # Determine if there's a dominant position
    max_position_count = max(position_counts.values()) if position_counts else 0
    dominant_position = None
    
    if max_position_count > 0:
        dominant_positions = [pos for pos, count in position_counts.items() if count == max_position_count]
        dominant_position = dominant_positions[0] if len(dominant_positions) == 1 else None
    
    results["bias_assessment"] = {
        "total_translations": total_translations,
        "biased_translations": biased_translations,
        "bias_percentage": bias_percentage,
        "dominant_position": dominant_position,
        "has_clear_bias": dominant_position is not None and bias_percentage > 50
    }
    
    return results

def create_framework_profile(framework_name: str, 
                           key_terms: Dict[str, List[str]],
                           hermeneutic_principles: List[str],
                           textual_emphasis: List[str] = None) -> Dict[str, Any]:
    """
    Create a profile for an interpretive framework.
    
    Args:
        framework_name: Name of the interpretive framework
        key_terms: Dictionary mapping concepts to lists of related terms
        hermeneutic_principles: List of hermeneutic principles of this framework
        textual_emphasis: Optional list of texts emphasized in this framework
        
    Returns:
        Dictionary representing the framework profile
        
    Example:
        >>> # Create profile for Trinitarian framework
        >>> trinitarian = create_framework_profile(
        ...     "Trinitarian",
        ...     {
        ...         "divine_nature": ["God", "divine", "deity", "godhead"],
        ...         "trinity": ["trinity", "triune", "three persons", "father son spirit"],
        ...         "jesus_deity": ["Jesus is God", "deity of Christ", "divine Son"]
        ...     },
        ...     ["Read NT in light of later church councils", 
        ...      "Look for implicit trinitarian patterns",
        ...      "Interpret Christ's divinity statements maximally"],
        ...     ["John 1:1-18", "Philippians 2:5-11", "Colossians 1:15-20"]
        ... )
    """
    return {
        "name": framework_name,
        "key_terms": key_terms,
        "hermeneutic_principles": hermeneutic_principles,
        "textual_emphasis": textual_emphasis or [],
        "associated_traditions": [],
        "scholarly_proponents": [],
        "created_date": None,
        "description": ""
    }

def compare_frameworks(bible_dict: Dict[str, Any], 
                     frameworks: List[Dict[str, Any]],
                     references: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compare different interpretive frameworks applied to the same biblical passages.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        frameworks: List of framework profiles created with create_framework_profile
        references: Optional list of specific verse references to analyze
        
    Returns:
        DataFrame with comparison results
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Compare Trinitarian vs Unitarian reading of key passages
        >>> trinitarian = create_framework_profile(...)
        >>> unitarian = create_framework_profile(...)
        >>> comparison = compare_frameworks(
        ...     bible, 
        ...     [trinitarian, unitarian],
        ...     ["John 1:1", "John 10:30", "John 17:5"]
        ... )
    """
    from religious_texts.text_analysis.frequency import word_frequency
    
    # Extract a list of all key terms from all frameworks
    all_terms = []
    framework_term_maps = {}
    
    for framework in frameworks:
        framework_terms = []
        for concept, terms in framework["key_terms"].items():
            all_terms.extend(terms)
            framework_terms.extend(terms)
        framework_term_maps[framework["name"]] = framework_terms
    
    # Remove duplicates
    all_terms = list(set(all_terms))
    
    # Process each reference or analyze whole Bible
    results = []
    
    if references:
        # Parse references into book, chapter, verse
        parsed_refs = []
        for ref in references:
            parts = ref.split()
            if len(parts) >= 2:
                # Handle multi-word book names (e.g., "1 Kings")
                book = " ".join(parts[:-1])
                chapter_verse = parts[-1].split(":")
                if len(chapter_verse) == 2:
                    chapter = int(chapter_verse[0])
                    verse = int(chapter_verse[1])
                    parsed_refs.append((book, chapter, verse))
        
        # Analyze specific references
        for book, chapter, verse in parsed_refs:
            if book in bible_dict and chapter in bible_dict[book] and verse in bible_dict[book][chapter]:
                verse_text = bible_dict[book][chapter][verse]
                reference = f"{book} {chapter}:{verse}"
                
                # Count terms for each framework
                framework_scores = {}
                for framework in frameworks:
                    score = 0
                    matched_terms = []
                    
                    # Check for each term
                    for concept, terms in framework["key_terms"].items():
                        for term in terms:
                            # Simple case-insensitive term matching
                            pattern = r'\b' + re.escape(term.lower()) + r'\b'
                            matches = re.findall(pattern, verse_text.lower())
                            if matches:
                                score += len(matches)
                                matched_terms.extend([term] * len(matches))
                    
                    framework_scores[framework["name"]] = {
                        "score": score,
                        "matched_terms": matched_terms
                    }
                
                # Add to results
                results.append({
                    "reference": reference,
                    "text": verse_text,
                    "frameworks": framework_scores
                })
    else:
        # Analyze all verses
        for book in bible_dict:
            for chapter in bible_dict[book]:
                for verse in bible_dict[book][chapter]:
                    verse_text = bible_dict[book][chapter][verse]
                    reference = f"{book} {chapter}:{verse}"
                    
                    # Count terms for each framework
                    framework_scores = {}
                    for framework in frameworks:
                        score = 0
                        matched_terms = []
                        
                        # Check for each term
                        for concept, terms in framework["key_terms"].items():
                            for term in terms:
                                # Simple case-insensitive term matching
                                pattern = r'\b' + re.escape(term.lower()) + r'\b'
                                matches = re.findall(pattern, verse_text.lower())
                                if matches:
                                    score += len(matches)
                                    matched_terms.extend([term] * len(matches))
                        
                        framework_scores[framework["name"]] = {
                            "score": score,
                            "matched_terms": matched_terms
                        }
                    
                    # Only include results with at least one match
                    if any(fs["score"] > 0 for fs in framework_scores.values()):
                        results.append({
                            "reference": reference,
                            "text": verse_text,
                            "frameworks": framework_scores
                        })
    
    # Convert results to a more tabular format for the DataFrame
    tabular_results = []
    
    for result in results:
        row = {
            "reference": result["reference"],
            "text": result["text"]
        }
        
        # Add framework scores and matched terms
        for framework_name, framework_data in result["frameworks"].items():
            row[f"{framework_name}_score"] = framework_data["score"]
            row[f"{framework_name}_terms"] = ", ".join(framework_data["matched_terms"])
        
        tabular_results.append(row)
    
    if tabular_results:
        return pd.DataFrame(tabular_results)
    else:
        # Return empty DataFrame with expected columns
        columns = ["reference", "text"]
        for framework in frameworks:
            columns.extend([f"{framework['name']}_score", f"{framework['name']}_terms"])
        return pd.DataFrame(columns=columns)

def evaluate_interpretive_fit(bible_dict: Dict[str, Any],
                            passages: List[str],
                            interpretations: List[Dict[str, Any]],
                            criteria: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Evaluate how well different interpretations fit with the given passages.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        passages: List of verse references to analyze
        interpretations: List of dictionaries with interpretation details
        criteria: Optional list of evaluation criteria
        
    Returns:
        DataFrame with evaluation results
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Evaluate competing interpretations of John 10:30
        >>> interpretations = [
        ...     {
        ...         "name": "Ontological Unity",
        ...         "description": "Jesus claims equality with God in nature/essence",
        ...         "supporting_passages": ["John 1:1", "John 8:58", "John 20:28"],
        ...         "key_terms": ["am", "deity", "worship"]
        ...     },
        ...     {
        ...         "name": "Functional Unity",
        ...         "description": "Jesus claims unity of purpose/will with God",
        ...         "supporting_passages": ["John 17:11", "John 17:21-23"],
        ...         "key_terms": ["sent", "will", "purpose"]
        ...     }
        ... ]
        >>> evaluation = evaluate_interpretive_fit(
        ...     bible,
        ...     ["John 10:30"],
        ...     interpretations,
        ...     ["contextual_consistency", "linguistic_evidence", "canonical_harmony"]
        ... )
    """
    # Default criteria if none provided
    if criteria is None:
        criteria = [
            "contextual_consistency",  # Consistent with surrounding context
            "linguistic_evidence",     # Supported by language/terms used
            "canonical_harmony",       # Consistent with other parts of canon
            "historical_plausibility", # Plausible in historical context
            "explanatory_power"        # Explains the passage well
        ]
    
    # Process passages
    passage_texts = {}
    for ref in passages:
        parts = ref.split()
        if len(parts) >= 2:
            # Handle multi-word book names (e.g., "1 Kings")
            book = " ".join(parts[:-1])
            chapter_verse = parts[-1].split(":")
            if len(chapter_verse) == 2:
                chapter = int(chapter_verse[0])
                verse = int(chapter_verse[1])
                
                if book in bible_dict and chapter in bible_dict[book] and verse in bible_dict[book][chapter]:
                    verse_text = bible_dict[book][chapter][verse]
                    passage_texts[ref] = verse_text
    
    # Prepare results structure
    results = []
    
    # For each passage and each interpretation, evaluate against criteria
    for passage, text in passage_texts.items():
        for interp in interpretations:
            # Calculate scores
            scores = {}
            
            # 1. Contextual consistency - check surrounding verses
            context_score = 0
            parts = passage.split()
            if len(parts) >= 2:
                # Get surrounding context (3 verses before and after)
                book = " ".join(parts[:-1])
                chapter_verse = parts[-1].split(":")
                if len(chapter_verse) == 2:
                    chapter = int(chapter_verse[0])
                    verse = int(chapter_verse[1])
                    
                    context_verses = []
                    for v in range(max(1, verse-3), verse+4):
                        if v != verse and v in bible_dict[book][chapter]:
                            context_verses.append(bible_dict[book][chapter][v])
                    
                    if context_verses:
                        context_text = " ".join(context_verses)
                        # Check for key terms in context
                        for term in interp.get("key_terms", []):
                            pattern = r'\b' + re.escape(term.lower()) + r'\b'
                            matches = re.findall(pattern, context_text.lower())
                            context_score += len(matches)
            
            scores["contextual_consistency"] = min(10, context_score)
            
            # 2. Linguistic evidence - check for key terms in the passage
            ling_score = 0
            for term in interp.get("key_terms", []):
                pattern = r'\b' + re.escape(term.lower()) + r'\b'
                matches = re.findall(pattern, text.lower())
                ling_score += len(matches) * 2
            
            scores["linguistic_evidence"] = min(10, ling_score)
            
            # 3. Canonical harmony - check supporting passages
            canon_score = 0
            for ref in interp.get("supporting_passages", []):
                if ref in passage_texts:
                    # Supporting passage is one of our analyzed passages
                    support_text = passage_texts[ref]
                else:
                    # Extract supporting passage text
                    parts = ref.split()
                    if len(parts) >= 2:
                        book = " ".join(parts[:-1])
                        chapter_verse = parts[-1].split(":")
                        if len(chapter_verse) == 2:
                            chapter = int(chapter_verse[0])
                            verse = int(chapter_verse[1])
                            
                            if book in bible_dict and chapter in bible_dict[book] and verse in bible_dict[book][chapter]:
                                support_text = bible_dict[book][chapter][verse]
                            else:
                                continue
                        else:
                            continue
                    else:
                        continue
                
                # Check for thematic consistency
                for term in interp.get("key_terms", []):
                    pattern = r'\b' + re.escape(term.lower()) + r'\b'
                    matches = re.findall(pattern, support_text.lower())
                    canon_score += len(matches)
            
            scores["canonical_harmony"] = min(10, canon_score / 2)
            
            # 4 & 5. Historical plausibility and explanatory power
            # These are more subjective and would require more sophisticated analysis
            # For now, assign default middle values
            scores["historical_plausibility"] = 5
            scores["explanatory_power"] = 5
            
            # Calculate total score
            total_score = sum(scores.values()) / len(scores)
            
            # Add to results
            results.append({
                "passage": passage,
                "text": text,
                "interpretation": interp["name"],
                "description": interp.get("description", ""),
                "total_score": total_score,
                **scores
            })
    
    if results:
        return pd.DataFrame(results)
    else:
        # Return empty DataFrame with expected columns
        columns = ["passage", "text", "interpretation", "description", "total_score"] + criteria
        return pd.DataFrame(columns=columns)

def compare_term_usage_across_frameworks(bible_dict: Dict[str, Any],
                                      term: str,
                                      frameworks: List[Dict[str, Any]],
                                      books: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compare how a term is used differently across interpretive frameworks.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        term: Term to analyze usage differences
        frameworks: List of framework profiles created with create_framework_profile
        books: Optional list of specific books to analyze
        
    Returns:
        DataFrame with comparison results
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Compare how "son of god" is interpreted in different frameworks
        >>> christological = create_framework_profile(...)
        >>> adoptionist = create_framework_profile(...)
        >>> comparison = compare_term_usage_across_frameworks(
        ...     bible,
        ...     "son of god",
        ...     [christological, adoptionist],
        ...     ["Matthew", "Mark", "Luke", "John"]
        ... )
    """
    from religious_texts.text_analysis.cooccurrence import word_cooccurrence
    
    # Find all verses with the term
    term_verses = []
    
    # Determine which books to include
    if books:
        books_to_check = {book: bible_dict[book] for book in books if book in bible_dict}
    else:
        books_to_check = bible_dict
    
    # Find verses containing the term
    pattern = r'\b' + re.escape(term.lower()) + r'\b'
    
    for book, chapters in books_to_check.items():
        for chapter, verses in chapters.items():
            for verse_num, verse_text in verses.items():
                if re.search(pattern, verse_text.lower()):
                    term_verses.append({
                        "book": book,
                        "chapter": chapter,
                        "verse": verse_num,
                        "text": verse_text,
                        "reference": f"{book} {chapter}:{verse_num}"
                    })
    
    if not term_verses:
        return pd.DataFrame()
    
    # Analyze co-occurrence with framework terms
    results = []
    
    for verse_data in term_verses:
        verse_text = verse_data["text"]
        tokenized = word_tokenize(verse_text.lower())
        
        # Find all term positions
        term_positions = []
        term_tokens = term.lower().split()
        
        if len(term_tokens) == 1:
            # Single word term
            for i, token in enumerate(tokenized):
                if token == term_tokens[0]:
                    term_positions.append(i)
        else:
            # Multi-word term
            for i in range(len(tokenized) - len(term_tokens) + 1):
                if tokenized[i:i+len(term_tokens)] == term_tokens:
                    term_positions.append(i)
        
        # Analyze context for each framework
        framework_contexts = {}
        
        for framework in frameworks:
            # Find co-occurrences with framework terms
            cooccurrences = []
            
            for concept, concept_terms in framework["key_terms"].items():
                for concept_term in concept_terms:
                    term_tokens = concept_term.lower().split()
                    
                    if len(term_tokens) == 1:
                        # Single word term
                        for i, token in enumerate(tokenized):
                            if token == term_tokens[0]:
                                # Calculate distance to nearest occurrence of main term
                                distances = [abs(i - pos) for pos in term_positions]
                                min_distance = min(distances) if distances else float('inf')
                                
                                if min_distance <= 10:  # Consider terms within 10 words
                                    cooccurrences.append({
                                        "term": concept_term,
                                        "concept": concept,
                                        "distance": min_distance
                                    })
                    else:
                        # Multi-word term
                        for i in range(len(tokenized) - len(term_tokens) + 1):
                            if tokenized[i:i+len(term_tokens)] == term_tokens:
                                # Calculate distance to nearest occurrence of main term
                                distances = [abs(i - pos) for pos in term_positions]
                                min_distance = min(distances) if distances else float('inf')
                                
                                if min_distance <= 10:  # Consider terms within 10 words
                                    cooccurrences.append({
                                        "term": concept_term,
                                        "concept": concept,
                                        "distance": min_distance
                                    })
            
            # Calculate a score based on co-occurrences
            score = 0
            matched_terms = []
            
            for cooc in cooccurrences:
                # Terms closer to the main term get higher weights
                weight = 1 - (cooc["distance"] / 20)  # 0.5 to 1.0 based on distance
                score += weight
                matched_terms.append(cooc["term"])
            
            framework_contexts[framework["name"]] = {
                "score": score,
                "matched_terms": matched_terms,
                "cooccurrences": cooccurrences
            }
        
        # Determine which framework has highest score
        max_score = 0
        best_framework = None
        
        for framework_name, context_data in framework_contexts.items():
            if context_data["score"] > max_score:
                max_score = context_data["score"]
                best_framework = framework_name
        
        # Add to results
        result_row = {
            "reference": verse_data["reference"],
            "text": verse_data["text"],
            "best_framework": best_framework,
            "best_score": max_score
        }
        
        # Add framework-specific data
        for framework in frameworks:
            framework_name = framework["name"]
            context_data = framework_contexts.get(framework_name, {"score": 0, "matched_terms": []})
            
            result_row[f"{framework_name}_score"] = context_data["score"]
            result_row[f"{framework_name}_terms"] = ", ".join(context_data["matched_terms"])
        
        results.append(result_row)
    
    if results:
        return pd.DataFrame(results)
    else:
        # Return empty DataFrame with expected columns
        columns = ["reference", "text", "best_framework", "best_score"]
        for framework in frameworks:
            columns.extend([f"{framework['name']}_score", f"{framework['name']}_terms"])
        return pd.DataFrame(columns=columns)

# Define some common interpretive frameworks for convenient use
TRINITARIAN_FRAMEWORK = create_framework_profile(
    "Trinitarian",
    {
        "divine_nature": ["God", "divine", "deity", "godhead"],
        "trinity": ["trinity", "triune", "three persons", "father son spirit"],
        "jesus_deity": ["Son of God", "deity of Christ", "divine Son", "worship Jesus"]
    },
    [
        "Read NT in light of later church councils",
        "Look for implicit trinitarian patterns",
        "Interpret Christ's divinity statements maximally"
    ],
    ["John 1:1-18", "Philippians 2:5-11", "Colossians 1:15-20"]
)

UNITARIAN_FRAMEWORK = create_framework_profile(
    "Unitarian",
    {
        "monotheism": ["one God", "only God", "only true God", "one true God"],
        "agency": ["agent", "representative", "sent by God", "subordinate"],
        "functional": ["unity with God", "one purpose", "name of God"]
    },
    [
        "Interpret NT in light of Jewish monotheism",
        "Apply agency concepts from Jewish literature",
        "Interpret Christ's divinity statements functionally"
    ],
    ["John 17:3", "John 10:34-36", "1 Corinthians 8:6"]
)

ADOPTIONIST_FRAMEWORK = create_framework_profile(
    "Adoptionist",
    {
        "adoption": ["adopted", "appointed", "became", "exalted"],
        "humanity": ["human", "man", "prophet", "servant"],
        "empowerment": ["spirit descended", "anointed", "power", "authority given"]
    },
    [
        "Emphasize Jesus's humanity and development",
        "Focus on baptism and resurrection as key moments",
        "Understand divine sonship as a status conferred"
    ],
    ["Mark 1:9-11", "Romans 1:4", "Acts 2:36"]
)