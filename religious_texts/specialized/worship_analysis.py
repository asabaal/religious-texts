"""
Specialized Worship Language Analysis Module

This module provides specialized functions for analyzing worship language in biblical texts,
focusing on Greek terms like proskuneo and latreo mentioned in theological debates.
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union, Any, Tuple, Set

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize

# Define dictionaries of worship-related terms by category
WORSHIP_TERMS = {
    "proskuneo": [
        "worship", "bow down", "prostrate", "reverence", "obeisance", 
        "kneel", "adore", "homage", "venerate"
    ],
    "latreo": [
        "serve", "service", "minister", "ministration", "religious service",
        "religious duty", "ritual service", "sacred service"
    ],
    "sebomai": [
        "revere", "reverence", "venerate", "devotion", "devout", "piety", "religious awe"
    ],
    "doxa": [
        "glory", "glorify", "glorified", "honor", "praise", "exalt", "magnify", "extol"
    ]
}

# Define worship recipients
WORSHIP_RECIPIENTS = {
    "god_terms": [
        "God", "the LORD", "Lord", "Father", "Most High", "Almighty",
        "Creator", "Holy One", "Ancient of Days"
    ],
    "jesus_terms": [
        "Jesus", "Christ", "Son", "Son of God", "Son of Man", "Messiah",
        "Master", "Teacher", "Rabbi", "Lord Jesus"
    ],
    "other_recipients": [
        "angel", "Satan", "devil", "idol", "image", "man", "emperor", "king",
        "beast", "false prophet", "creation"
    ]
}

def extract_worship_instances(bible_dict: Dict[str, Any],
                            worship_category: Optional[str] = None,
                            recipient_category: Optional[str] = None,
                            books: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Extract instances of worship language from biblical texts.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        worship_category: Optional worship term category to filter by
        recipient_category: Optional worship recipient category to filter by
        books: Optional list of books to include
        
    Returns:
        DataFrame with worship instances
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Extract proskuneo instances directed toward Jesus
        >>> worship_df = extract_worship_instances(
        ...     bible,
        ...     worship_category="proskuneo",
        ...     recipient_category="jesus_terms",
        ...     books=["Matthew", "Mark", "Luke", "John"]
        ... )
    """
    # Initialize results
    results = []
    
    # Determine which worship terms to include
    if worship_category:
        if worship_category in WORSHIP_TERMS:
            worship_terms = WORSHIP_TERMS[worship_category]
        else:
            raise ValueError(f"Invalid worship category: {worship_category}. " 
                           f"Choose from {list(WORSHIP_TERMS.keys())}")
    else:
        # Include all terms
        worship_terms = []
        for terms in WORSHIP_TERMS.values():
            worship_terms.extend(terms)
    
    # Determine which recipient terms to include
    if recipient_category:
        if recipient_category in WORSHIP_RECIPIENTS:
            recipient_terms = WORSHIP_RECIPIENTS[recipient_category]
        else:
            raise ValueError(f"Invalid recipient category: {recipient_category}. "
                           f"Choose from {list(WORSHIP_RECIPIENTS.keys())}")
    else:
        # Include all recipients
        recipient_terms = []
        for terms in WORSHIP_RECIPIENTS.values():
            recipient_terms.extend(terms)
    
    # Prepare search patterns
    worship_patterns = [re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE) for term in worship_terms]
    recipient_patterns = [re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE) for term in recipient_terms]
    
    # Determine which books to include
    if books:
        book_subset = {book: bible_dict[book] for book in books if book in bible_dict}
    else:
        book_subset = bible_dict
    
    # Process each verse
    for book_name, chapters in book_subset.items():
        for chapter_num, verses in chapters.items():
            for verse_num, verse_text in verses.items():
                if not verse_text:
                    continue
                
                # Check for worship terms
                worship_matches = []
                for i, pattern in enumerate(worship_patterns):
                    matches = list(pattern.finditer(verse_text))
                    for match in matches:
                        worship_matches.append({
                            "term": match.group(0),
                            "original_term": worship_terms[i],
                            "start": match.start(),
                            "end": match.end(),
                            "worship_type": worship_category if worship_category else "unknown"
                        })
                
                # Skip if no worship terms found
                if not worship_matches:
                    continue
                
                # Check for recipient terms
                recipient_matches = []
                for i, pattern in enumerate(recipient_patterns):
                    matches = list(pattern.finditer(verse_text))
                    for match in matches:
                        recipient_matches.append({
                            "term": match.group(0),
                            "original_term": recipient_terms[i],
                            "start": match.start(),
                            "end": match.end(),
                            "recipient_type": recipient_category if recipient_category else "unknown"
                        })
                
                # Analyze relationship between worship terms and recipients
                for worship in worship_matches:
                    # Determine likely recipient
                    closest_recipient = None
                    min_distance = float('inf')
                    
                    for recipient in recipient_matches:
                        # Calculate distance
                        w_pos = (worship["start"] + worship["end"]) / 2
                        r_pos = (recipient["start"] + recipient["end"]) / 2
                        distance = abs(w_pos - r_pos)
                        
                        if distance < min_distance:
                            min_distance = distance
                            closest_recipient = recipient
                    
                    # Determine recipient type
                    if closest_recipient:
                        # Check which category it belongs to
                        recipient_term = closest_recipient["term"]
                        
                        if any(re.match(r'\b' + re.escape(term) + r'\b', recipient_term, re.IGNORECASE) 
                              for term in WORSHIP_RECIPIENTS["god_terms"]):
                            recipient_type = "god_terms"
                        elif any(re.match(r'\b' + re.escape(term) + r'\b', recipient_term, re.IGNORECASE) 
                                for term in WORSHIP_RECIPIENTS["jesus_terms"]):
                            recipient_type = "jesus_terms"
                        elif any(re.match(r'\b' + re.escape(term) + r'\b', recipient_term, re.IGNORECASE) 
                                for term in WORSHIP_RECIPIENTS["other_recipients"]):
                            recipient_type = "other_recipients"
                        else:
                            recipient_type = "unknown"
                    else:
                        recipient_term = None
                        recipient_type = "unknown"
                    
                    # Add to results
                    results.append({
                        "book": book_name,
                        "chapter": chapter_num,
                        "verse": verse_num,
                        "reference": f"{book_name} {chapter_num}:{verse_num}",
                        "text": verse_text,
                        "worship_term": worship["term"],
                        "worship_type": worship["worship_type"],
                        "recipient_term": recipient_term,
                        "recipient_type": recipient_type,
                        "distance": min_distance if closest_recipient else None
                    })
    
    # Convert to DataFrame
    if results:
        df = pd.DataFrame(results)
    else:
        # Create empty DataFrame with expected columns
        columns = ["book", "chapter", "verse", "reference", "text", "worship_term",
                  "worship_type", "recipient_term", "recipient_type", "distance"]
        df = pd.DataFrame(columns=columns)
    
    return df

def analyze_proskuneo_usage(bible_dict: Dict[str, Any],
                          books: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Perform specialized analysis of proskuneo usage in biblical texts.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        books: Optional list of books to include
        
    Returns:
        Dictionary with proskuneo analysis results
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Analyze proskuneo usage across the New Testament
        >>> analysis = analyze_proskuneo_usage(
        ...     bible,
        ...     books=["Matthew", "Mark", "Luke", "John", "Acts", "Revelation"]
        ... )
    """
    # Extract proskuneo instances
    proskuneo_df = extract_worship_instances(
        bible_dict,
        worship_category="proskuneo",
        books=books
    )
    
    if proskuneo_df.empty:
        return {
            "total_instances": 0,
            "recipient_distribution": {},
            "book_distribution": {},
            "examples": {}
        }
    
    # Count instances
    total_instances = len(proskuneo_df)
    
    # Analyze recipient distribution
    recipient_counts = proskuneo_df["recipient_type"].value_counts().to_dict()
    
    # Calculate percentages
    recipient_percentages = {}
    for recipient, count in recipient_counts.items():
        recipient_percentages[recipient] = count / total_instances * 100
    
    # Analyze book distribution
    book_counts = proskuneo_df["book"].value_counts().to_dict()
    
    # Get examples for each recipient type
    examples = {}
    for recipient_type in proskuneo_df["recipient_type"].unique():
        # Filter to this recipient type
        recipient_df = proskuneo_df[proskuneo_df["recipient_type"] == recipient_type]
        
        # Get examples
        examples[recipient_type] = []
        for _, row in recipient_df.head(3).iterrows():
            examples[recipient_type].append({
                "reference": row["reference"],
                "text": row["text"],
                "worship_term": row["worship_term"],
                "recipient_term": row["recipient_term"]
            })
    
    # Analyze gospel distribution specifically
    gospels = ["Matthew", "Mark", "Luke", "John"]
    gospel_df = proskuneo_df[proskuneo_df["book"].isin(gospels)]
    
    gospel_stats = {}
    if not gospel_df.empty:
        # Count by gospel
        gospel_counts = gospel_df["book"].value_counts().to_dict()
        
        # Count by recipient in gospels
        for gospel in gospels:
            if gospel in gospel_counts:
                gospel_recipient_df = gospel_df[gospel_df["book"] == gospel]
                recipient_counts = gospel_recipient_df["recipient_type"].value_counts().to_dict()
                
                gospel_stats[gospel] = {
                    "total": gospel_counts.get(gospel, 0),
                    "recipients": recipient_counts
                }
    
    # Prepare results
    results = {
        "total_instances": total_instances,
        "recipient_distribution": {
            "counts": recipient_counts,
            "percentages": recipient_percentages
        },
        "book_distribution": book_counts,
        "gospel_distribution": gospel_stats,
        "examples": examples
    }
    
    return results

def compare_proskuneo_latreo(bible_dict: Dict[str, Any],
                           books: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compare usage patterns of proskuneo vs. latreo across biblical texts.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        books: Optional list of books to include
        
    Returns:
        Dictionary with comparison results
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Compare proskuneo vs. latreo in New Testament
        >>> comparison = compare_proskuneo_latreo(
        ...     bible,
        ...     books=["Matthew", "Mark", "Luke", "John", "Acts", "Romans", "Hebrews", "Revelation"]
        ... )
    """
    # Extract instances of both worship types
    proskuneo_df = extract_worship_instances(
        bible_dict,
        worship_category="proskuneo",
        books=books
    )
    
    latreo_df = extract_worship_instances(
        bible_dict,
        worship_category="latreo",
        books=books
    )
    
    # Count instances
    proskuneo_count = len(proskuneo_df)
    latreo_count = len(latreo_df)
    
    # Analyze recipient distribution
    proskuneo_recipients = proskuneo_df["recipient_type"].value_counts().to_dict() if not proskuneo_df.empty else {}
    latreo_recipients = latreo_df["recipient_type"].value_counts().to_dict() if not latreo_df.empty else {}
    
    # Calculate percentages
    proskuneo_percentages = {}
    for recipient, count in proskuneo_recipients.items():
        proskuneo_percentages[recipient] = count / proskuneo_count * 100 if proskuneo_count > 0 else 0
    
    latreo_percentages = {}
    for recipient, count in latreo_recipients.items():
        latreo_percentages[recipient] = count / latreo_count * 100 if latreo_count > 0 else 0
    
    # Analyze book distribution
    proskuneo_books = proskuneo_df["book"].value_counts().to_dict() if not proskuneo_df.empty else {}
    latreo_books = latreo_df["book"].value_counts().to_dict() if not latreo_df.empty else {}
    
    # Find verses with both terms
    if not proskuneo_df.empty and not latreo_df.empty:
        proskuneo_refs = set(proskuneo_df["reference"])
        latreo_refs = set(latreo_df["reference"])
        
        overlap_refs = proskuneo_refs.intersection(latreo_refs)
        
        # Get details for overlapping references
        overlap_examples = []
        
        for ref in overlap_refs:
            p_row = proskuneo_df[proskuneo_df["reference"] == ref].iloc[0]
            l_row = latreo_df[latreo_df["reference"] == ref].iloc[0]
            
            overlap_examples.append({
                "reference": ref,
                "text": p_row["text"],
                "proskuneo_term": p_row["worship_term"],
                "proskuneo_recipient": p_row["recipient_term"],
                "latreo_term": l_row["worship_term"],
                "latreo_recipient": l_row["recipient_term"]
            })
    else:
        overlap_refs = set()
        overlap_examples = []
    
    # Analyze Jesus as recipient
    proskuneo_jesus = proskuneo_df[proskuneo_df["recipient_type"] == "jesus_terms"] if not proskuneo_df.empty else pd.DataFrame()
    latreo_jesus = latreo_df[latreo_df["recipient_type"] == "jesus_terms"] if not latreo_df.empty else pd.DataFrame()
    
    proskuneo_jesus_count = len(proskuneo_jesus)
    latreo_jesus_count = len(latreo_jesus)
    
    # Get examples for Jesus as recipient
    proskuneo_jesus_examples = []
    for _, row in proskuneo_jesus.head(3).iterrows():
        proskuneo_jesus_examples.append({
            "reference": row["reference"],
            "text": row["text"]
        })
    
    latreo_jesus_examples = []
    for _, row in latreo_jesus.head(3).iterrows():
        latreo_jesus_examples.append({
            "reference": row["reference"],
            "text": row["text"]
        })
    
    # Prepare results
    results = {
        "proskuneo_instances": proskuneo_count,
        "latreo_instances": latreo_count,
        "proskuneo_recipients": {
            "counts": proskuneo_recipients,
            "percentages": proskuneo_percentages
        },
        "latreo_recipients": {
            "counts": latreo_recipients,
            "percentages": latreo_percentages
        },
        "proskuneo_books": proskuneo_books,
        "latreo_books": latreo_books,
        "overlap_count": len(overlap_refs),
        "overlap_examples": overlap_examples,
        "proskuneo_jesus_count": proskuneo_jesus_count,
        "latreo_jesus_count": latreo_jesus_count,
        "proskuneo_jesus_examples": proskuneo_jesus_examples,
        "latreo_jesus_examples": latreo_jesus_examples
    }
    
    return results

def analyze_worship_recipients(bible_dict: Dict[str, Any],
                             worship_categories: Optional[List[str]] = None,
                             recipient: str = "jesus_terms",
                             books: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Analyze instances where a specific recipient receives worship.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        worship_categories: Optional list of worship categories to include
        recipient: Recipient category to analyze
        books: Optional list of books to include
        
    Returns:
        DataFrame with worship instances directed at the specified recipient
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Analyze all worship directed to Jesus in the Gospels
        >>> jesus_worship = analyze_worship_recipients(
        ...     bible,
        ...     recipient="jesus_terms",
        ...     books=["Matthew", "Mark", "Luke", "John"]
        ... )
    """
    # Validate recipient
    if recipient not in WORSHIP_RECIPIENTS:
        raise ValueError(f"Invalid recipient: {recipient}. Choose from {list(WORSHIP_RECIPIENTS.keys())}")
    
    # Determine which worship categories to include
    if worship_categories:
        for category in worship_categories:
            if category not in WORSHIP_TERMS:
                raise ValueError(f"Invalid worship category: {category}. Choose from {list(WORSHIP_TERMS.keys())}")
    else:
        worship_categories = list(WORSHIP_TERMS.keys())
    
    # Extract worship instances for each category
    all_instances = []
    
    for category in worship_categories:
        instances = extract_worship_instances(
            bible_dict,
            worship_category=category,
            recipient_category=recipient,
            books=books
        )
        
        if not instances.empty:
            # Add category column
            instances["worship_category"] = category
            all_instances.append(instances)
    
    # Combine results
    if all_instances:
        combined_df = pd.concat(all_instances, ignore_index=True)
    else:
        # Create empty DataFrame with expected columns
        columns = ["book", "chapter", "verse", "reference", "text", "worship_term",
                  "worship_type", "recipient_term", "recipient_type", "distance", "worship_category"]
        combined_df = pd.DataFrame(columns=columns)
    
    return combined_df

def analyze_worship_contexts(bible_dict: Dict[str, Any],
                           context_window: int = 3,
                           books: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Analyze the context surrounding worship instances to identify patterns.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        context_window: Number of verses before/after to include in context
        books: Optional list of books to include
        
    Returns:
        Dictionary with worship context analysis results
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Analyze worship contexts in the New Testament
        >>> contexts = analyze_worship_contexts(
        ...     bible,
        ...     books=["Matthew", "Mark", "Luke", "John", "Acts", "Revelation"]
        ... )
    """
    # Extract all worship instances
    worship_df = pd.DataFrame()
    
    for category in WORSHIP_TERMS:
        instances = extract_worship_instances(
            bible_dict,
            worship_category=category,
            books=books
        )
        
        if not instances.empty:
            instances["worship_category"] = category
            worship_df = pd.concat([worship_df, instances], ignore_index=True)
    
    if worship_df.empty:
        return {
            "total_instances": 0,
            "common_contexts": {},
            "recipient_contexts": {}
        }
    
    # Build context for each worship instance
    contexts = []
    
    for _, row in worship_df.iterrows():
        book = row["book"]
        chapter = row["chapter"]
        verse = row["verse"]
        
        # Skip if book not in bible_dict
        if book not in bible_dict:
            continue
        
        # Skip if chapter not in book
        if chapter not in bible_dict[book]:
            continue
        
        # Get preceding context
        preceding_verses = []
        for v in range(max(1, verse - context_window), verse):
            if v in bible_dict[book][chapter]:
                preceding_verses.append({
                    "position": v - verse,
                    "text": bible_dict[book][chapter][v]
                })
        
        # Get following context
        following_verses = []
        max_verse = max(bible_dict[book][chapter].keys())
        for v in range(verse + 1, min(max_verse + 1, verse + context_window + 1)):
            if v in bible_dict[book][chapter]:
                following_verses.append({
                    "position": v - verse,
                    "text": bible_dict[book][chapter][v]
                })
        
        # Add to contexts
        contexts.append({
            "reference": row["reference"],
            "worship_category": row["worship_category"],
            "worship_term": row["worship_term"],
            "recipient_type": row["recipient_type"],
            "recipient_term": row["recipient_term"],
            "verse_text": row["text"],
            "preceding_context": preceding_verses,
            "following_context": following_verses
        })
    
    # Analyze common context patterns
    common_preceding = defaultdict(list)
    common_following = defaultdict(list)
    
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    
    for context in contexts:
        category = context["worship_category"]
        
        # Process preceding context
        preceding_words = []
        for verse in context["preceding_context"]:
            words = [w.lower() for w in word_tokenize(verse["text"]) if w.lower() not in stop_words and w.isalpha()]
            preceding_words.extend(words)
        
        common_preceding[category].extend(preceding_words)
        
        # Process following context
        following_words = []
        for verse in context["following_context"]:
            words = [w.lower() for w in word_tokenize(verse["text"]) if w.lower() not in stop_words and w.isalpha()]
            following_words.extend(words)
        
        common_following[category].extend(following_words)
    
    # Count common context words
    common_context_words = {}
    
    for category in WORSHIP_TERMS:
        preceding_counts = Counter(common_preceding.get(category, []))
        following_counts = Counter(common_following.get(category, []))
        
        common_context_words[category] = {
            "preceding": preceding_counts.most_common(10),
            "following": following_counts.most_common(10)
        }
    
    # Analyze context by recipient
    recipient_contexts = {}
    
    for recipient_type in WORSHIP_RECIPIENTS:
        # Filter contexts to this recipient
        recipient_ctx = [ctx for ctx in contexts if ctx["recipient_type"] == recipient_type]
        
        if not recipient_ctx:
            continue
        
        # Process context words
        preceding_words = []
        following_words = []
        
        for ctx in recipient_ctx:
            for verse in ctx["preceding_context"]:
                words = [w.lower() for w in word_tokenize(verse["text"]) if w.lower() not in stop_words and w.isalpha()]
                preceding_words.extend(words)
            
            for verse in ctx["following_context"]:
                words = [w.lower() for w in word_tokenize(verse["text"]) if w.lower() not in stop_words and w.isalpha()]
                following_words.extend(words)
        
        # Count words
        preceding_counts = Counter(preceding_words)
        following_counts = Counter(following_words)
        
        recipient_contexts[recipient_type] = {
            "preceding": preceding_counts.most_common(10),
            "following": following_counts.most_common(10)
        }
    
    # Prepare results
    results = {
        "total_instances": len(worship_df),
        "category_counts": worship_df["worship_category"].value_counts().to_dict(),
        "recipient_counts": worship_df["recipient_type"].value_counts().to_dict(),
        "common_contexts": common_context_words,
        "recipient_contexts": recipient_contexts
    }
    
    return results

def analyze_worship_in_debate_passages(bible_dict: Dict[str, Any],
                                     passages: List[str]) -> Dict[str, Any]:
    """
    Analyze worship language in specific passages mentioned in theological debates.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        passages: List of verse references to analyze
        
    Returns:
        Dictionary with worship analysis for debate passages
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Analyze worship in passages from the Wood-O'Connor debate
        >>> debate_passages = ["John 20:28", "Matthew 28:17", "Hebrews 1:6", "Revelation 5:14"]
        >>> analysis = analyze_worship_in_debate_passages(bible, debate_passages)
    """
    # Parse references
    parsed_refs = []
    
    for ref in passages:
        parts = ref.split()
        if len(parts) < 2:
            continue
            
        # Handle multi-word book names (e.g., "1 Kings")
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
        "passages": [],
        "proskuneo_count": 0,
        "latreo_count": 0,
        "jesus_worship_count": 0
    }
    
    # Extract worship language from each passage
    for book, chapter, verse in parsed_refs:
        # Skip if book/chapter/verse not in bible_dict
        if book not in bible_dict or chapter not in bible_dict[book] or verse not in bible_dict[book][chapter]:
            continue
            
        reference = f"{book} {chapter}:{verse}"
        verse_text = bible_dict[book][chapter][verse]
        
        # Check for worship terms
        worship_terms = []
        
        for category, terms in WORSHIP_TERMS.items():
            for term in terms:
                pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
                if pattern.search(verse_text):
                    worship_terms.append({
                        "term": term,
                        "category": category
                    })
        
        # Check for recipient terms
        recipient_terms = []
        
        for category, terms in WORSHIP_RECIPIENTS.items():
            for term in terms:
                pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
                if pattern.search(verse_text):
                    recipient_terms.append({
                        "term": term,
                        "category": category
                    })
        
        # Count worship types
        has_proskuneo = any(t["category"] == "proskuneo" for t in worship_terms)
        has_latreo = any(t["category"] == "latreo" for t in worship_terms)
        has_jesus_recipient = any(t["category"] == "jesus_terms" for t in recipient_terms)
        
        if has_proskuneo:
            results["proskuneo_count"] += 1
        
        if has_latreo:
            results["latreo_count"] += 1
        
        if has_jesus_recipient and (has_proskuneo or has_latreo):
            results["jesus_worship_count"] += 1
        
        # Add to passage analysis
        passage_analysis = {
            "reference": reference,
            "text": verse_text,
            "worship_terms": worship_terms,
            "recipient_terms": recipient_terms,
            "has_proskuneo": has_proskuneo,
            "has_latreo": has_latreo,
            "has_jesus_recipient": has_jesus_recipient
        }
        
        results["passages"].append(passage_analysis)
    
    return results
