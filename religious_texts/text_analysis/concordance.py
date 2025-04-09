"""
Concordance Generation Module

This module provides functions for generating concordances and analyzing
word usage in context throughout biblical texts, which is essential for
understanding how terms are used across different passages and books.
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union, Any, Tuple, Set

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk

def generate_concordance(bible_dict: Dict[str, Any],
                       search_term: str,
                       context_words: int = 5,
                       case_sensitive: bool = False,
                       books: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Generate a concordance for a search term with surrounding context.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        search_term: Term to search for
        context_words: Number of words before/after to include in context
        case_sensitive: Whether to perform case-sensitive search
        books: Optional list of books to include
        
    Returns:
        DataFrame with concordance entries
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Generate concordance for "logos" in John
        >>> concordance = generate_concordance(
        ...     bible,
        ...     search_term="logos",
        ...     context_words=7,
        ...     books=["John"]
        ... )
    """
    # Initialize results
    results = []
    
    # Compile regex pattern for search
    if case_sensitive:
        pattern = re.compile(r'\b' + re.escape(search_term) + r'\b')
    else:
        pattern = re.compile(r'\b' + re.escape(search_term) + r'\b', re.IGNORECASE)
    
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
                
                # Find all occurrences of search term
                matches = list(pattern.finditer(verse_text))
                
                if not matches:
                    continue
                
                # Reference for this verse
                reference = f"{book_name} {chapter_num}:{verse_num}"
                
                # Process each match
                for match in matches:
                    # Get matched text (preserving original case)
                    matched_text = match.group(0)
                    
                    # Get context
                    tokens = word_tokenize(verse_text)
                    
                    # Find position of matched term in tokens
                    match_positions = []
                    for i, token in enumerate(tokens):
                        if (case_sensitive and token == matched_text) or \
                           (not case_sensitive and token.lower() == matched_text.lower()):
                            match_positions.append(i)
                    
                    # Process each position (for multi-word search terms or multiple occurrences)
                    for pos in match_positions:
                        # Get context before
                        start_pos = max(0, pos - context_words)
                        before_context = " ".join(tokens[start_pos:pos])
                        
                        # Get context after
                        end_pos = min(len(tokens), pos + 1 + context_words)
                        after_context = " ".join(tokens[pos+1:end_pos])
                        
                        # Add to results
                        results.append({
                            "book": book_name,
                            "chapter": chapter_num,
                            "verse": verse_num,
                            "reference": reference,
                            "before_context": before_context,
                            "term": matched_text,
                            "after_context": after_context,
                            "full_verse": verse_text
                        })
    
    # Convert to DataFrame
    if results:
        df = pd.DataFrame(results)
    else:
        # Create empty DataFrame with expected columns
        columns = ["book", "chapter", "verse", "reference", "before_context", 
                  "term", "after_context", "full_verse"]
        df = pd.DataFrame(columns=columns)
    
    return df

def search_phrase(bible_dict: Dict[str, Any],
                phrase: str,
                exact: bool = False,
                case_sensitive: bool = False,
                books: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Search for a phrase in the biblical text.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        phrase: Phrase to search for
        exact: Whether to require exact phrase match
        case_sensitive: Whether to perform case-sensitive search
        books: Optional list of books to include
        
    Returns:
        DataFrame with search results
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Search for "kingdom of heaven" in Matthew
        >>> results = search_phrase(
        ...     bible,
        ...     phrase="kingdom of heaven",
        ...     books=["Matthew"]
        ... )
    """
    # Initialize results
    results = []
    
    # Prepare search pattern
    if exact:
        # Exact phrase match
        if case_sensitive:
            pattern = re.compile(r'\b' + re.escape(phrase) + r'\b')
        else:
            pattern = re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)
    else:
        # Word-level match (words can be in different order or separated)
        words = phrase.split()
        if case_sensitive:
            patterns = [re.compile(r'\b' + re.escape(word) + r'\b') for word in words]
        else:
            patterns = [re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE) for word in words]
    
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
                
                # Check for match
                if exact:
                    matches = list(pattern.finditer(verse_text))
                    match_found = bool(matches)
                    matched_text = matches[0].group(0) if matches else None
                else:
                    # Check if all words are present
                    match_found = all(pattern.search(verse_text) for pattern in patterns)
                    matched_text = phrase  # Use original phrase for non-exact matches
                
                if not match_found:
                    continue
                
                # Add to results
                results.append({
                    "book": book_name,
                    "chapter": chapter_num,
                    "verse": verse_num,
                    "reference": f"{book_name} {chapter_num}:{verse_num}",
                    "text": verse_text,
                    "matched_text": matched_text if exact else None
                })
    
    # Convert to DataFrame
    if results:
        df = pd.DataFrame(results)
    else:
        # Create empty DataFrame with expected columns
        columns = ["book", "chapter", "verse", "reference", "text", "matched_text"]
        df = pd.DataFrame(columns=columns)
    
    return df

def analyze_term_context(bible_dict: Dict[str, Any],
                       term: str,
                       context_window: int = 5,
                       books: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Analyze the context in which a term appears throughout the text.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        term: Term to analyze
        context_window: Number of words before/after to analyze
        books: Optional list of books to include
        
    Returns:
        Dictionary with context analysis results
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Analyze context for "faith" in Paul's epistles
        >>> context_analysis = analyze_term_context(
        ...     bible,
        ...     term="faith",
        ...     books=["Romans", "Galatians", "Ephesians"]
        ... )
    """
    # Get concordance
    concordance_df = generate_concordance(
        bible_dict,
        search_term=term,
        context_words=context_window,
        books=books
    )
    
    if concordance_df.empty:
        return {
            "term": term,
            "occurrences": 0,
            "common_preceding": [],
            "common_following": [],
            "common_contexts": []
        }
    
    # Analyze context
    # 1. Count occurrences
    occurrences = len(concordance_df)
    
    # 2. Analyze preceding words
    preceding_words = []
    for context in concordance_df["before_context"]:
        words = word_tokenize(context)
        preceding_words.extend(words[-min(len(words), context_window):])
    
    # Remove stopwords
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    preceding_filtered = [w.lower() for w in preceding_words if w.lower() not in stop_words and w.isalpha()]
    
    # Count preceding words
    preceding_counts = Counter(preceding_filtered)
    common_preceding = preceding_counts.most_common(10)
    
    # 3. Analyze following words
    following_words = []
    for context in concordance_df["after_context"]:
        words = word_tokenize(context)
        following_words.extend(words[:min(len(words), context_window)])
    
    # Remove stopwords
    following_filtered = [w.lower() for w in following_words if w.lower() not in stop_words and w.isalpha()]
    
    # Count following words
    following_counts = Counter(following_filtered)
    common_following = following_counts.most_common(10)
    
    # 4. Analyze common phrases
    # Combine before and after context
    contexts = []
    for _, row in concordance_df.iterrows():
        context = row["before_context"] + " " + term + " " + row["after_context"]
        contexts.append(context)
    
    # Extract ngrams
    bigrams = []
    trigrams = []
    
    for context in contexts:
        tokens = word_tokenize(context)
        
        # Generate bigrams
        for i in range(len(tokens) - 1):
            if tokens[i].lower() == term.lower() or tokens[i+1].lower() == term.lower():
                bigrams.append((tokens[i].lower(), tokens[i+1].lower()))
        
        # Generate trigrams
        for i in range(len(tokens) - 2):
            if tokens[i].lower() == term.lower() or tokens[i+1].lower() == term.lower() or tokens[i+2].lower() == term.lower():
                trigrams.append((tokens[i].lower(), tokens[i+1].lower(), tokens[i+2].lower()))
    
    # Count ngrams
    bigram_counts = Counter(bigrams)
    trigram_counts = Counter(trigrams)
    
    common_bigrams = bigram_counts.most_common(10)
    common_trigrams = trigram_counts.most_common(10)
    
    # Format common phrases
    common_contexts = []
    
    for bg, count in common_bigrams:
        common_contexts.append({
            "phrase": " ".join(bg),
            "count": count,
            "type": "bigram"
        })
    
    for tg, count in common_trigrams:
        common_contexts.append({
            "phrase": " ".join(tg),
            "count": count,
            "type": "trigram"
        })
    
    # Sort by count
    common_contexts.sort(key=lambda x: x["count"], reverse=True)
    common_contexts = common_contexts[:10]  # Top 10
    
    # 5. Book distribution
    book_counts = concordance_df["book"].value_counts().to_dict()
    
    # Prepare results
    results = {
        "term": term,
        "occurrences": occurrences,
        "common_preceding": [{"word": word, "count": count} for word, count in common_preceding],
        "common_following": [{"word": word, "count": count} for word, count in common_following],
        "common_contexts": common_contexts,
        "book_distribution": [{"book": book, "count": count} for book, count in book_counts.items()]
    }
    
    return results

def generate_keyword_in_context(bible_dict: Dict[str, Any],
                              term: str,
                              context_chars: int = 40,
                              max_results: int = 100,
                              books: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Generate a keyword-in-context (KWIC) display for a search term.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        term: Term to search for
        context_chars: Number of characters before/after to show
        max_results: Maximum number of results to return
        books: Optional list of books to include
        
    Returns:
        DataFrame with KWIC display
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Generate KWIC display for "logos" in John
        >>> kwic = generate_keyword_in_context(
        ...     bible,
        ...     term="logos",
        ...     books=["John"]
        ... )
    """
    # Initialize results
    results = []
    
    # Compile regex pattern for search
    pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
    
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
                
                # Find all occurrences of search term
                matches = list(pattern.finditer(verse_text))
                
                if not matches:
                    continue
                
                # Reference for this verse
                reference = f"{book_name} {chapter_num}:{verse_num}"
                
                # Process each match
                for match in matches:
                    # Get matched text (preserving original case)
                    matched_text = match.group(0)
                    
                    # Get match position
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Get context
                    before_context = verse_text[max(0, start_pos - context_chars):start_pos]
                    after_context = verse_text[end_pos:min(len(verse_text), end_pos + context_chars)]
                    
                    # Add to results
                    results.append({
                        "reference": reference,
                        "book": book_name,
                        "chapter": chapter_num,
                        "verse": verse_num,
                        "before_context": before_context,
                        "term": matched_text,
                        "after_context": after_context,
                        "kwic": f"{before_context} [{matched_text}] {after_context}"
                    })
                    
                    # Check if we've reached the maximum
                    if len(results) >= max_results:
                        break
            
            # Check if we've reached the maximum
            if len(results) >= max_results:
                break
        
        # Check if we've reached the maximum
        if len(results) >= max_results:
            break
    
    # Convert to DataFrame
    if results:
        df = pd.DataFrame(results)
    else:
        # Create empty DataFrame with expected columns
        columns = ["reference", "book", "chapter", "verse", "before_context", 
                  "term", "after_context", "kwic"]
        df = pd.DataFrame(columns=columns)
    
    return df

def compare_word_usage(bible_dict: Dict[str, Any],
                     words: List[str],
                     context_analysis: bool = True,
                     books: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compare usage patterns of multiple words across biblical texts.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        words: List of words to compare
        context_analysis: Whether to include context analysis
        books: Optional list of books to include
        
    Returns:
        Dictionary with comparison results
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Compare usage of "love" and "faith" in Paul's epistles
        >>> comparison = compare_word_usage(
        ...     bible,
        ...     words=["love", "faith"],
        ...     books=["Romans", "1 Corinthians", "Galatians", "Ephesians"]
        ... )
    """
    # Initialize results
    results = {
        "words": words,
        "total_occurrences": {},
        "book_distributions": {},
        "co_occurrences": 0,
        "context_similarities": {},
        "distinctive_contexts": {}
    }
    
    # Get concordance for each word
    concordances = {}
    for word in words:
        concordances[word] = generate_concordance(
            bible_dict,
            search_term=word,
            books=books
        )
        
        # Count occurrences
        results["total_occurrences"][word] = len(concordances[word])
        
        # Book distribution
        if not concordances[word].empty:
            book_counts = concordances[word]["book"].value_counts().to_dict()
            results["book_distributions"][word] = book_counts
    
    # Calculate co-occurrences (words in same verse)
    if len(words) > 1:
        verse_sets = {}
        for word in words:
            if not concordances[word].empty:
                verse_sets[word] = set(concordances[word]["reference"])
        
        # Find intersection of all verse sets
        if verse_sets:
            co_occurrence_verses = set.intersection(*verse_sets.values())
            results["co_occurrences"] = len(co_occurrence_verses)
            
            # List co-occurrence references
            results["co_occurrence_references"] = sorted(co_occurrence_verses)
    
    # Context analysis if requested
    if context_analysis:
        # Initialize context collections
        preceding_contexts = {}
        following_contexts = {}
        
        # Collect contexts for each word
        for word in words:
            if concordances[word].empty:
                continue
            
            # Get contexts
            preceding = []
            following = []
            
            for _, row in concordances[word].iterrows():
                preceding_words = word_tokenize(row["before_context"])
                following_words = word_tokenize(row["after_context"])
                
                # Remove stopwords
                from nltk.corpus import stopwords
                stop_words = set(stopwords.words('english'))
                
                preceding.extend([w.lower() for w in preceding_words 
                                 if w.lower() not in stop_words and w.isalpha()])
                
                following.extend([w.lower() for w in following_words 
                                if w.lower() not in stop_words and w.isalpha()])
            
            # Store context word counts
            preceding_contexts[word] = Counter(preceding)
            following_contexts[word] = Counter(following)
        
        # Calculate similarity between contexts
        if len(words) > 1:
            for i, word1 in enumerate(words):
                for word2 in words[i+1:]:
                    # Skip if either word not found
                    if word1 not in preceding_contexts or word2 not in preceding_contexts:
                        continue
                    
                    # Calculate Jaccard similarity for preceding contexts
                    prec1 = set(preceding_contexts[word1].keys())
                    prec2 = set(preceding_contexts[word2].keys())
                    
                    preceding_similarity = len(prec1.intersection(prec2)) / len(prec1.union(prec2)) if prec1 or prec2 else 0
                    
                    # Calculate Jaccard similarity for following contexts
                    foll1 = set(following_contexts[word1].keys())
                    foll2 = set(following_contexts[word2].keys())
                    
                    following_similarity = len(foll1.intersection(foll2)) / len(foll1.union(foll2)) if foll1 or foll2 else 0
                    
                    # Store similarity scores
                    pair_key = f"{word1}_{word2}"
                    results["context_similarities"][pair_key] = {
                        "preceding_similarity": preceding_similarity,
                        "following_similarity": following_similarity,
                        "overall_similarity": (preceding_similarity + following_similarity) / 2
                    }
        
        # Find distinctive contexts for each word
        for word in words:
            # Skip if word not found
            if word not in preceding_contexts:
                continue
            
            # Get all context words for this word
            word_contexts = set(preceding_contexts[word].keys()).union(set(following_contexts[word].keys()))
            
            # Get all context words for other words
            other_contexts = set()
            for other_word in words:
                if other_word != word and other_word in preceding_contexts:
                    other_contexts.update(preceding_contexts[other_word].keys())
                    other_contexts.update(following_contexts[other_word].keys())
            
            # Find distinctive context words (in this word's context but not others)
            distinctive_contexts = word_contexts - other_contexts
            
            # Get the most common distinctive context words
            distinctive_words = []
            
            for context_word in distinctive_contexts:
                prec_count = preceding_contexts[word].get(context_word, 0)
                foll_count = following_contexts[word].get(context_word, 0)
                total_count = prec_count + foll_count
                
                if total_count > 0:
                    distinctive_words.append((context_word, total_count))
            
            # Sort by count and take top 10
            distinctive_words.sort(key=lambda x: x[1], reverse=True)
            results["distinctive_contexts"][word] = [
                {"word": w, "count": c} for w, c in distinctive_words[:10]
            ]
    
    return results

def build_comprehensive_concordance(bible_dict: Dict[str, Any],
                                  search_terms: List[str],
                                  books: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """
    Build a comprehensive concordance for multiple search terms.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        search_terms: List of terms to include in concordance
        books: Optional list of books to include
        
    Returns:
        Dictionary mapping terms to concordance DataFrames
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Build concordance for salvation-related terms
        >>> terms = ["salvation", "save", "redeem", "justify"]
        >>> concordances = build_comprehensive_concordance(
        ...     bible,
        ...     search_terms=terms,
        ...     books=["Romans", "Galatians", "Ephesians"]
        ... )
    """
    # Generate concordance for each term
    concordances = {}
    
    for term in search_terms:
        concordances[term] = generate_concordance(
            bible_dict,
            search_term=term,
            books=books
        )
    
    return concordances
