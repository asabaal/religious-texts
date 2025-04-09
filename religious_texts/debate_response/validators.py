"""
Statistical Claim Validators Module

This module provides functions for validating statistical claims about biblical content,
including word frequency, co-occurrence patterns, and distribution claims.
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Callable
import math

import pandas as pd
import numpy as np
from scipy import stats


def validate_word_frequency_claim(bible_dict: Dict[str, Any], 
                                claim: str,
                                terms: List[str],
                                books: Optional[List[str]] = None,
                                threshold: float = 0.05) -> Dict[str, Any]:
    """
    Validate claims about the frequency of specific words in biblical texts.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        claim: Description of the claim to validate
        terms: List of terms mentioned in the claim
        books: Optional list of specific books to analyze
        threshold: Significance threshold for statistical tests
        
    Returns:
        Dictionary with validation results
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Validate claim: "The word 'love' appears more frequently in John than in any other Gospel"
        >>> result = validate_word_frequency_claim(
        ...     bible,
        ...     claim="The word 'love' appears more frequently in John than in any other Gospel",
        ...     terms=["love"],
        ...     books=["Matthew", "Mark", "Luke", "John"]
        ... )
        >>> print(result['valid'])
    """
    from religious_texts.text_analysis import frequency
    
    # Initialize result
    result = {
        'claim': claim,
        'terms': terms,
        'books': books,
        'valid': None,
        'explanation': None,
        'data': None,
        'statistical_tests': None
    }
    
    # Check claim type based on keywords
    comparative_claim = any(word in claim.lower() for word in 
                           ['more', 'less', 'than', 'most', 'least', 'highest', 'lowest'])
    
    absolute_claim = any(word in claim.lower() for word in 
                        ['times', 'occurrences', 'mentions', 'frequency', 'count'])
    
    # Extract data for validation
    term_data = {}
    
    for term in terms:
        # Get relative frequency data
        rel_freq = frequency.relative_frequency(bible_dict, term, books=books)
        
        if rel_freq.empty:
            result['valid'] = False
            result['explanation'] = f"Term '{term}' not found in the specified texts."
            return result
        
        term_data[term] = rel_freq
    
    # Combined data for all terms
    combined_df = pd.concat([term_data[term] for term in terms])
    combined_df = combined_df.groupby('book').sum().reset_index()
    
    # Store data for return
    result['data'] = combined_df
    
    # Validate based on claim type
    if comparative_claim:
        # Check if it's a claim about a specific book having highest/lowest frequency
        book_comparison = None
        for book in combined_df['book']:
            if book.lower() in claim.lower():
                book_comparison = book
                break
        
        if book_comparison:
            # Check if book has highest/lowest frequency
            if 'more' in claim.lower() or 'most' in claim.lower() or 'highest' in claim.lower():
                max_book = combined_df.loc[combined_df['relative_frequency'].idxmax()]['book']
                is_valid = book_comparison == max_book
                
                explanation = f"The book with the highest relative frequency of {', '.join(terms)} is {max_book} "
                explanation += f"({combined_df[combined_df['book'] == max_book]['relative_frequency'].values[0]:.2f} per 1000 words). "
                
                if is_valid:
                    explanation += f"The claim that {book_comparison} has the highest frequency is correct."
                else:
                    explanation += f"The claim that {book_comparison} has the highest frequency is incorrect."
                
                result['valid'] = is_valid
                result['explanation'] = explanation
            
            elif 'less' in claim.lower() or 'least' in claim.lower() or 'lowest' in claim.lower():
                min_book = combined_df.loc[combined_df['relative_frequency'].idxmin()]['book']
                is_valid = book_comparison == min_book
                
                explanation = f"The book with the lowest relative frequency of {', '.join(terms)} is {min_book} "
                explanation += f"({combined_df[combined_df['book'] == min_book]['relative_frequency'].values[0]:.2f} per 1000 words). "
                
                if is_valid:
                    explanation += f"The claim that {book_comparison} has the lowest frequency is correct."
                else:
                    explanation += f"The claim that {book_comparison} has the lowest frequency is incorrect."
                
                result['valid'] = is_valid
                result['explanation'] = explanation
                
        # Check for comparison between two specific books
        else:
            # Try to find two books mentioned in the claim
            mentioned_books = []
            for book in combined_df['book']:
                if book.lower() in claim.lower():
                    mentioned_books.append(book)
            
            if len(mentioned_books) >= 2:
                book1, book2 = mentioned_books[:2]
                freq1 = combined_df[combined_df['book'] == book1]['relative_frequency'].values[0]
                freq2 = combined_df[combined_df['book'] == book2]['relative_frequency'].values[0]
                
                # Determine expected relationship based on claim
                if 'more' in claim.lower() and book1.lower() in claim.lower().split('more')[0]:
                    expected = freq1 > freq2
                elif 'more' in claim.lower() and book2.lower() in claim.lower().split('more')[0]:
                    expected = freq2 > freq1
                elif 'less' in claim.lower() and book1.lower() in claim.lower().split('less')[0]:
                    expected = freq1 < freq2
                elif 'less' in claim.lower() and book2.lower() in claim.lower().split('less')[0]:
                    expected = freq2 < freq1
                else:
                    # Can't determine expected relationship
                    result['valid'] = None
                    result['explanation'] = f"Cannot determine expected relationship between {book1} and {book2} from claim."
                    return result
                
                # Validate based on actual data
                result['valid'] = expected
                result['explanation'] = f"Relative frequency in {book1}: {freq1:.2f} per 1000 words. "
                result['explanation'] += f"Relative frequency in {book2}: {freq2:.2f} per 1000 words. "
                
                if expected:
                    result['explanation'] += f"The claim about the relationship between {book1} and {book2} is correct."
                else:
                    result['explanation'] += f"The claim about the relationship between {book1} and {book2} is incorrect."
    
    elif absolute_claim:
        # Try to extract numeric value from claim
        numbers = re.findall(r'\b\d+\b', claim)
        
        if numbers:
            claimed_count = int(numbers[0])
            
            # Check if the claim is about a specific book
            mentioned_book = None
            for book in combined_df['book']:
                if book.lower() in claim.lower():
                    mentioned_book = book
                    break
            
            if mentioned_book:
                # Compare actual count to claimed count
                actual_count = combined_df[combined_df['book'] == mentioned_book]['word_count'].values[0]
                
                # Check for approximation terms
                approx_terms = ['about', 'approximately', 'around', 'roughly', 'nearly']
                is_approximate = any(term in claim.lower() for term in approx_terms)
                
                if is_approximate:
                    # Allow 10% margin of error for approximate claims
                    margin = 0.1 * claimed_count
                    is_valid = abs(actual_count - claimed_count) <= margin
                else:
                    # Exact match required
                    is_valid = actual_count == claimed_count
                
                result['valid'] = is_valid
                result['explanation'] = f"Actual count in {mentioned_book}: {actual_count}. Claimed count: {claimed_count}. "
                
                if is_approximate:
                    result['explanation'] += f"With approximate matching (±10%), the claim is {'correct' if is_valid else 'incorrect'}."
                else:
                    result['explanation'] += f"The claim is {'correct' if is_valid else 'incorrect'}."
            
            else:
                # Assume the claim is about total occurrences across all analyzed books
                total_count = combined_df['word_count'].sum()
                
                # Check for approximation terms
                approx_terms = ['about', 'approximately', 'around', 'roughly', 'nearly']
                is_approximate = any(term in claim.lower() for term in approx_terms)
                
                if is_approximate:
                    # Allow 10% margin of error for approximate claims
                    margin = 0.1 * claimed_count
                    is_valid = abs(total_count - claimed_count) <= margin
                else:
                    # Exact match required
                    is_valid = total_count == claimed_count
                
                result['valid'] = is_valid
                result['explanation'] = f"Actual total count: {total_count}. Claimed count: {claimed_count}. "
                
                if is_approximate:
                    result['explanation'] += f"With approximate matching (±10%), the claim is {'correct' if is_valid else 'incorrect'}."
                else:
                    result['explanation'] += f"The claim is {'correct' if is_valid else 'incorrect'}."
    
    else:
        # Generic claim - provide summary information
        result['valid'] = None
        result['explanation'] = "Could not identify a specific statistical claim to validate. "
        result['explanation'] += f"Total occurrences of {', '.join(terms)}: {combined_df['word_count'].sum()}. "
        
        # Add top 3 books by frequency
        top_books = combined_df.sort_values('relative_frequency', ascending=False).head(3)
        result['explanation'] += f"Top 3 books by frequency: "
        for i, row in top_books.iterrows():
            result['explanation'] += f"{row['book']} ({row['relative_frequency']:.2f} per 1000 words), "
        
        result['explanation'] = result['explanation'].rstrip(', ') + '.'
    
    return result


def validate_cooccurrence_claim(bible_dict: Dict[str, Any],
                              claim: str,
                              terms: List[str],
                              max_distance: int = 5,
                              books: Optional[List[str]] = None,
                              threshold: float = 0.05) -> Dict[str, Any]:
    """
    Validate claims about the co-occurrence of terms in biblical texts.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        claim: Description of the claim to validate
        terms: List of terms mentioned in the claim
        max_distance: Maximum distance between terms to count as co-occurrence
        books: Optional list of specific books to analyze
        threshold: Significance threshold for statistical tests
        
    Returns:
        Dictionary with validation results
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Validate claim: "'Faith' and 'works' frequently appear together in James"
        >>> result = validate_cooccurrence_claim(
        ...     bible,
        ...     claim="'Faith' and 'works' frequently appear together in James",
        ...     terms=["faith", "works"],
        ...     books=["James"]
        ... )
        >>> print(result['valid'])
    """
    from religious_texts.text_analysis.cooccurrence import word_cooccurrence
    
    # Initialize result
    result = {
        'claim': claim,
        'terms': terms,
        'books': books,
        'valid': None,
        'explanation': None,
        'data': None,
        'statistical_tests': None
    }
    
    if len(terms) < 2:
        result['valid'] = False
        result['explanation'] = "Co-occurrence claims require at least two terms to validate."
        return result
    
    # Get co-occurrence data
    cooccur_df = word_cooccurrence(
        bible_dict, 
        terms, 
        window_size=max_distance, 
        unit='window', 
        books=books
    )
    
    if cooccur_df.empty:
        result['valid'] = False
        result['explanation'] = f"Terms {', '.join(terms)} not found in the specified texts."
        return result
    
    # Store data for return
    result['data'] = cooccur_df
    
    # Extract co-occurrence counts
    cooccur_counts = {}
    for i, term1 in enumerate(terms):
        for term2 in terms[i+1:]:
            # Get co-occurrence count (symmetric)
            count = max(
                cooccur_df.loc[term1, term2] if term1 in cooccur_df.index and term2 in cooccur_df.columns else 0,
                cooccur_df.loc[term2, term1] if term2 in cooccur_df.index and term1 in cooccur_df.columns else 0
            )
            
            cooccur_counts[(term1, term2)] = count
    
    # Calculate individual term frequencies for comparison
    from religious_texts.text_analysis import frequency
    
    term_freqs = {}
    for term in terms:
        rel_freq = frequency.relative_frequency(bible_dict, term, books=books)
        
        if not rel_freq.empty:
            term_freqs[term] = rel_freq['word_count'].sum()
        else:
            term_freqs[term] = 0
    
    # Calculate expected co-occurrences based on random distribution
    # This is a simplified model assuming terms are randomly distributed in the text
    total_words = 0
    for book_name, chapters in bible_dict.items():
        if books and book_name not in books:
            continue
            
        for chapter_verses in chapters.values():
            for verse_text in chapter_verses.values():
                total_words += len(verse_text.split())
    
    expected_cooccur = {}
    for term1, term2 in cooccur_counts.keys():
        # Probability of finding terms within max_distance of each other
        # This is a simplification - a more accurate model would account for sentence boundaries, etc.
        prob1 = term_freqs[term1] / total_words
        prob2 = term_freqs[term2] / total_words
        
        # Expected co-occurrences within window_size
        window_count = total_words - max_distance + 1  # Number of possible windows
        expected = window_count * prob1 * prob2 * (2 * max_distance + 1)  # Expanded window for both directions
        
        expected_cooccur[(term1, term2)] = expected
    
    # Evaluate claim based on observed vs. expected co-occurrences
    significant_pairs = []
    
    for (term1, term2), observed in cooccur_counts.items():
        expected = expected_cooccur[(term1, term2)]
        
        # Skip if no occurrences
        if observed == 0:
            continue
        
        # Calculate significance (binomial test)
        # Null hypothesis: terms co-occur at rate expected by chance
        p_value = stats.binom_test(observed, n=int(total_words), p=expected/total_words, alternative='greater')
        
        # Check if statistically significant
        if p_value < threshold:
            significant_pairs.append((term1, term2, observed, expected, p_value))
    
    # Store statistical tests
    result['statistical_tests'] = {
        'method': 'binomial_test',
        'observed': {f"{t1}-{t2}": count for (t1, t2), count in cooccur_counts.items()},
        'expected': {f"{t1}-{t2}": count for (t1, t2), count in expected_cooccur.items()},
        'significant_pairs': significant_pairs
    }
    
    # Validate claim based on results
    if 'frequently' in claim.lower() or 'often' in claim.lower() or 'commonly' in claim.lower():
        # Claim is about significant co-occurrence
        result['valid'] = len(significant_pairs) > 0
        
        if result['valid']:
            result['explanation'] = f"The terms {', '.join(terms)} co-occur significantly more often than expected by chance. "
            result['explanation'] += f"This supports the claim that they frequently appear together."
        else:
            result['explanation'] = f"The terms {', '.join(terms)} do not co-occur more frequently than expected by chance. "
            result['explanation'] += f"This does not support the claim that they frequently appear together."
    
    elif 'rarely' in claim.lower() or 'seldom' in claim.lower() or 'infrequently' in claim.lower():
        # Claim is about lack of co-occurrence
        result['valid'] = len(significant_pairs) == 0
        
        if result['valid']:
            result['explanation'] = f"The terms {', '.join(terms)} do not co-occur more frequently than expected by chance. "
            result['explanation'] += f"This supports the claim that they rarely appear together."
        else:
            result['explanation'] = f"The terms {', '.join(terms)} co-occur significantly more often than expected by chance. "
            result['explanation'] += f"This contradicts the claim that they rarely appear together."
    
    else:
        # Generic claim or specific count claim
        numbers = re.findall(r'\b\d+\b', claim)
        
        if numbers:
            # Claim about specific number of co-occurrences
            claimed_count = int(numbers[0])
            actual_count = sum(cooccur_counts.values())
            
            # Allow 10% margin for approximate claims
            approx_terms = ['about', 'approximately', 'around', 'roughly', 'nearly']
            is_approximate = any(term in claim.lower() for term in approx_terms)
            
            if is_approximate:
                margin = 0.1 * claimed_count
                result['valid'] = abs(actual_count - claimed_count) <= margin
            else:
                result['valid'] = actual_count == claimed_count
            
            result['explanation'] = f"Actual co-occurrence count: {actual_count}. Claimed count: {claimed_count}. "
            
            if is_approximate:
                result['explanation'] += f"With approximate matching (±10%), the claim is {'correct' if result['valid'] else 'incorrect'}."
            else:
                result['explanation'] += f"The claim is {'correct' if result['valid'] else 'incorrect'}."
        
        else:
            # Generic claim - provide summary information
            result['valid'] = None
            result['explanation'] = "Could not identify a specific co-occurrence claim to validate. "
            
            # Summarize co-occurrences
            result['explanation'] += f"Total co-occurrences among {', '.join(terms)}: "
            result['explanation'] += f"{sum(cooccur_counts.values())}. "
            
            # Note significant pairs
            if significant_pairs:
                result['explanation'] += f"Significant co-occurrences: "
                for term1, term2, obs, exp, p in significant_pairs:
                    result['explanation'] += f"{term1}-{term2} ({obs} observed vs. {exp:.1f} expected), "
                
                result['explanation'] = result['explanation'].rstrip(', ') + '.'
            else:
                result['explanation'] += f"No significant co-occurrences found."
    
    return result


def validate_theological_claim(bible_dict: Dict[str, Any],
                             claim: str,
                             related_terms: Dict[str, List[str]],
                             books: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Validate theological claims about biblical content based on textual evidence.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        claim: Description of the theological claim to validate
        related_terms: Dictionary mapping concepts to related terms
        books: Optional list of specific books to analyze
        
    Returns:
        Dictionary with validation results
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Validate claim: "Paul emphasizes faith over works in Romans"
        >>> terms = {
        ...     "faith": ["faith", "believe", "trust"],
        ...     "works": ["works", "deeds", "labor", "do", "act"]
        ... }
        >>> result = validate_theological_claim(
        ...     bible,
        ...     claim="Paul emphasizes faith over works in Romans",
        ...     related_terms=terms,
        ...     books=["Romans"]
        ... )
        >>> print(result['valid'])
    """
    from religious_texts.text_analysis import frequency
    from religious_texts.text_analysis.cooccurrence import word_cooccurrence
    
    # Initialize result
    result = {
        'claim': claim,
        'related_terms': related_terms,
        'books': books,
        'valid': None,
        'explanation': None,
        'evidence': [],
        'counter_evidence': []
    }
    
    # Extract main concepts from claim
    concepts = list(related_terms.keys())
    
    if len(concepts) < 1:
        result['valid'] = False
        result['explanation'] = "Need at least one concept to validate a theological claim."
        return result
    
    # Collect evidence for verification
    # 1. Term frequency comparison
    concept_freqs = {}
    
    for concept, terms in related_terms.items():
        # Combine frequencies for all related terms
        total_freq = 0
        term_data = []
        
        for term in terms:
            rel_freq = frequency.relative_frequency(bible_dict, term, books=books)
            
            if not rel_freq.empty:
                # Calculate total relative frequency
                freq_sum = rel_freq['relative_frequency'].sum()
                total_freq += freq_sum
                
                # Store individual term data
                term_data.append({
                    'term': term,
                    'freq': freq_sum,
                    'references': rel_freq
                })
        
        concept_freqs[concept] = {
            'total_freq': total_freq,
            'term_data': term_data
        }
    
    # 2. Co-occurrence patterns
    all_terms = []
    for terms in related_terms.values():
        all_terms.extend(terms)
    
    cooccur_df = word_cooccurrence(bible_dict, all_terms, unit='verse', books=books)
    
    # 3. Analyze claim structure for comparison terms
    comparison_claim = False
    compared_concepts = []
    
    comparison_terms = ['more than', 'less than', 'over', 'above', 'below', 'versus', 'vs', 'against', 'rather than']
    
    for comp_term in comparison_terms:
        if comp_term in claim.lower():
            comparison_claim = True
            
            # Try to identify which concepts are being compared
            parts = claim.lower().split(comp_term)
            for concept in concepts:
                if concept.lower() in parts[0]:
                    compared_concepts.append(concept)
                    break
            
            for concept in concepts:
                if concept.lower() in parts[1]:
                    compared_concepts.append(concept)
                    break
            
            break
    
    # 4. Check for emphasis claims
    emphasis_claim = False
    emphasized_concept = None
    
    emphasis_terms = ['emphasizes', 'emphasize', 'emphasis', 'focus', 'focuses', 'highlight', 'highlights', 'stress', 'stresses']
    
    for emph_term in emphasis_terms:
        if emph_term in claim.lower():
            emphasis_claim = True
            
            # Try to identify which concept is emphasized
            for concept in concepts:
                if concept.lower() in claim.lower():
                    emphasized_concept = concept
                    break
            
            break
    
    # Evaluate the claim based on evidence
    if comparison_claim and len(compared_concepts) == 2:
        # Handle comparison claim (e.g., "faith over works")
        concept1, concept2 = compared_concepts
        freq1 = concept_freqs[concept1]['total_freq']
        freq2 = concept_freqs[concept2]['total_freq']
        
        # Determine expected relationship based on claim
        if any(term in claim.lower() for term in ['more than', 'over', 'above']):
            expected = freq1 > freq2
            relation = "greater than"
        elif any(term in claim.lower() for term in ['less than', 'below']):
            expected = freq1 < freq2
            relation = "less than"
        else:
            # Can't determine expected relationship
            expected = None
            relation = "compared to"
        
        # Calculate statistical significance
        # Simple approach: check if difference is substantial (>20%)
        if freq1 > 0 and freq2 > 0:
            ratio = freq1 / freq2
            substantial = ratio < 0.8 or ratio > 1.2
        else:
            substantial = freq1 > 0 or freq2 > 0
        
        # Validate based on data
        if expected is not None:
            result['valid'] = expected and substantial
        else:
            result['valid'] = None
        
        # Construct explanation
        result['explanation'] = f"Relative frequency of '{concept1}' related terms: {freq1:.2f} per 1000 words. "
        result['explanation'] += f"Relative frequency of '{concept2}' related terms: {freq2:.2f} per 1000 words. "
        
        if expected is not None:
            if result['valid']:
                result['explanation'] += f"The data supports the claim that '{concept1}' is {relation} '{concept2}'."
            else:
                result['explanation'] += f"The data does not support the claim that '{concept1}' is {relation} '{concept2}'."
        
        # Add evidence
        result['evidence'].append({
            'type': 'frequency_comparison',
            'concept1': concept1,
            'concept2': concept2,
            'freq1': freq1,
            'freq2': freq2,
            'ratio': freq1 / freq2 if freq2 > 0 else float('inf'),
            'substantial': substantial
        })
    
    elif emphasis_claim and emphasized_concept:
        # Handle emphasis claim (e.g., "Paul emphasizes faith")
        # Check if the emphasized concept is significantly more frequent than others
        concept_freq = concept_freqs[emphasized_concept]['total_freq']
        other_freqs = [freq['total_freq'] for concept, freq in concept_freqs.items() 
                      if concept != emphasized_concept]
        
        if other_freqs:
            avg_other_freq = sum(other_freqs) / len(other_freqs)
            
            # Check if emphasized concept is significantly more frequent (>50% more)
            significant = concept_freq > 1.5 * avg_other_freq
            
            result['valid'] = significant
            
            # Construct explanation
            result['explanation'] = f"Relative frequency of '{emphasized_concept}' related terms: {concept_freq:.2f} per 1000 words. "
            result['explanation'] += f"Average frequency of other concepts: {avg_other_freq:.2f} per 1000 words. "
            
            if result['valid']:
                result['explanation'] += f"The data supports the claim that '{emphasized_concept}' is emphasized."
            else:
                result['explanation'] += f"The data does not support the claim that '{emphasized_concept}' is emphasized."
            
            # Add evidence
            result['evidence'].append({
                'type': 'emphasis_analysis',
                'emphasized_concept': emphasized_concept,
                'concept_freq': concept_freq,
                'avg_other_freq': avg_other_freq,
                'ratio': concept_freq / avg_other_freq if avg_other_freq > 0 else float('inf'),
                'significant': significant
            })
        else:
            result['valid'] = None
            result['explanation'] = f"Could not validate emphasis claim due to lack of comparison concepts."
    
    else:
        # Generic theological claim - provide summary information
        result['valid'] = None
        result['explanation'] = "Could not identify a specific theological claim pattern to validate. "
        
        # Summarize concept frequencies
        result['explanation'] += f"Concept frequencies (per 1000 words): "
        for concept, data in concept_freqs.items():
            result['explanation'] += f"{concept}: {data['total_freq']:.2f}, "
        
        result['explanation'] = result['explanation'].rstrip(', ') + '.'
    
    return result


def statistical_confidence(data: Union[Dict[str, Any], pd.DataFrame], 
                          confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    Calculate statistical confidence for biblical text analysis results.
    
    Args:
        data: Dictionary or DataFrame with analysis results
        confidence_level: Desired confidence level (0-1)
        
    Returns:
        Dictionary with confidence intervals and statistical metrics
        
    Example:
        >>> from religious_texts.text_analysis import frequency
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Get frequency data for 'love' in the Gospels
        >>> freq_data = frequency.relative_frequency(
        ...     bible, 
        ...     "love", 
        ...     books=["Matthew", "Mark", "Luke", "John"]
        ... )
        >>> # Calculate statistical confidence
        >>> conf = statistical_confidence(freq_data)
        >>> print(conf['confidence_interval'])
    """
    # Initialize result
    result = {
        'confidence_level': confidence_level,
        'confidence_interval': None,
        'standard_error': None,
        'margin_of_error': None,
        'sample_size': None,
        'warning': None
    }
    
    # Convert dictionary to DataFrame if needed
    if isinstance(data, dict):
        if 'data' in data and isinstance(data['data'], pd.DataFrame):
            df = data['data']
        else:
            result['warning'] = "Input data not in expected format. Need DataFrame or dict with 'data' key."
            return result
    else:
        df = data
    
    # Check if DataFrame has numeric data
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        result['warning'] = "No numeric columns found in data."
        return result
    
    # Try to identify key metrics for confidence calculation
    frequency_col = None
    count_col = None
    
    for col in numeric_cols:
        col_lower = col.lower()
        if 'freq' in col_lower:
            frequency_col = col
        elif 'count' in col_lower:
            count_col = col
    
    if not frequency_col and not count_col:
        # Use first numeric column
        target_col = numeric_cols[0]
    else:
        # Prefer frequency over count
        target_col = frequency_col if frequency_col else count_col
    
    # Calculate statistics
    values = df[target_col].dropna()
    
    if len(values) == 0:
        result['warning'] = f"No valid data in column '{target_col}'."
        return result
    
    # Sample size
    n = len(values)
    result['sample_size'] = n
    
    # Mean and standard deviation
    mean = values.mean()
    std = values.std()
    
    # Standard error
    se = std / math.sqrt(n)
    result['standard_error'] = se
    
    # Critical value for confidence level
    alpha = 1 - confidence_level
    critical_value = stats.t.ppf(1 - alpha/2, n-1)
    
    # Margin of error
    margin = critical_value * se
    result['margin_of_error'] = margin
    
    # Confidence interval
    lower = mean - margin
    upper = mean + margin
    result['confidence_interval'] = (float(lower), float(upper))
    
    # Additional statistics
    result['mean'] = float(mean)
    result['std_dev'] = float(std)
    result['critical_value'] = float(critical_value)
    result['degrees_of_freedom'] = n - 1
    
    return result


def check_distribution_claim(bible_dict: Dict[str, Any],
                           claim: str,
                           term: str,
                           distribution_type: str = 'temporal',
                           books: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Validate claims about the distribution of terms across the biblical text.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        claim: Description of the distribution claim to validate
        term: Term to analyze distribution for
        distribution_type: Type of distribution to analyze ('temporal', 'genre', 'author', etc.)
        books: Optional list of specific books to analyze
        
    Returns:
        Dictionary with validation results
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Validate claim: "The concept of 'grace' is more prevalent in the New Testament than the Old Testament"
        >>> result = check_distribution_claim(
        ...     bible,
        ...     claim="The concept of 'grace' is more prevalent in the New Testament than the Old Testament",
        ...     term="grace",
        ...     distribution_type="testament"
        ... )
        >>> print(result['valid'])
    """
    from religious_texts.theological_analysis.divine_names import divine_name_usage
    from religious_texts.text_analysis import frequency
    
    # Initialize result
    result = {
        'claim': claim,
        'term': term,
        'distribution_type': distribution_type,
        'books': books,
        'valid': None,
        'explanation': None,
        'data': None
    }
    
    # Get frequency data
    rel_freq = frequency.relative_frequency(bible_dict, term, books=books)
    
    if rel_freq.empty:
        result['valid'] = False
        result['explanation'] = f"Term '{term}' not found in the specified texts."
        return result
    
    # Store raw data
    result['data'] = rel_freq
    
    # Process based on distribution type
    if distribution_type.lower() == 'temporal' or distribution_type.lower() == 'chronological':
        # Temporal distribution analysis
        from religious_texts.visualization.timelines import CHRONOLOGICAL_ORDER
        
        # Add chronological data
        chronological_data = []
        
        for _, row in rel_freq.iterrows():
            book = row['book']
            
            if book in CHRONOLOGICAL_ORDER:
                chrono_info = CHRONOLOGICAL_ORDER[book]
                
                chronological_data.append({
                    'book': book,
                    'order': chrono_info['order'],
                    'period': chrono_info['period'],
                    'date': chrono_info['approx_date'],
                    'word_count': row['word_count'],
                    'relative_frequency': row['relative_frequency']
                })
        
        if not chronological_data:
            result['valid'] = False
            result['explanation'] = f"No chronological data available for books containing '{term}'."
            return result
        
        # Create DataFrame
        chrono_df = pd.DataFrame(chronological_data)
        
        # Sort by chronological order
        chrono_df = chrono_df.sort_values('order')
        
        # Calculate correlation
        corr, p_value = stats.spearmanr(chrono_df['order'], chrono_df['relative_frequency'])
        
        # Store correlation data
        result['correlation'] = corr
        result['p_value'] = p_value
        result['significant'] = p_value < 0.05
        
        # Check claim type
        if 'increase' in claim.lower() or 'more prevalent' in claim.lower() or 'more common' in claim.lower():
            # Claim about increasing prevalence over time
            result['valid'] = corr > 0 and p_value < 0.05
            
            result['explanation'] = f"Correlation between chronological order and frequency: {corr:.2f} (p-value: {p_value:.3f}). "
            
            if result['valid']:
                result['explanation'] += f"The data supports the claim that '{term}' increases in prevalence over time."
            else:
                result['explanation'] += f"The data does not support the claim that '{term}' increases in prevalence over time."
        
        elif 'decrease' in claim.lower() or 'less prevalent' in claim.lower() or 'less common' in claim.lower():
            # Claim about decreasing prevalence over time
            result['valid'] = corr < 0 and p_value < 0.05
            
            result['explanation'] = f"Correlation between chronological order and frequency: {corr:.2f} (p-value: {p_value:.3f}). "
            
            if result['valid']:
                result['explanation'] += f"The data supports the claim that '{term}' decreases in prevalence over time."
            else:
                result['explanation'] += f"The data does not support the claim that '{term}' decreases in prevalence over time."
        
        else:
            # Generic claim about temporal distribution
            result['valid'] = None
            
            result['explanation'] = f"Correlation between chronological order and frequency: {corr:.2f} (p-value: {p_value:.3f}). "
            
            if p_value < 0.05:
                if corr > 0:
                    result['explanation'] += f"The term '{term}' significantly increases in prevalence over time."
                else:
                    result['explanation'] += f"The term '{term}' significantly decreases in prevalence over time."
            else:
                result['explanation'] += f"No significant trend in the prevalence of '{term}' over time."
        
        # Add chronological data to result
        result['chronological_data'] = chrono_df.to_dict(orient='records')
    
    elif distribution_type.lower() == 'testament':
        # Testament-based distribution analysis
        # Define testament groupings
        old_testament = [
            'Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy',
            'Joshua', 'Judges', 'Ruth', '1 Samuel', '2 Samuel', '1 Kings', '2 Kings',
            '1 Chronicles', '2 Chronicles', 'Ezra', 'Nehemiah', 'Esther',
            'Job', 'Psalms', 'Proverbs', 'Ecclesiastes', 'Song of Solomon',
            'Isaiah', 'Jeremiah', 'Lamentations', 'Ezekiel', 'Daniel',
            'Hosea', 'Joel', 'Amos', 'Obadiah', 'Jonah', 'Micah', 'Nahum',
            'Habakkuk', 'Zephaniah', 'Haggai', 'Zechariah', 'Malachi'
        ]
        
        new_testament = [
            'Matthew', 'Mark', 'Luke', 'John', 'Acts',
            'Romans', '1 Corinthians', '2 Corinthians', 'Galatians', 'Ephesians',
            'Philippians', 'Colossians', '1 Thessalonians', '2 Thessalonians',
            '1 Timothy', '2 Timothy', 'Titus', 'Philemon', 'Hebrews',
            'James', '1 Peter', '2 Peter', '1 John', '2 John', '3 John', 'Jude',
            'Revelation'
        ]
        
        # Calculate testament frequencies
        ot_books = [book for book in rel_freq['book'] if book in old_testament]
        nt_books = [book for book in rel_freq['book'] if book in new_testament]
        
        ot_freq = rel_freq[rel_freq['book'].isin(ot_books)]
        nt_freq = rel_freq[rel_freq['book'].isin(nt_books)]
        
        # Calculate average frequencies
        if not ot_freq.empty:
            ot_avg_freq = ot_freq['relative_frequency'].mean()
            ot_total_count = ot_freq['word_count'].sum()
        else:
            ot_avg_freq = 0
            ot_total_count = 0
        
        if not nt_freq.empty:
            nt_avg_freq = nt_freq['relative_frequency'].mean()
            nt_total_count = nt_freq['word_count'].sum()
        else:
            nt_avg_freq = 0
            nt_total_count = 0
        
        # Store testament data
        result['testament_data'] = {
            'old_testament': {
                'avg_frequency': ot_avg_freq,
                'total_count': ot_total_count,
                'books': ot_books
            },
            'new_testament': {
                'avg_frequency': nt_avg_freq,
                'total_count': nt_total_count,
                'books': nt_books
            }
        }
        
        # Check statistical significance if enough data
        if len(ot_books) >= 3 and len(nt_books) >= 3:
            ot_values = ot_freq['relative_frequency'].values
            nt_values = nt_freq['relative_frequency'].values
            
            # T-test for difference between testaments
            t_stat, p_value = stats.ttest_ind(ot_values, nt_values, equal_var=False)
            
            result['t_stat'] = t_stat
            result['p_value'] = p_value
            result['significant'] = p_value < 0.05
        
        # Check claim type
        if ('new testament' in claim.lower() and 'more' in claim.lower()) or \
           ('old testament' in claim.lower() and 'less' in claim.lower()):
            # Claim that term is more prevalent in New Testament
            result['valid'] = nt_avg_freq > ot_avg_freq
            
            if 'significant' in result and result['significant']:
                result['valid'] = result['valid'] and result['significant']
            
            result['explanation'] = f"Average frequency in Old Testament: {ot_avg_freq:.2f} per 1000 words. "
            result['explanation'] += f"Average frequency in New Testament: {nt_avg_freq:.2f} per 1000 words. "
            
            if result['valid']:
                result['explanation'] += f"The data supports the claim that '{term}' is more prevalent in the New Testament."
            else:
                result['explanation'] += f"The data does not support the claim that '{term}' is more prevalent in the New Testament."
        
        elif ('old testament' in claim.lower() and 'more' in claim.lower()) or \
             ('new testament' in claim.lower() and 'less' in claim.lower()):
            # Claim that term is more prevalent in Old Testament
            result['valid'] = ot_avg_freq > nt_avg_freq
            
            if 'significant' in result and result['significant']:
                result['valid'] = result['valid'] and result['significant']
            
            result['explanation'] = f"Average frequency in Old Testament: {ot_avg_freq:.2f} per 1000 words. "
            result['explanation'] += f"Average frequency in New Testament: {nt_avg_freq:.2f} per 1000 words. "
            
            if result['valid']:
                result['explanation'] += f"The data supports the claim that '{term}' is more prevalent in the Old Testament."
            else:
                result['explanation'] += f"The data does not support the claim that '{term}' is more prevalent in the Old Testament."
        
        else:
            # Generic claim about testament distribution
            result['valid'] = None
            
            result['explanation'] = f"Average frequency in Old Testament: {ot_avg_freq:.2f} per 1000 words. "
            result['explanation'] += f"Average frequency in New Testament: {nt_avg_freq:.2f} per 1000 words. "
            
            if ot_avg_freq > nt_avg_freq:
                result['explanation'] += f"The term '{term}' appears more frequently in the Old Testament."
            else:
                result['explanation'] += f"The term '{term}' appears more frequently in the New Testament."
            
            if 'significant' in result:
                if result['significant']:
                    result['explanation'] += f" This difference is statistically significant (p-value: {p_value:.3f})."
                else:
                    result['explanation'] += f" However, this difference is not statistically significant (p-value: {p_value:.3f})."
    
    elif distribution_type.lower() == 'genre':
        # Genre-based distribution analysis
        # Define genre groupings
        genres = {
            'Law': ['Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy'],
            'History': ['Joshua', 'Judges', 'Ruth', '1 Samuel', '2 Samuel', '1 Kings', '2 Kings',
                       '1 Chronicles', '2 Chronicles', 'Ezra', 'Nehemiah', 'Esther', 'Acts'],
            'Poetry': ['Job', 'Psalms', 'Proverbs', 'Ecclesiastes', 'Song of Solomon', 'Lamentations'],
            'Prophecy': ['Isaiah', 'Jeremiah', 'Ezekiel', 'Daniel',
                        'Hosea', 'Joel', 'Amos', 'Obadiah', 'Jonah', 'Micah', 'Nahum',
                        'Habakkuk', 'Zephaniah', 'Haggai', 'Zechariah', 'Malachi',
                        'Revelation'],
            'Gospel': ['Matthew', 'Mark', 'Luke', 'John'],
            'Epistle': ['Romans', '1 Corinthians', '2 Corinthians', 'Galatians', 'Ephesians',
                       'Philippians', 'Colossians', '1 Thessalonians', '2 Thessalonians',
                       '1 Timothy', '2 Timothy', 'Titus', 'Philemon', 'Hebrews',
                       'James', '1 Peter', '2 Peter', '1 John', '2 John', '3 John', 'Jude']
        }
        
        # Calculate genre frequencies
        genre_data = {}
        
        for genre, genre_books in genres.items():
            # Filter by books in this genre
            books_in_genre = [book for book in rel_freq['book'] if book in genre_books]
            genre_freq = rel_freq[rel_freq['book'].isin(books_in_genre)]
            
            if not genre_freq.empty:
                avg_freq = genre_freq['relative_frequency'].mean()
                total_count = genre_freq['word_count'].sum()
                
                genre_data[genre] = {
                    'avg_frequency': avg_freq,
                    'total_count': total_count,
                    'books': books_in_genre
                }
        
        # Store genre data
        result['genre_data'] = genre_data
        
        # Find genre with highest frequency
        max_genre = None
        max_freq = 0
        
        for genre, data in genre_data.items():
            if data['avg_frequency'] > max_freq:
                max_freq = data['avg_frequency']
                max_genre = genre
        
        # Check claim type
        claimed_genre = None
        for genre in genres:
            if genre.lower() in claim.lower():
                claimed_genre = genre
                break
        
        if claimed_genre and ('most common' in claim.lower() or 'most prevalent' in claim.lower() or 
                             'highest frequency' in claim.lower()):
            # Claim that term is most prevalent in specific genre
            if claimed_genre in genre_data:
                result['valid'] = claimed_genre == max_genre
                
                result['explanation'] = f"Average frequencies by genre: "
                for genre, data in sorted(genre_data.items(), key=lambda x: x[1]['avg_frequency'], reverse=True):
                    result['explanation'] += f"{genre}: {data['avg_frequency']:.2f}, "
                
                result['explanation'] = result['explanation'].rstrip(', ') + '. '
                
                if result['valid']:
                    result['explanation'] += f"The data supports the claim that '{term}' is most prevalent in {claimed_genre}."
                else:
                    result['explanation'] += f"The data does not support the claim that '{term}' is most prevalent in {claimed_genre}. "
                    result['explanation'] += f"It is most prevalent in {max_genre}."
            else:
                result['valid'] = False
                result['explanation'] = f"No data available for {claimed_genre} books to validate the claim."
        
        else:
            # Generic claim about genre distribution
            result['valid'] = None
            
            result['explanation'] = f"Average frequencies by genre: "
            for genre, data in sorted(genre_data.items(), key=lambda x: x[1]['avg_frequency'], reverse=True):
                result['explanation'] += f"{genre}: {data['avg_frequency']:.2f}, "
            
            result['explanation'] = result['explanation'].rstrip(', ') + '. '
            
            if max_genre:
                result['explanation'] += f"The term '{term}' is most prevalent in {max_genre} literature."
    
    elif distribution_type.lower() == 'author':
        # Author-based distribution analysis
        # Define author groupings (simplified)
        authors = {
            'Moses': ['Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy'],
            'David': ['Psalms'],  # Traditionally attributed to David, though contains works by others
            'Solomon': ['Proverbs', 'Ecclesiastes', 'Song of Solomon'],
            'Isaiah': ['Isaiah'],
            'Jeremiah': ['Jeremiah', 'Lamentations'],
            'Ezekiel': ['Ezekiel'],
            'Daniel': ['Daniel'],
            'Matthew': ['Matthew'],
            'Mark': ['Mark'],
            'Luke': ['Luke', 'Acts'],
            'John': ['John', '1 John', '2 John', '3 John', 'Revelation'],
            'Paul': ['Romans', '1 Corinthians', '2 Corinthians', 'Galatians', 'Ephesians',
                    'Philippians', 'Colossians', '1 Thessalonians', '2 Thessalonians',
                    '1 Timothy', '2 Timothy', 'Titus', 'Philemon'],
            'James': ['James'],
            'Peter': ['1 Peter', '2 Peter'],
            'Jude': ['Jude']
        }
        
        # Calculate author frequencies
        author_data = {}
        
        for author, author_books in authors.items():
            # Filter by books by this author
            books_by_author = [book for book in rel_freq['book'] if book in author_books]
            author_freq = rel_freq[rel_freq['book'].isin(books_by_author)]
            
            if not author_freq.empty:
                avg_freq = author_freq['relative_frequency'].mean()
                total_count = author_freq['word_count'].sum()
                
                author_data[author] = {
                    'avg_frequency': avg_freq,
                    'total_count': total_count,
                    'books': books_by_author
                }
        
        # Store author data
        result['author_data'] = author_data
        
        # Find author with highest frequency
        max_author = None
        max_freq = 0
        
        for author, data in author_data.items():
            if data['avg_frequency'] > max_freq:
                max_freq = data['avg_frequency']
                max_author = author
        
        # Check claim type
        claimed_author = None
        for author in authors:
            if author.lower() in claim.lower():
                claimed_author = author
                break
        
        if claimed_author and ('most common' in claim.lower() or 'most prevalent' in claim.lower() or 
                              'highest frequency' in claim.lower() or 'emphasizes' in claim.lower()):
            # Claim that term is most prevalent in specific author's writings
            if claimed_author in author_data:
                result['valid'] = claimed_author == max_author
                
                result['explanation'] = f"Average frequencies by author: "
                for author, data in sorted(author_data.items(), key=lambda x: x[1]['avg_frequency'], reverse=True):
                    result['explanation'] += f"{author}: {data['avg_frequency']:.2f}, "
                
                result['explanation'] = result['explanation'].rstrip(', ') + '. '
                
                if result['valid']:
                    result['explanation'] += f"The data supports the claim that '{term}' is most prevalent in {claimed_author}'s writings."
                else:
                    result['explanation'] += f"The data does not support the claim that '{term}' is most prevalent in {claimed_author}'s writings. "
                    result['explanation'] += f"It is most prevalent in {max_author}'s writings."
            else:
                result['valid'] = False
                result['explanation'] = f"No data available for {claimed_author}'s writings to validate the claim."
        
        else:
            # Generic claim about author distribution
            result['valid'] = None
            
            result['explanation'] = f"Average frequencies by author: "
            for author, data in sorted(author_data.items(), key=lambda x: x[1]['avg_frequency'], reverse=True):
                result['explanation'] += f"{author}: {data['avg_frequency']:.2f}, "
            
            result['explanation'] = result['explanation'].rstrip(', ') + '. '
            
            if max_author:
                result['explanation'] += f"The term '{term}' is most prevalent in {max_author}'s writings."
    
    else:
        # Unsupported distribution type
        result['valid'] = None
        result['explanation'] = f"Distribution type '{distribution_type}' not supported for validation."
    
    return result
