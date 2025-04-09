"""
Parallel Passage Analysis Module

This module provides functions for identifying and analyzing parallel passages
in biblical texts, particularly focusing on the synoptic gospels and other
texts with parallel narrative accounts.
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from difflib import SequenceMatcher

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize

# Known parallel passages in the Synoptic Gospels
# This is a simplified set of key parallels 
SYNOPTIC_PARALLELS = [
    {
        'name': 'John the Baptist',
        'passages': {
            'Matthew': {'chapters': [3], 'verses': [1, 12]},
            'Mark': {'chapters': [1], 'verses': [1, 8]},
            'Luke': {'chapters': [3], 'verses': [1, 20]}
        }
    },
    {
        'name': 'Baptism of Jesus',
        'passages': {
            'Matthew': {'chapters': [3], 'verses': [13, 17]},
            'Mark': {'chapters': [1], 'verses': [9, 11]},
            'Luke': {'chapters': [3], 'verses': [21, 22]}
        }
    },
    {
        'name': 'Temptation of Jesus',
        'passages': {
            'Matthew': {'chapters': [4], 'verses': [1, 11]},
            'Mark': {'chapters': [1], 'verses': [12, 13]},
            'Luke': {'chapters': [4], 'verses': [1, 13]}
        }
    },
    {
        'name': 'Sermon on the Mount/Plain',
        'passages': {
            'Matthew': {'chapters': [5, 6, 7], 'verses': [1, 29]},
            'Luke': {'chapters': [6], 'verses': [17, 49]}
        }
    },
    {
        'name': 'Calming the Storm',
        'passages': {
            'Matthew': {'chapters': [8], 'verses': [23, 27]},
            'Mark': {'chapters': [4], 'verses': [35, 41]},
            'Luke': {'chapters': [8], 'verses': [22, 25]}
        }
    },
    {
        'name': 'Feeding the Five Thousand',
        'passages': {
            'Matthew': {'chapters': [14], 'verses': [13, 21]},
            'Mark': {'chapters': [6], 'verses': [30, 44]},
            'Luke': {'chapters': [9], 'verses': [10, 17]},
            'John': {'chapters': [6], 'verses': [1, 14]}
        }
    },
    {
        'name': 'Peter\'s Confession',
        'passages': {
            'Matthew': {'chapters': [16], 'verses': [13, 20]},
            'Mark': {'chapters': [8], 'verses': [27, 30]},
            'Luke': {'chapters': [9], 'verses': [18, 21]}
        }
    },
    {
        'name': 'Transfiguration',
        'passages': {
            'Matthew': {'chapters': [17], 'verses': [1, 9]},
            'Mark': {'chapters': [9], 'verses': [2, 10]},
            'Luke': {'chapters': [9], 'verses': [28, 36]}
        }
    },
    {
        'name': 'Triumphal Entry',
        'passages': {
            'Matthew': {'chapters': [21], 'verses': [1, 11]},
            'Mark': {'chapters': [11], 'verses': [1, 11]},
            'Luke': {'chapters': [19], 'verses': [28, 40]},
            'John': {'chapters': [12], 'verses': [12, 19]}
        }
    },
    {
        'name': 'Last Supper',
        'passages': {
            'Matthew': {'chapters': [26], 'verses': [17, 35]},
            'Mark': {'chapters': [14], 'verses': [12, 31]},
            'Luke': {'chapters': [22], 'verses': [7, 38]},
            'John': {'chapters': [13], 'verses': [1, 30]}
        }
    },
    {
        'name': 'Arrest',
        'passages': {
            'Matthew': {'chapters': [26], 'verses': [47, 56]},
            'Mark': {'chapters': [14], 'verses': [43, 52]},
            'Luke': {'chapters': [22], 'verses': [47, 53]},
            'John': {'chapters': [18], 'verses': [1, 12]}
        }
    },
    {
        'name': 'Crucifixion',
        'passages': {
            'Matthew': {'chapters': [27], 'verses': [32, 56]},
            'Mark': {'chapters': [15], 'verses': [21, 41]},
            'Luke': {'chapters': [23], 'verses': [26, 49]},
            'John': {'chapters': [19], 'verses': [17, 37]}
        }
    },
    {
        'name': 'Resurrection',
        'passages': {
            'Matthew': {'chapters': [28], 'verses': [1, 10]},
            'Mark': {'chapters': [16], 'verses': [1, 8]},
            'Luke': {'chapters': [24], 'verses': [1, 12]},
            'John': {'chapters': [20], 'verses': [1, 10]}
        }
    }
]

# Other parallel passages in the Old Testament
OLD_TESTAMENT_PARALLELS = [
    {
        'name': 'Creation',
        'passages': {
            'Genesis': {'chapters': [1, 2], 'verses': [1, 3]}
        }
    },
    {
        'name': 'Law Repetition',
        'passages': {
            'Exodus': {'chapters': [20], 'verses': [1, 17]},
            'Deuteronomy': {'chapters': [5], 'verses': [6, 21]}
        }
    },
    {
        'name': 'David and Goliath',
        'passages': {
            '1 Samuel': {'chapters': [17], 'verses': [1, 58]}
        }
    },
    {
        'name': 'Chronicles-Kings Parallels',
        'passages': {
            '1 Kings': {'chapters': [1, 2], 'verses': [1, 46]},
            '1 Chronicles': {'chapters': [28, 29], 'verses': [1, 30]}
        }
    }
]


def _extract_passage_text(bible_dict: Dict[str, Any], book: str, 
                        chapters: List[int], verses: List[int]) -> str:
    """
    Helper function to extract text from a passage based on book, chapters, and verse range.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        book: Book name
        chapters: List of chapters to include
        verses: List [start_verse, end_verse] for verse range
        
    Returns:
        String containing the combined text of the passage
    """
    if book not in bible_dict:
        return ""
    
    start_verse, end_verse = verses
    
    passage_text = []
    
    for chapter_num in chapters:
        if chapter_num not in bible_dict[book]:
            continue
        
        for verse_num, verse_text in bible_dict[book][chapter_num].items():
            # For first chapter, start from start_verse
            if chapter_num == chapters[0] and verse_num < start_verse:
                continue
            
            # For last chapter, end at end_verse
            if chapter_num == chapters[-1] and verse_num > end_verse:
                continue
            
            passage_text.append(verse_text)
    
    return " ".join(passage_text)


def identify_parallel_passages(bible_dict: Dict[str, Any], 
                             source_book: Optional[str] = None,
                             include_known_parallels: bool = True,
                             similarity_threshold: float = 0.5) -> pd.DataFrame:
    """
    Identify parallel passages in biblical texts.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        source_book: Optional book to use as the source for finding parallels
        include_known_parallels: Whether to include known parallel passages
        similarity_threshold: Minimum similarity score for automated parallel detection
        
    Returns:
        DataFrame with identified parallel passages
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Find parallels in the Gospels
        >>> parallels = identify_parallel_passages(bible, source_book="Matthew")
    """
    results = []
    
    # Include known parallel passages if requested
    if include_known_parallels:
        # Add synoptic parallels
        for parallel in SYNOPTIC_PARALLELS:
            parallel_name = parallel['name']
            passages = []
            
            for book, passage_info in parallel['passages'].items():
                # Skip if source_book is specified and this passage doesn't include it
                if source_book and book != source_book and not any(b == source_book for b in parallel['passages']):
                    continue
                
                # Extract passage text
                chapters = passage_info['chapters']
                verses = passage_info['verses']
                passage_text = _extract_passage_text(bible_dict, book, chapters, verses)
                
                if passage_text:
                    passage = {
                        'book': book,
                        'chapters': chapters,
                        'start_verse': verses[0],
                        'end_verse': verses[1],
                        'reference': f"{book} {chapters[0]}:{verses[0]}-{chapters[-1]}:{verses[1]}",
                        'text': passage_text[:100] + '...' if len(passage_text) > 100 else passage_text,
                        'word_count': len(word_tokenize(passage_text))
                    }
                    passages.append(passage)
            
            # Add to results if we have at least two passages
            if len(passages) >= 2:
                for i, passage1 in enumerate(passages):
                    for passage2 in passages[i+1:]:
                        # Calculate similarity score
                        text1 = passage1['text']
                        text2 = passage2['text']
                        
                        # Use SequenceMatcher for similarity
                        similarity = SequenceMatcher(None, text1, text2).ratio()
                        
                        result = {
                            'parallel_name': parallel_name,
                            'book1': passage1['book'],
                            'reference1': passage1['reference'],
                            'book2': passage2['book'],
                            'reference2': passage2['reference'],
                            'similarity_score': similarity,
                            'source': 'known_parallel',
                            'text1': passage1['text'],
                            'text2': passage2['text'],
                            'word_count1': passage1['word_count'],
                            'word_count2': passage2['word_count']
                        }
                        
                        results.append(result)
        
        # Add Old Testament parallels
        for parallel in OLD_TESTAMENT_PARALLELS:
            parallel_name = parallel['name']
            passages = []
            
            for book, passage_info in parallel['passages'].items():
                # Skip if source_book is specified and this passage doesn't include it
                if source_book and book != source_book and not any(b == source_book for b in parallel['passages']):
                    continue
                
                # Extract passage text
                chapters = passage_info['chapters']
                verses = passage_info['verses']
                passage_text = _extract_passage_text(bible_dict, book, chapters, verses)
                
                if passage_text:
                    passage = {
                        'book': book,
                        'chapters': chapters,
                        'start_verse': verses[0],
                        'end_verse': verses[1],
                        'reference': f"{book} {chapters[0]}:{verses[0]}-{chapters[-1]}:{verses[1]}",
                        'text': passage_text[:100] + '...' if len(passage_text) > 100 else passage_text,
                        'word_count': len(word_tokenize(passage_text))
                    }
                    passages.append(passage)
            
            # Add to results if we have at least two passages
            if len(passages) >= 2:
                for i, passage1 in enumerate(passages):
                    for passage2 in passages[i+1:]:
                        # Calculate similarity score
                        text1 = passage1['text']
                        text2 = passage2['text']
                        
                        # Use SequenceMatcher for similarity
                        similarity = SequenceMatcher(None, text1, text2).ratio()
                        
                        result = {
                            'parallel_name': parallel_name,
                            'book1': passage1['book'],
                            'reference1': passage1['reference'],
                            'book2': passage2['book'],
                            'reference2': passage2['reference'],
                            'similarity_score': similarity,
                            'source': 'known_parallel',
                            'text1': passage1['text'],
                            'text2': passage2['text'],
                            'word_count1': passage1['word_count'],
                            'word_count2': passage2['word_count']
                        }
                        
                        results.append(result)
    
    # Automated parallel detection is more complex and resource-intensive
    # This is a simplified implementation for illustration
    if source_book:
        # Only find parallels with passages in the source book
        if source_book not in bible_dict:
            return pd.DataFrame()
        
        # Sample a few chapters for comparison (for performance reasons)
        sample_chapters = list(bible_dict[source_book].keys())[:5]  # Limit to first 5 chapters
        
        for chapter_num in sample_chapters:
            # Get all verses in this chapter
            chapter_text = " ".join(bible_dict[source_book][chapter_num].values())
            
            # Compare with other books
            for other_book, other_chapters in bible_dict.items():
                # Skip self-comparison
                if other_book == source_book:
                    continue
                
                for other_chapter_num, other_verses in other_chapters.items():
                    # Get all verses in this chapter
                    other_chapter_text = " ".join(other_verses.values())
                    
                    # Calculate similarity
                    similarity = SequenceMatcher(None, chapter_text, other_chapter_text).ratio()
                    
                    # Only include if similarity is above threshold
                    if similarity >= similarity_threshold:
                        result = {
                            'parallel_name': f"Potential parallel: {source_book} {chapter_num} - {other_book} {other_chapter_num}",
                            'book1': source_book,
                            'reference1': f"{source_book} {chapter_num}",
                            'book2': other_book,
                            'reference2': f"{other_book} {other_chapter_num}",
                            'similarity_score': similarity,
                            'source': 'automated_detection',
                            'text1': chapter_text[:100] + '...' if len(chapter_text) > 100 else chapter_text,
                            'text2': other_chapter_text[:100] + '...' if len(other_chapter_text) > 100 else other_chapter_text,
                            'word_count1': len(word_tokenize(chapter_text)),
                            'word_count2': len(word_tokenize(other_chapter_text))
                        }
                        
                        results.append(result)
    
    # Convert results to DataFrame
    if results:
        df = pd.DataFrame(results)
        
        # Sort by similarity score (descending)
        df = df.sort_values('similarity_score', ascending=False)
    else:
        # Create empty DataFrame with expected columns
        df = pd.DataFrame(columns=['parallel_name', 'book1', 'reference1', 'book2', 'reference2',
                                 'similarity_score', 'source', 'text1', 'text2', 
                                 'word_count1', 'word_count2'])
    
    return df


def compare_parallel_passages(bible_dict: Dict[str, Any], 
                            passage1: Dict[str, Any],
                            passage2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two parallel passages to identify similarities and differences.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        passage1: Dict with book, chapters, and verses for first passage
        passage2: Dict with book, chapters, and verses for second passage
        
    Returns:
        Dictionary with comparison results
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Compare the Sermon on the Mount/Plain
        >>> passage1 = {
        ...     'book': 'Matthew',
        ...     'chapters': [5, 6, 7],
        ...     'verses': [1, 29]
        ... }
        >>> passage2 = {
        ...     'book': 'Luke',
        ...     'chapters': [6],
        ...     'verses': [17, 49]
        ... }
        >>> comparison = compare_parallel_passages(bible, passage1, passage2)
    """
    # Extract text from both passages
    text1 = _extract_passage_text(
        bible_dict, 
        passage1['book'], 
        passage1['chapters'], 
        passage1['verses']
    )
    
    text2 = _extract_passage_text(
        bible_dict, 
        passage2['book'], 
        passage2['chapters'], 
        passage2['verses']
    )
    
    if not text1 or not text2:
        return {
            'error': 'One or both passages not found',
            'text1_found': bool(text1),
            'text2_found': bool(text2)
        }
    
    # Tokenize text for analysis
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)
    
    # Calculate basic statistics
    word_count1 = len(tokens1)
    word_count2 = len(tokens2)
    unique_words1 = len(set(tokens1))
    unique_words2 = len(set(tokens2))
    
    # Calculate vocabulary overlap
    vocab1 = set(tokens1)
    vocab2 = set(tokens2)
    shared_vocab = vocab1.intersection(vocab2)
    unique_to_1 = vocab1 - vocab2
    unique_to_2 = vocab2 - vocab1
    
    # Calculate overall similarity with SequenceMatcher
    similarity = SequenceMatcher(None, text1, text2).ratio()
    
    # Identify key shared phrases
    shared_phrases = []
    shared_word_count = 0
    
    # This is a simplified approach - a more sophisticated analysis would use 
    # techniques like longest common subsequence
    sentences1 = sent_tokenize(text1)
    sentences2 = sent_tokenize(text2)
    
    for sent1 in sentences1:
        words1 = word_tokenize(sent1)
        for sent2 in sentences2:
            words2 = word_tokenize(sent2)
            
            # Check for similar sentences
            sent_similarity = SequenceMatcher(None, sent1, sent2).ratio()
            
            if sent_similarity > 0.5:
                shared_phrases.append({
                    'text1': sent1,
                    'text2': sent2,
                    'similarity': sent_similarity
                })
                
                # Count approximately how many words are shared
                shared_word_count += min(len(words1), len(words2))
    
    # Compile results
    comparison = {
        'passage1': {
            'book': passage1['book'],
            'reference': f"{passage1['book']} {passage1['chapters'][0]}:{passage1['verses'][0]}-{passage1['chapters'][-1]}:{passage1['verses'][1]}",
            'word_count': word_count1,
            'unique_words': unique_words1,
            'text_sample': text1[:200] + '...' if len(text1) > 200 else text1
        },
        'passage2': {
            'book': passage2['book'],
            'reference': f"{passage2['book']} {passage2['chapters'][0]}:{passage2['verses'][0]}-{passage2['chapters'][-1]}:{passage2['verses'][1]}",
            'word_count': word_count2,
            'unique_words': unique_words2,
            'text_sample': text2[:200] + '...' if len(text2) > 200 else text2
        },
        'comparison': {
            'overall_similarity': similarity,
            'shared_vocab_count': len(shared_vocab),
            'unique_to_1_count': len(unique_to_1),
            'unique_to_2_count': len(unique_to_2),
            'shared_phrases_count': len(shared_phrases),
            'approx_shared_words': shared_word_count,
            'shared_vocabulary_pct1': len(shared_vocab) / unique_words1 if unique_words1 > 0 else 0,
            'shared_vocabulary_pct2': len(shared_vocab) / unique_words2 if unique_words2 > 0 else 0,
            'shared_content_pct1': shared_word_count / word_count1 if word_count1 > 0 else 0,
            'shared_content_pct2': shared_word_count / word_count2 if word_count2 > 0 else 0
        },
        'shared_phrases': shared_phrases[:10]  # Limit to top 10 for readability
    }
    
    return comparison


def analyze_narrative_differences(bible_dict: Dict[str, Any],
                                parallel_passages: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Analyze narrative differences between parallel passages.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        parallel_passages: List of dicts with parallel passage info
        
    Returns:
        DataFrame with narrative difference analysis
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Analyze differences in the Resurrection accounts
        >>> resurrection_parallels = [
        ...     {'book': 'Matthew', 'chapters': [28], 'verses': [1, 10]},
        ...     {'book': 'Mark', 'chapters': [16], 'verses': [1, 8]},
        ...     {'book': 'Luke', 'chapters': [24], 'verses': [1, 12]},
        ...     {'book': 'John', 'chapters': [20], 'verses': [1, 10]}
        ... ]
        >>> diff_analysis = analyze_narrative_differences(bible, resurrection_parallels)
    """
    if len(parallel_passages) < 2:
        return pd.DataFrame()
    
    results = []
    
    # Extract passage texts
    passages = []
    for passage_info in parallel_passages:
        book = passage_info['book']
        chapters = passage_info['chapters']
        verses = passage_info['verses']
        
        passage_text = _extract_passage_text(bible_dict, book, chapters, verses)
        
        if passage_text:
            passage = {
                'book': book,
                'reference': f"{book} {chapters[0]}:{verses[0]}-{chapters[-1]}:{verses[1]}",
                'text': passage_text,
                'sentences': sent_tokenize(passage_text),
                'words': word_tokenize(passage_text)
            }
            passages.append(passage)
    
    if len(passages) < 2:
        return pd.DataFrame()
    
    # Analyze each pair of passages
    for i, passage1 in enumerate(passages):
        for j, passage2 in enumerate(passages[i+1:], i+1):
            # Compare sentences to find potential differences
            sentence_comparisons = []
            
            # For each sentence in passage1, find the most similar sentence in passage2
            for sent1 in passage1['sentences']:
                best_match = None
                best_similarity = 0
                
                for sent2 in passage2['sentences']:
                    similarity = SequenceMatcher(None, sent1, sent2).ratio()
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = sent2
                
                # If best match is not very similar, it could be unique to passage1
                if best_similarity < 0.4:
                    sentence_comparisons.append({
                        'sentence1': sent1,
                        'sentence2': None,
                        'similarity': 0,
                        'difference_type': 'unique_to_1'
                    })
                else:
                    sentence_comparisons.append({
                        'sentence1': sent1,
                        'sentence2': best_match,
                        'similarity': best_similarity,
                        'difference_type': 'shared' if best_similarity > 0.7 else 'variant'
                    })
            
            # Find sentences unique to passage2
            matched_sentences2 = set()
            for comp in sentence_comparisons:
                if comp['sentence2']:
                    matched_sentences2.add(comp['sentence2'])
            
            for sent2 in passage2['sentences']:
                if sent2 not in matched_sentences2:
                    sentence_comparisons.append({
                        'sentence1': None,
                        'sentence2': sent2,
                        'similarity': 0,
                        'difference_type': 'unique_to_2'
                    })
            
            # Count different types of differences
            unique_to_1 = sum(1 for comp in sentence_comparisons if comp['difference_type'] == 'unique_to_1')
            unique_to_2 = sum(1 for comp in sentence_comparisons if comp['difference_type'] == 'unique_to_2')
            variants = sum(1 for comp in sentence_comparisons if comp['difference_type'] == 'variant')
            shared = sum(1 for comp in sentence_comparisons if comp['difference_type'] == 'shared')
            
            # Extract some examples of each difference type
            examples = {
                'unique_to_1': [comp['sentence1'] for comp in sentence_comparisons 
                               if comp['difference_type'] == 'unique_to_1'][:3],
                'unique_to_2': [comp['sentence2'] for comp in sentence_comparisons 
                               if comp['difference_type'] == 'unique_to_2'][:3],
                'variant': [(comp['sentence1'], comp['sentence2']) for comp in sentence_comparisons 
                          if comp['difference_type'] == 'variant'][:3]
            }
            
            # Add result
            result = {
                'book1': passage1['book'],
                'reference1': passage1['reference'],
                'book2': passage2['book'],
                'reference2': passage2['reference'],
                'unique_to_1_count': unique_to_1,
                'unique_to_2_count': unique_to_2,
                'variant_count': variants,
                'shared_count': shared,
                'total_sentence_count': len(sentence_comparisons),
                'unique_content_pct': (unique_to_1 + unique_to_2) / len(sentence_comparisons) if sentence_comparisons else 0,
                'variant_content_pct': variants / len(sentence_comparisons) if sentence_comparisons else 0,
                'shared_content_pct': shared / len(sentence_comparisons) if sentence_comparisons else 0,
                'unique_to_1_examples': '; '.join(examples['unique_to_1']),
                'unique_to_2_examples': '; '.join(examples['unique_to_2']),
                'variant_examples': '; '.join([f"{v[0]} vs. {v[1]}" for v in examples['variant']])
            }
            
            results.append(result)
    
    # Convert results to DataFrame
    if results:
        df = pd.DataFrame(results)
    else:
        # Create empty DataFrame with expected columns
        df = pd.DataFrame(columns=['book1', 'reference1', 'book2', 'reference2',
                                 'unique_to_1_count', 'unique_to_2_count', 'variant_count',
                                 'shared_count', 'total_sentence_count', 'unique_content_pct',
                                 'variant_content_pct', 'shared_content_pct',
                                 'unique_to_1_examples', 'unique_to_2_examples', 'variant_examples'])
    
    return df


def synoptic_analysis(bible_dict: Dict[str, Any], 
                     parallel_name: Optional[str] = None) -> pd.DataFrame:
    """
    Perform synoptic analysis on the Gospels (Matthew, Mark, Luke).
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        parallel_name: Optional name of specific parallel to analyze
        
    Returns:
        DataFrame with synoptic analysis results
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Analyze all synoptic parallels
        >>> synoptic_df = synoptic_analysis(bible)
        >>> # Or analyze a specific parallel
        >>> baptism_df = synoptic_analysis(bible, parallel_name="Baptism of Jesus")
    """
    results = []
    
    # Filter parallels by name if specified
    if parallel_name:
        parallels = [p for p in SYNOPTIC_PARALLELS if p['name'] == parallel_name]
        if not parallels:
            return pd.DataFrame()
    else:
        parallels = SYNOPTIC_PARALLELS
    
    for parallel in parallels:
        parallel_name = parallel['name']
        passages = {}
        
        # Extract passages
        for book, passage_info in parallel['passages'].items():
            # Skip if not a Synoptic Gospel (we include John in SYNOPTIC_PARALLELS but not in synoptic analysis)
            if book not in ['Matthew', 'Mark', 'Luke']:
                continue
                
            # Extract passage text
            chapters = passage_info['chapters']
            verses = passage_info['verses']
            passage_text = _extract_passage_text(bible_dict, book, chapters, verses)
            
            if passage_text:
                passages[book] = {
                    'text': passage_text,
                    'reference': f"{book} {chapters[0]}:{verses[0]}-{chapters[-1]}:{verses[1]}",
                    'word_count': len(word_tokenize(passage_text))
                }
        
        # Skip if we don't have at least two Synoptic Gospels
        if len(passages) < 2:
            continue
        
        # Pairwise comparisons
        pairs = [
            ('Matthew', 'Mark'),
            ('Matthew', 'Luke'),
            ('Mark', 'Luke')
        ]
        
        for book1, book2 in pairs:
            if book1 not in passages or book2 not in passages:
                continue
                
            # Calculate similarity
            text1 = passages[book1]['text']
            text2 = passages[book2]['text']
            similarity = SequenceMatcher(None, text1, text2).ratio()
            
            # Calculate Triple Tradition (MT-MK-LK shared material)
            triple_tradition = False
            if 'Matthew' in passages and 'Mark' in passages and 'Luke' in passages:
                triple_tradition = True
            
            # Calculate Double Tradition
            double_tradition = not triple_tradition and len(passages) >= 2
            
            # Mark-Matthew Relationship
            mark_priority = False
            if book1 == 'Mark' and book2 == 'Matthew' and len(text1) < len(text2):
                mark_priority = True
            elif book1 == 'Matthew' and book2 == 'Mark' and len(text2) < len(text1):
                mark_priority = True
            
            # Add result
            result = {
                'parallel_name': parallel_name,
                'book1': book1,
                'reference1': passages[book1]['reference'],
                'book2': book2,
                'reference2': passages[book2]['reference'],
                'similarity_score': similarity,
                'word_count1': passages[book1]['word_count'],
                'word_count2': passages[book2]['word_count'],
                'triple_tradition': triple_tradition,
                'double_tradition': double_tradition,
                'mark_priority_evidence': mark_priority
            }
            
            results.append(result)
    
    # Convert results to DataFrame
    if results:
        df = pd.DataFrame(results)
        
        # Sort by similarity score (descending)
        df = df.sort_values(['parallel_name', 'similarity_score'], ascending=[True, False])
    else:
        # Create empty DataFrame with expected columns
        df = pd.DataFrame(columns=['parallel_name', 'book1', 'reference1', 'book2', 'reference2',
                                 'similarity_score', 'word_count1', 'word_count2',
                                 'triple_tradition', 'double_tradition', 'mark_priority_evidence'])
    
    return df
