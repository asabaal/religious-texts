"""
Heatmap Visualization Module

This module provides functions for creating heatmaps to visualize patterns
in biblical texts, including word frequency, concept distribution, and
argument strength assessment.
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def create_heatmap(data: Union[pd.DataFrame, np.ndarray], 
                 row_labels: Optional[List[str]] = None,
                 col_labels: Optional[List[str]] = None,
                 title: str = 'Heatmap',
                 cmap: str = 'YlOrRd',
                 annotate: bool = True,
                 figsize: Tuple[int, int] = (10, 8),
                 filename: Optional[str] = None) -> plt.Figure:
    """
    Create a customizable heatmap visualization.
    
    Args:
        data: Data to visualize (DataFrame or numpy array)
        row_labels: Labels for rows (optional)
        col_labels: Labels for columns (optional)
        title: Title for the heatmap
        cmap: Colormap to use
        annotate: Whether to annotate cells with values
        figsize: Figure size (width, height) in inches
        filename: Optional filename to save the figure
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> # Create a heatmap of word frequency data
        >>> from religious_texts.text_analysis import frequency
        >>> bible = loaders.load_text("kjv.txt")
        >>> divine_names = ["god", "lord", "jesus", "christ"]
        >>> freq_df = frequency.frequency_distribution(bible, divine_names, unit='book')
        >>> fig = create_heatmap(freq_df, title='Divine Name Usage')
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert data to numpy array if it's a DataFrame
    if isinstance(data, pd.DataFrame):
        # Save row and column labels if not provided
        if row_labels is None:
            row_labels = data.index.tolist()
        if col_labels is None:
            col_labels = data.columns.tolist()
        
        # Convert to numpy array
        data = data.values
    
    # Create heatmap
    if annotate:
        sns.heatmap(data, annot=True, fmt=".2f", linewidths=.5, ax=ax, cmap=cmap)
    else:
        sns.heatmap(data, annot=False, linewidths=.5, ax=ax, cmap=cmap)
    
    # Set labels if provided
    if row_labels:
        ax.set_yticks(np.arange(len(row_labels)) + 0.5)
        ax.set_yticklabels(row_labels, rotation=0)
    
    if col_labels:
        ax.set_xticks(np.arange(len(col_labels)) + 0.5)
        ax.set_xticklabels(col_labels, rotation=45, ha='right')
    
    # Set title
    ax.set_title(title, fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig


def create_book_heatmap(bible_dict: Dict[str, Any], 
                      terms: List[str],
                      normalize: bool = True,
                      books: Optional[List[str]] = None,
                      title: str = 'Term Frequency by Book',
                      figsize: Tuple[int, int] = (12, 10),
                      filename: Optional[str] = None) -> plt.Figure:
    """
    Create a heatmap showing term frequency across books.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        terms: List of terms to analyze
        normalize: Whether to normalize frequencies by book length
        books: Optional list of specific books to include
        title: Title for the heatmap
        figsize: Figure size (width, height) in inches
        filename: Optional filename to save the figure
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Create heatmap of divine names across the Gospels
        >>> divine_names = ["god", "lord", "jesus", "christ", "spirit"]
        >>> fig = create_book_heatmap(bible, divine_names, 
        ...                         books=["Matthew", "Mark", "Luke", "John"])
    """
    from religious_texts.text_analysis import frequency
    
    # Get term frequency distribution
    freq_df = frequency.frequency_distribution(bible_dict, terms, unit='book', 
                                             normalize=normalize, books=books)
    
    if freq_df.empty:
        # Create empty figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Prepare data for heatmap
    books = freq_df.index.tolist()
    
    # Select relevant columns
    if normalize:
        # Use normalized frequencies
        term_cols = []
        for term in terms:
            if f"relative_frequency_{term}" in freq_df.columns:
                term_cols.append(f"relative_frequency_{term}")
            else:
                term_cols.append(term)
        
        heatmap_data = freq_df[term_cols]
        
        # Rename columns to remove prefix
        heatmap_data.columns = [col.replace('relative_frequency_', '') for col in heatmap_data.columns]
    else:
        # Use raw frequencies
        heatmap_data = freq_df[terms]
    
    # Create the heatmap
    fig = create_heatmap(
        heatmap_data, 
        row_labels=books,
        col_labels=terms,
        title=title,
        figsize=figsize,
        filename=filename
    )
    
    return fig


def create_term_heatmap(bible_dict: Dict[str, Any], 
                      term: str,
                      books: Optional[List[str]] = None,
                      normalize: bool = True,
                      num_books: int = 20,
                      title: Optional[str] = None,
                      figsize: Tuple[int, int] = (12, 10),
                      filename: Optional[str] = None) -> plt.Figure:
    """
    Create a heatmap showing the distribution of a single term across books and chapters.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        term: Term to analyze
        books: Optional list of specific books to include
        normalize: Whether to normalize frequencies by chapter length
        num_books: Number of top books to include (if books not specified)
        title: Title for the heatmap (defaults to "Distribution of '[term]'")
        figsize: Figure size (width, height) in inches
        filename: Optional filename to save the figure
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Create heatmap of the term "love" distribution
        >>> fig = create_term_heatmap(bible, "love")
    """
    from religious_texts.text_analysis import frequency
    
    # Set default title if not provided
    if title is None:
        title = f"Distribution of '{term}'"
    
    # Get term frequency by book
    book_freq = frequency.relative_frequency(bible_dict, term, books=books)
    
    if book_freq.empty:
        # Create empty figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"No occurrences of '{term}' found", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Sort by relative frequency and get top books
    book_freq = book_freq.sort_values('relative_frequency', ascending=False)
    
    if books is None:
        # Limit to top N books
        top_books = book_freq.head(num_books)['book'].tolist()
    else:
        top_books = books
    
    # Initialize matrix for heatmap
    # Find max chapter number across selected books
    max_chapter = 0
    for book in top_books:
        if book in bible_dict:
            max_chapter = max(max_chapter, max(bible_dict[book].keys()))
    
    # Create empty matrix
    matrix = np.zeros((len(top_books), max_chapter))
    
    # Fill matrix with term frequencies by chapter
    for i, book in enumerate(top_books):
        if book not in bible_dict:
            continue
            
        for chapter_num in range(1, max_chapter + 1):
            if chapter_num not in bible_dict[book]:
                continue
                
            # Extract chapter text
            chapter_text = " ".join(bible_dict[book][chapter_num].values())
            
            # Count term occurrences
            term_count = chapter_text.lower().count(term.lower())
            
            if normalize:
                # Normalize by chapter length (words)
                word_count = len(chapter_text.split())
                if word_count > 0:
                    matrix[i, chapter_num - 1] = term_count / word_count * 1000
            else:
                matrix[i, chapter_num - 1] = term_count
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(matrix, cmap='YlOrRd')
    
    # Set labels
    ax.set_xticks(np.arange(max_chapter))
    ax.set_xticklabels(np.arange(1, max_chapter + 1))
    ax.set_yticks(np.arange(len(top_books)))
    ax.set_yticklabels(top_books)
    
    # Rotate x-axis labels if there are many
    if max_chapter > 20:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    label = f"Frequency per 1000 words" if normalize else "Count"
    cbar.ax.set_ylabel(label, rotation=-90, va="bottom")
    
    # Set title
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Chapter", fontsize=12)
    ax.set_ylabel("Book", fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig


def create_concept_heatmap(bible_dict: Dict[str, Any],
                         concepts: Dict[str, List[str]],
                         books: Optional[List[str]] = None,
                         normalize: bool = True,
                         title: str = 'Concept Distribution',
                         figsize: Tuple[int, int] = (12, 10),
                         filename: Optional[str] = None) -> plt.Figure:
    """
    Create a heatmap showing concept distribution across books.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        concepts: Dictionary mapping concept names to lists of related terms
        books: Optional list of specific books to include
        normalize: Whether to normalize frequencies by book length
        title: Title for the heatmap
        figsize: Figure size (width, height) in inches
        filename: Optional filename to save the figure
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Create heatmap of theological concepts
        >>> concepts = {
        ...     "salvation": ["save", "salvation", "redeem", "deliver"],
        ...     "judgment": ["judge", "judgment", "wrath", "punishment"],
        ...     "love": ["love", "charity", "compassion", "kindness"],
        ...     "faith": ["faith", "believe", "trust", "confidence"]
        ... }
        >>> fig = create_concept_heatmap(bible, concepts)
    """
    # Initialize data structure
    data = {}
    
    # Determine which books to include
    if books:
        books_to_check = [book for book in books if book in bible_dict]
    else:
        books_to_check = list(bible_dict.keys())
    
    # Skip if no books available
    if not books_to_check:
        # Create empty figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No book data available", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Process each book
    for book in books_to_check:
        # Combine all text in the book
        book_text = " ".join(
            verse_text 
            for chapter_verses in bible_dict[book].values() 
            for verse_text in chapter_verses.values()
        )
        
        # Convert to lowercase for case-insensitive matching
        book_text_lower = book_text.lower()
        
        # Count total words for normalization
        total_words = len(book_text.split())
        
        # Count concept term occurrences
        concept_counts = {}
        
        for concept_name, terms in concepts.items():
            # Initialize count
            count = 0
            
            # Count occurrences of each term in the concept
            for term in terms:
                # Use word boundary regex for more accurate counting
                pattern = r'\b' + re.escape(term.lower()) + r'\b'
                matches = re.findall(pattern, book_text_lower)
                count += len(matches)
            
            # Store the count
            if normalize and total_words > 0:
                concept_counts[concept_name] = count / total_words * 1000
            else:
                concept_counts[concept_name] = count
        
        # Add to data structure
        data[book] = concept_counts
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')
    
    # Fill NaN values with 0
    df = df.fillna(0)
    
    # Create the heatmap
    fig = create_heatmap(
        df,
        title=title,
        figsize=figsize,
        filename=filename
    )
    
    return fig


def create_chapter_heatmap(bible_dict: Dict[str, Any],
                         book: str,
                         terms: List[str],
                         normalize: bool = True,
                         title: Optional[str] = None,
                         figsize: Tuple[int, int] = (12, 10),
                         filename: Optional[str] = None) -> plt.Figure:
    """
    Create a heatmap showing term distribution across chapters of a single book.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        book: Book to analyze
        terms: List of terms to analyze
        normalize: Whether to normalize frequencies by chapter length
        title: Title for the heatmap (defaults to "[Book] Term Distribution by Chapter")
        figsize: Figure size (width, height) in inches
        filename: Optional filename to save the figure
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Create heatmap of key terms in Romans
        >>> theological_terms = ["faith", "grace", "law", "sin", "righteousness"]
        >>> fig = create_chapter_heatmap(bible, "Romans", theological_terms)
    """
    # Check if book exists
    if book not in bible_dict:
        # Create empty figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Book '{book}' not found", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Set default title if not provided
    if title is None:
        title = f"{book} Term Distribution by Chapter"
    
    # Get chapter numbers
    chapters = sorted(bible_dict[book].keys())
    
    # Initialize data structure
    data = {}
    
    # Process each chapter
    for chapter_num in chapters:
        # Combine all verses in the chapter
        chapter_text = " ".join(bible_dict[book][chapter_num].values())
        
        # Convert to lowercase for case-insensitive matching
        chapter_text_lower = chapter_text.lower()
        
        # Count total words for normalization
        total_words = len(chapter_text.split())
        
        # Count term occurrences
        term_counts = {}
        
        for term in terms:
            # Use word boundary regex for more accurate counting
            pattern = r'\b' + re.escape(term.lower()) + r'\b'
            matches = re.findall(pattern, chapter_text_lower)
            count = len(matches)
            
            # Store the count
            if normalize and total_words > 0:
                term_counts[term] = count / total_words * 1000
            else:
                term_counts[term] = count
        
        # Add to data structure
        data[f"Chapter {chapter_num}"] = term_counts
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')
    
    # Fill NaN values with 0
    df = df.fillna(0)
    
    # Create the heatmap
    fig = create_heatmap(
        df,
        title=title,
        figsize=figsize,
        filename=filename
    )
    
    return fig
