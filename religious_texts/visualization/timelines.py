"""
Timeline Visualization Module

This module provides functions for creating timeline visualizations of biblical
texts, including chronological order, term usage over time, and narrative flow.
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


# Approximate chronological order of biblical books
# This is a simplified ordering based on traditional dating
CHRONOLOGICAL_ORDER = {
    # Torah (Pentateuch)
    'Genesis': {'order': 1, 'period': 'Torah', 'approx_date': 'Creation-1500 BCE'},
    'Exodus': {'order': 2, 'period': 'Torah', 'approx_date': '1500-1400 BCE'},
    'Leviticus': {'order': 3, 'period': 'Torah', 'approx_date': '1500-1400 BCE'},
    'Numbers': {'order': 4, 'period': 'Torah', 'approx_date': '1500-1400 BCE'},
    'Deuteronomy': {'order': 5, 'period': 'Torah', 'approx_date': '1400 BCE'},
    
    # Historical Books (Early)
    'Joshua': {'order': 6, 'period': 'Historical', 'approx_date': '1400-1350 BCE'},
    'Judges': {'order': 7, 'period': 'Historical', 'approx_date': '1350-1050 BCE'},
    'Ruth': {'order': 8, 'period': 'Historical', 'approx_date': '1100 BCE'},
    '1 Samuel': {'order': 9, 'period': 'Historical', 'approx_date': '1050-970 BCE'},
    '2 Samuel': {'order': 10, 'period': 'Historical', 'approx_date': '1000-970 BCE'},
    
    # Historical Books (Kingdom)
    '1 Kings': {'order': 11, 'period': 'Historical', 'approx_date': '970-850 BCE'},
    '2 Kings': {'order': 12, 'period': 'Historical', 'approx_date': '850-586 BCE'},
    '1 Chronicles': {'order': 13, 'period': 'Historical', 'approx_date': '970-586 BCE'},
    '2 Chronicles': {'order': 14, 'period': 'Historical', 'approx_date': '970-586 BCE'},
    
    # Early Prophets
    'Jonah': {'order': 15, 'period': 'Prophetic', 'approx_date': '850-750 BCE'},
    'Amos': {'order': 16, 'period': 'Prophetic', 'approx_date': '760 BCE'},
    'Hosea': {'order': 17, 'period': 'Prophetic', 'approx_date': '750 BCE'},
    'Isaiah': {'order': 18, 'period': 'Prophetic', 'approx_date': '740-690 BCE'},
    'Micah': {'order': 19, 'period': 'Prophetic', 'approx_date': '735-700 BCE'},
    
    # Middle Prophets
    'Nahum': {'order': 20, 'period': 'Prophetic', 'approx_date': '650 BCE'},
    'Zephaniah': {'order': 21, 'period': 'Prophetic', 'approx_date': '630 BCE'},
    'Jeremiah': {'order': 22, 'period': 'Prophetic', 'approx_date': '626-586 BCE'},
    'Lamentations': {'order': 23, 'period': 'Poetic', 'approx_date': '586 BCE'},
    'Habakkuk': {'order': 24, 'period': 'Prophetic', 'approx_date': '605 BCE'},
    'Ezekiel': {'order': 25, 'period': 'Prophetic', 'approx_date': '593-570 BCE'},
    'Obadiah': {'order': 26, 'period': 'Prophetic', 'approx_date': '586 BCE'},
    
    # Historical Books (Exile and Return)
    'Ezra': {'order': 27, 'period': 'Historical', 'approx_date': '538-458 BCE'},
    'Nehemiah': {'order': 28, 'period': 'Historical', 'approx_date': '445-420 BCE'},
    'Esther': {'order': 29, 'period': 'Historical', 'approx_date': '483-473 BCE'},
    
    # Late Prophets
    'Daniel': {'order': 30, 'period': 'Prophetic', 'approx_date': '605-530 BCE'},
    'Haggai': {'order': 31, 'period': 'Prophetic', 'approx_date': '520 BCE'},
    'Zechariah': {'order': 32, 'period': 'Prophetic', 'approx_date': '520-518 BCE'},
    'Joel': {'order': 33, 'period': 'Prophetic', 'approx_date': '500 BCE'},
    'Malachi': {'order': 34, 'period': 'Prophetic', 'approx_date': '450-400 BCE'},
    
    # Wisdom Literature
    'Job': {'order': 35, 'period': 'Wisdom', 'approx_date': 'Unknown'},
    'Psalms': {'order': 36, 'period': 'Poetic', 'approx_date': '1000-400 BCE'},
    'Proverbs': {'order': 37, 'period': 'Wisdom', 'approx_date': '970-700 BCE'},
    'Ecclesiastes': {'order': 38, 'period': 'Wisdom', 'approx_date': '940-400 BCE'},
    'Song of Solomon': {'order': 39, 'period': 'Poetic', 'approx_date': '970-930 BCE'},
    
    # New Testament
    'Matthew': {'order': 40, 'period': 'Gospel', 'approx_date': '50-70 CE'},
    'Mark': {'order': 41, 'period': 'Gospel', 'approx_date': '55-65 CE'},
    'Luke': {'order': 42, 'period': 'Gospel', 'approx_date': '60-80 CE'},
    'John': {'order': 43, 'period': 'Gospel', 'approx_date': '85-95 CE'},
    'Acts': {'order': 44, 'period': 'Historical', 'approx_date': '62-70 CE'},
    'Romans': {'order': 45, 'period': 'Epistle', 'approx_date': '57 CE'},
    '1 Corinthians': {'order': 46, 'period': 'Epistle', 'approx_date': '53-54 CE'},
    '2 Corinthians': {'order': 47, 'period': 'Epistle', 'approx_date': '55-56 CE'},
    'Galatians': {'order': 48, 'period': 'Epistle', 'approx_date': '48-55 CE'},
    'Ephesians': {'order': 49, 'period': 'Epistle', 'approx_date': '60-62 CE'},
    'Philippians': {'order': 50, 'period': 'Epistle', 'approx_date': '60-62 CE'},
    'Colossians': {'order': 51, 'period': 'Epistle', 'approx_date': '60-62 CE'},
    '1 Thessalonians': {'order': 52, 'period': 'Epistle', 'approx_date': '50-51 CE'},
    '2 Thessalonians': {'order': 53, 'period': 'Epistle', 'approx_date': '50-52 CE'},
    '1 Timothy': {'order': 54, 'period': 'Epistle', 'approx_date': '62-65 CE'},
    '2 Timothy': {'order': 55, 'period': 'Epistle', 'approx_date': '63-67 CE'},
    'Titus': {'order': 56, 'period': 'Epistle', 'approx_date': '63-65 CE'},
    'Philemon': {'order': 57, 'period': 'Epistle', 'approx_date': '60-62 CE'},
    'Hebrews': {'order': 58, 'period': 'Epistle', 'approx_date': '60-70 CE'},
    'James': {'order': 59, 'period': 'Epistle', 'approx_date': '45-50 CE'},
    '1 Peter': {'order': 60, 'period': 'Epistle', 'approx_date': '60-65 CE'},
    '2 Peter': {'order': 61, 'period': 'Epistle', 'approx_date': '65-68 CE'},
    '1 John': {'order': 62, 'period': 'Epistle', 'approx_date': '85-95 CE'},
    '2 John': {'order': 63, 'period': 'Epistle', 'approx_date': '85-95 CE'},
    '3 John': {'order': 64, 'period': 'Epistle', 'approx_date': '85-95 CE'},
    'Jude': {'order': 65, 'period': 'Epistle', 'approx_date': '65-80 CE'},
    'Revelation': {'order': 66, 'period': 'Apocalyptic', 'approx_date': '90-95 CE'}
}


def create_timeline(data: pd.DataFrame,
                  x: str,
                  y: str,
                  size: Optional[str] = None,
                  color: Optional[str] = None,
                  title: str = 'Timeline',
                  figsize: Tuple[int, int] = (12, 8),
                  filename: Optional[str] = None) -> plt.Figure:
    """
    Create a customizable timeline visualization.
    
    Args:
        data: DataFrame containing timeline data
        x: Column name for x-axis (typically time/order)
        y: Column name for y-axis (typically categories)
        size: Optional column name for point size
        color: Optional column name for point color
        title: Title for the plot
        figsize: Figure size (width, height) in inches
        filename: Optional filename to save the figure
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> # Create a timeline of prophetic books
        >>> prophets_data = pd.DataFrame([
        ...     {'book': 'Isaiah', 'year': -740, 'length': 66, 'period': 'Pre-exile'},
        ...     {'book': 'Jeremiah', 'year': -626, 'length': 52, 'period': 'Pre-exile'},
        ...     {'book': 'Ezekiel', 'year': -593, 'length': 48, 'period': 'Exile'},
        ...     {'book': 'Daniel', 'year': -605, 'length': 12, 'period': 'Exile'},
        ...     {'book': 'Hosea', 'year': -750, 'length': 14, 'period': 'Pre-exile'},
        ...     {'book': 'Joel', 'year': -500, 'length': 3, 'period': 'Post-exile'},
        ...     {'book': 'Amos', 'year': -760, 'length': 9, 'period': 'Pre-exile'},
        ...     {'book': 'Obadiah', 'year': -586, 'length': 1, 'period': 'Exile'},
        ...     {'book': 'Jonah', 'year': -780, 'length': 4, 'period': 'Pre-exile'},
        ...     {'book': 'Micah', 'year': -735, 'length': 7, 'period': 'Pre-exile'},
        ...     {'book': 'Nahum', 'year': -650, 'length': 3, 'period': 'Pre-exile'},
        ...     {'book': 'Habakkuk', 'year': -605, 'length': 3, 'period': 'Pre-exile'},
        ...     {'book': 'Zephaniah', 'year': -630, 'length': 3, 'period': 'Pre-exile'},
        ...     {'book': 'Haggai', 'year': -520, 'length': 2, 'period': 'Post-exile'},
        ...     {'book': 'Zechariah', 'year': -520, 'length': 14, 'period': 'Post-exile'},
        ...     {'book': 'Malachi', 'year': -450, 'length': 4, 'period': 'Post-exile'},
        ... ])
        >>> fig = create_timeline(prophets_data, x='year', y='book', 
        ...                       size='length', color='period')
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine size values if specified
    if size:
        # Normalize size to a reasonable range
        sizes = data[size].values
        size_values = 100 * (sizes / sizes.max())
    else:
        size_values = 100  # Default size
    
    # Determine color values if specified
    if color and color in data.columns:
        # If categorical, create a color map
        if pd.api.types.is_categorical_dtype(data[color]) or pd.api.types.is_object_dtype(data[color]):
            categories = data[color].unique()
            cmap = plt.cm.get_cmap('viridis', len(categories))
            color_dict = {cat: cmap(i) for i, cat in enumerate(categories)}
            colors = [color_dict[val] for val in data[color]]
            
            # Create legend handles
            handles = [mpatches.Patch(color=color_dict[cat], label=cat) for cat in categories]
            ax.legend(handles=handles, title=color)
        else:
            # If numeric, use a continuous colormap
            colors = data[color].values
            scatter = ax.scatter(data[x], data[y], s=size_values, c=colors, cmap='viridis')
            plt.colorbar(scatter, ax=ax, label=color)
    else:
        # Use default color
        colors = 'steelblue'
    
    # Create the scatter plot
    if color and color in data.columns and not (pd.api.types.is_categorical_dtype(data[color]) or pd.api.types.is_object_dtype(data[color])):
        # Already created with colorbar above
        pass
    else:
        ax.scatter(data[x], data[y], s=size_values, c=colors)
    
    # Set axis labels
    ax.set_xlabel(x, fontsize=12)
    ax.set_ylabel(y, fontsize=12)
    
    # Set title
    ax.set_title(title, fontsize=14)
    
    # Add text labels for points
    for _, row in data.iterrows():
        ax.text(row[x], row[y], row[y], fontsize=9, 
                ha='right', va='center', backgroundcolor='white', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig


def create_term_timeline(bible_dict: Dict[str, Any],
                       term: str,
                       chronological: bool = True,
                       normalize: bool = True,
                       title: Optional[str] = None,
                       figsize: Tuple[int, int] = (14, 8),
                       filename: Optional[str] = None) -> plt.Figure:
    """
    Create a timeline showing usage of a term across biblical books in chronological order.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        term: Term to track over time
        chronological: Whether to sort books chronologically
        normalize: Whether to normalize frequency by book length
        title: Title for the plot (defaults to "Timeline of '[term]' Usage")
        figsize: Figure size (width, height) in inches
        filename: Optional filename to save the figure
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Create timeline of "covenant" usage
        >>> fig = create_term_timeline(bible, "covenant")
    """
    from religious_texts.text_analysis import frequency
    
    # Set default title if not provided
    if title is None:
        title = f"Timeline of '{term}' Usage"
    
    # Get term frequency by book
    rel_freq = frequency.relative_frequency(bible_dict, term)
    
    if rel_freq.empty:
        # Create empty figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"No occurrences of '{term}' found", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Create DataFrame for timeline
    timeline_data = []
    
    for _, row in rel_freq.iterrows():
        book = row['book']
        count = row['word_count']
        
        # Add chronological information if available
        chrono_info = CHRONOLOGICAL_ORDER.get(book, {'order': 999, 'period': 'Unknown', 'approx_date': 'Unknown'})
        
        data_point = {
            'book': book,
            'count': count,
            'relative_frequency': row['relative_frequency'],
            'order': chrono_info['order'],
            'period': chrono_info['period'],
            'approx_date': chrono_info['approx_date']
        }
        
        timeline_data.append(data_point)
    
    # Convert to DataFrame
    df = pd.DataFrame(timeline_data)
    
    # Sort by chronological order if requested
    if chronological:
        df = df.sort_values('order')
    else:
        df = df.sort_values('relative_frequency', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine x-axis values
    if chronological:
        x = 'order'
        # Add some jitter to books with same order value
        if df['order'].duplicated().any():
            df['order'] = df['order'] + np.random.normal(0, 0.1, len(df))
    else:
        # Create indices for non-chronological view
        df['index'] = range(len(df))
        x = 'index'
    
    # Set y-axis values based on normalization
    y = 'relative_frequency' if normalize else 'count'
    
    # Create the bar chart
    bars = ax.bar(
        df[x], 
        df[y], 
        color=df['period'].map(lambda p: plt.cm.tab10(hash(p) % 10)),
        alpha=0.7
    )
    
    # Add book labels
    if chronological:
        ax.set_xticks(df['order'])
        ax.set_xticklabels(df['book'], rotation=90, ha='center')
        ax.set_xlabel('Chronological Order', fontsize=12)
    else:
        ax.set_xticks(df['index'])
        ax.set_xticklabels(df['book'], rotation=90, ha='center')
        ax.set_xlabel('Books', fontsize=12)
    
    # Set y-axis label
    if normalize:
        ax.set_ylabel(f'Frequency of "{term}" per 1000 words', fontsize=12)
    else:
        ax.set_ylabel(f'Occurrences of "{term}"', fontsize=12)
    
    # Create legend for periods
    period_colors = {period: plt.cm.tab10(hash(period) % 10) for period in df['period'].unique()}
    legend_patches = [mpatches.Patch(color=color, label=period) for period, color in period_colors.items()]
    ax.legend(handles=legend_patches, title='Period', loc='upper right')
    
    # Set title
    ax.set_title(title, fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig


def create_narrative_timeline(bible_dict: Dict[str, Any],
                            book: str,
                            terms: List[str],
                            normalize: bool = True,
                            title: Optional[str] = None,
                            figsize: Tuple[int, int] = (12, 8),
                            filename: Optional[str] = None) -> plt.Figure:
    """
    Create a timeline showing term usage across the narrative flow of a single book.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        book: Book to analyze
        terms: List of terms to track
        normalize: Whether to normalize frequency by chapter length
        title: Title for the plot (defaults to "Narrative Flow in [book]")
        figsize: Figure size (width, height) in inches
        filename: Optional filename to save the figure
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Track key terms through the narrative of Genesis
        >>> terms = ["God", "covenant", "Abraham", "Isaac", "Jacob", "Joseph"]
        >>> fig = create_narrative_timeline(bible, "Genesis", terms)
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
        title = f"Narrative Flow in {book}"
    
    # Get chapters in order
    chapters = sorted(bible_dict[book].keys())
    
    # Initialize data for each term
    term_data = {term: [] for term in terms}
    
    # Process each chapter
    for chapter_num in chapters:
        # Combine all verses in the chapter
        chapter_text = " ".join(bible_dict[book][chapter_num].values())
        
        # Convert to lowercase for case-insensitive matching
        chapter_text_lower = chapter_text.lower()
        
        # Count total words for normalization
        total_words = len(chapter_text.split())
        
        # Count occurrences of each term
        for term in terms:
            # Use word boundary regex for more accurate counting
            pattern = r'\b' + re.escape(term.lower()) + r'\b'
            matches = re.findall(pattern, chapter_text_lower)
            count = len(matches)
            
            # Normalize if requested
            if normalize and total_words > 0:
                frequency = count / total_words * 1000
            else:
                frequency = count
            
            term_data[term].append(frequency)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot line for each term
    for term, frequencies in term_data.items():
        ax.plot(chapters, frequencies, marker='o', linewidth=2, label=term)
    
    # Set axis labels
    ax.set_xlabel('Chapter', fontsize=12)
    if normalize:
        ax.set_ylabel('Frequency per 1000 words', fontsize=12)
    else:
        ax.set_ylabel('Occurrences', fontsize=12)
    
    # Set title
    ax.set_title(title, fontsize=14)
    
    # Add legend
    ax.legend(fontsize=10)
    
    # Add grid for readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig


def create_book_timeline(bible_dict: Dict[str, Any],
                       period_filter: Optional[str] = None,
                       title: Optional[str] = None,
                       figsize: Tuple[int, int] = (12, 10),
                       filename: Optional[str] = None) -> plt.Figure:
    """
    Create a timeline of biblical books with their chronological information.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        period_filter: Optional filter for a specific period
        title: Title for the plot (defaults to "Timeline of Biblical Books")
        figsize: Figure size (width, height) in inches
        filename: Optional filename to save the figure
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Create timeline of all books
        >>> fig = create_book_timeline(bible)
        >>> # Create timeline of just the Epistles
        >>> fig = create_book_timeline(bible, period_filter="Epistle")
    """
    # Set default title if not provided
    if title is None:
        if period_filter:
            title = f"Timeline of Biblical {period_filter} Books"
        else:
            title = "Timeline of Biblical Books"
    
    # Get available books
    available_books = set(bible_dict.keys())
    
    # Create timeline data
    timeline_data = []
    
    for book, chrono_info in CHRONOLOGICAL_ORDER.items():
        if book not in available_books:
            continue
            
        # Skip if period filter applied and doesn't match
        if period_filter and chrono_info['period'] != period_filter:
            continue
            
        # Count total verses
        verse_count = sum(len(verses) for verses in bible_dict[book].values())
        
        # Count total words
        word_count = sum(len(" ".join(verses.values()).split()) 
                         for verses in bible_dict[book].values())
        
        data_point = {
            'book': book,
            'order': chrono_info['order'],
            'period': chrono_info['period'],
            'approx_date': chrono_info['approx_date'],
            'verse_count': verse_count,
            'word_count': word_count,
            'chapter_count': len(bible_dict[book])
        }
        
        timeline_data.append(data_point)
    
    # Skip if no books match filter
    if not timeline_data:
        # Create empty figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No books match the specified filter", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Convert to DataFrame
    df = pd.DataFrame(timeline_data)
    
    # Sort by chronological order
    df = df.sort_values('order')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the scatter plot
    scatter = ax.scatter(
        df['order'], 
        df['book'], 
        s=df['word_count'] / 100,  # Scale size for visibility
        c=df['period'].map(lambda p: plt.cm.tab10(hash(p) % 10)),
        alpha=0.7
    )
    
    # Create legend for periods
    period_colors = {period: plt.cm.tab10(hash(period) % 10) for period in df['period'].unique()}
    legend_patches = [mpatches.Patch(color=color, label=period) for period, color in period_colors.items()]
    ax.legend(handles=legend_patches, title='Period', loc='upper right')
    
    # Add text labels for approximate dates
    for _, row in df.iterrows():
        ax.text(row['order'] + 0.5, row['book'], row['approx_date'], fontsize=8, 
                ha='left', va='center', alpha=0.8)
    
    # Set axis labels
    ax.set_xlabel('Chronological Order', fontsize=12)
    ax.set_ylabel('Book', fontsize=12)
    
    # Set title
    ax.set_title(title, fontsize=14)
    
    # Set x-axis limits with some padding
    ax.set_xlim(df['order'].min() - 1, df['order'].max() + 10)
    
    # Create size legend
    sizes = [1000, 5000, 10000, 20000]
    labels = ['1,000', '5,000', '10,000', '20,000']
    legend_sizes = [size / 100 for size in sizes]  # Scale to match the plot
    
    # Get position for the legend
    handles = [plt.scatter([], [], s=size, color='gray', alpha=0.7) for size in legend_sizes]
    ax.legend(handles, labels, scatterpoints=1, title='Word Count', 
              loc='lower right', frameon=True, fontsize=8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig


def plot_chronology(data: pd.DataFrame,
                  start_col: str,
                  end_col: Optional[str] = None,
                  label_col: str = 'name',
                  category_col: Optional[str] = None,
                  title: str = 'Chronological Timeline',
                  figsize: Tuple[int, int] = (12, 8),
                  filename: Optional[str] = None) -> plt.Figure:
    """
    Create a chronological timeline visualization with time spans.
    
    Args:
        data: DataFrame containing timeline data
        start_col: Column name for starting point
        end_col: Optional column name for ending point (if None, creates points not spans)
        label_col: Column name for point/span labels
        category_col: Optional column for categorizing and coloring items
        title: Title for the plot
        figsize: Figure size (width, height) in inches
        filename: Optional filename to save the figure
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> # Create a timeline of Old Testament kings
        >>> kings_data = pd.DataFrame([
        ...     {'name': 'Saul', 'start': -1050, 'end': -1010, 'kingdom': 'United'},
        ...     {'name': 'David', 'start': -1010, 'end': -970, 'kingdom': 'United'},
        ...     {'name': 'Solomon', 'start': -970, 'end': -930, 'kingdom': 'United'},
        ...     {'name': 'Rehoboam', 'start': -930, 'end': -913, 'kingdom': 'Judah'},
        ...     {'name': 'Jeroboam', 'start': -930, 'end': -909, 'kingdom': 'Israel'},
        ...     {'name': 'Hezekiah', 'start': -729, 'end': -686, 'kingdom': 'Judah'},
        ...     {'name': 'Josiah', 'start': -640, 'end': -609, 'kingdom': 'Judah'},
        ...     {'name': 'Hoshea', 'start': -732, 'end': -722, 'kingdom': 'Israel'},
        ... ])
        >>> fig = plot_chronology(kings_data, 'start', 'end', 'name', 'kingdom')
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate y-positions
    if category_col and category_col in data.columns:
        # Group by category
        categories = data[category_col].unique()
        cat_positions = {cat: i for i, cat in enumerate(categories)}
        
        # Calculate y-positions within categories
        data_sorted = data.sort_values(start_col)
        positions = {}
        
        for cat in categories:
            cat_data = data_sorted[data_sorted[category_col] == cat]
            positions.update({
                i: cat_positions[cat] + (i - cat_data.index.min()) / (cat_data.index.max() - cat_data.index.min() + 1) * 0.8
                for i in cat_data.index
            })
        
        # Assign y-positions
        y_pos = [positions[i] for i in data.index]
        
        # Create category colors
        cmap = plt.cm.get_cmap('tab10', len(categories))
        colors = [cmap(cat_positions[cat] / len(categories)) for cat in data[category_col]]
        
        # Create legend for categories
        handles = [mpatches.Patch(color=cmap(i / len(categories)), label=cat) 
                 for i, cat in enumerate(categories)]
        ax.legend(handles=handles, title=category_col, loc='upper right')
    else:
        # Simple sequential y-positions
        y_pos = range(len(data))
        colors = 'steelblue'
    
    # Plot horizontal spans if end_col provided
    if end_col and end_col in data.columns:
        for i, (_, row) in enumerate(data.iterrows()):
            y = y_pos[i]
            start = row[start_col]
            end = row[end_col]
            label = row[label_col]
            color = colors[i] if isinstance(colors, list) else colors
            
            # Plot the span
            ax.plot([start, end], [y, y], '-', linewidth=6, solid_capstyle='butt', 
                   color=color, alpha=0.7)
            
            # Add markers at start and end
            ax.plot(start, y, 'o', color=color, markersize=6)
            ax.plot(end, y, 'o', color=color, markersize=6)
            
            # Add label text
            ax.text((start + end) / 2, y, label, fontsize=10, 
                   ha='center', va='bottom', backgroundcolor='white', alpha=0.7)
    else:
        # Plot points only
        for i, (_, row) in enumerate(data.iterrows()):
            y = y_pos[i]
            x = row[start_col]
            label = row[label_col]
            color = colors[i] if isinstance(colors, list) else colors
            
            # Plot the point
            ax.plot(x, y, 'o', color=color, markersize=8)
            
            # Add label text
            ax.text(x, y, label, fontsize=10, 
                   ha='left', va='center', backgroundcolor='white', alpha=0.7)
    
    # Set axis labels
    ax.set_xlabel('Year', fontsize=12)
    
    # Hide y-axis ticks and labels
    ax.set_yticks([])
    
    # Set title
    ax.set_title(title, fontsize=14)
    
    # Add grid for readability
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig
