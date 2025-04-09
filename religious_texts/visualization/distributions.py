"""
Distributions Visualization Module

This module provides functions for visualizing word and concept distributions
in biblical texts, including frequency plots, comparisons, and word clouds.
"""

import re
from collections import Counter
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Try to import optional wordcloud package
try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False


def plot_word_frequency(freq_data: Dict[str, int], 
                       title: str = 'Word Frequency',
                       top_n: int = 20,
                       color: str = 'steelblue',
                       figsize: Tuple[int, int] = (10, 6),
                       horizontal: bool = True,
                       filename: Optional[str] = None) -> plt.Figure:
    """
    Create a bar chart of word frequencies.
    
    Args:
        freq_data: Dictionary of {word: frequency}
        title: Title for the plot
        top_n: Number of top words to display
        color: Bar color
        figsize: Figure size (width, height) in inches
        horizontal: Whether to use horizontal bars (better for long words)
        filename: Optional filename to save the figure
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> from religious_texts.text_analysis import frequency
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Calculate word frequencies in Genesis
        >>> freq = frequency.word_frequency(bible, book="Genesis")
        >>> # Plot top 20 words
        >>> fig = plot_word_frequency(freq)
    """
    # Sort by frequency and get top N
    sorted_freq = dict(sorted(freq_data.items(), key=lambda x: x[1], reverse=True)[:top_n])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get words and frequencies
    words = list(sorted_freq.keys())
    freqs = list(sorted_freq.values())
    
    # Reverse lists for horizontal bars (to have highest frequency at top)
    if horizontal:
        words.reverse()
        freqs.reverse()
        
        # Create horizontal bar chart
        ax.barh(words, freqs, color=color)
        ax.set_xlabel('Frequency', fontsize=12)
        ax.tick_params(axis='y', labelsize=10)
    else:
        # Create vertical bar chart
        ax.bar(words, freqs, color=color)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_xticklabels(words, rotation=45, ha='right')
        ax.tick_params(axis='x', labelsize=10)
    
    # Set title
    ax.set_title(title, fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig


def plot_frequency_comparison(data: Dict[str, Dict[str, int]],
                            title: str = 'Word Frequency Comparison',
                            top_n: int = 15,
                            figsize: Tuple[int, int] = (12, 8),
                            palette: str = 'viridis',
                            horizontal: bool = True,
                            filename: Optional[str] = None) -> plt.Figure:
    """
    Create a comparative bar chart of word frequencies across different texts.
    
    Args:
        data: Dictionary of {text_name: {word: frequency}}
        title: Title for the plot
        top_n: Number of top words to display
        figsize: Figure size (width, height) in inches
        palette: Color palette to use
        horizontal: Whether to use horizontal bars
        filename: Optional filename to save the figure
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> from religious_texts.text_analysis import frequency
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Compare word frequencies in different gospels
        >>> matthew_freq = frequency.word_frequency(bible, book="Matthew")
        >>> mark_freq = frequency.word_frequency(bible, book="Mark")
        >>> luke_freq = frequency.word_frequency(bible, book="Luke")
        >>> john_freq = frequency.word_frequency(bible, book="John")
        >>> comparison = {
        ...     "Matthew": matthew_freq,
        ...     "Mark": mark_freq,
        ...     "Luke": luke_freq,
        ...     "John": john_freq
        ... }
        >>> fig = plot_frequency_comparison(comparison)
    """
    # Combine frequencies across all texts
    all_words = {}
    for text_freqs in data.values():
        for word, freq in text_freqs.items():
            if word in all_words:
                all_words[word] += freq
            else:
                all_words[word] = freq
    
    # Get top N words overall
    top_words = dict(sorted(all_words.items(), key=lambda x: x[1], reverse=True)[:top_n]).keys()
    
    # Create DataFrame for comparison
    df = pd.DataFrame({text: {word: text_freqs.get(word, 0) for word in top_words} 
                      for text, text_freqs in data.items()})
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot based on orientation
    if horizontal:
        # Transpose DataFrame for horizontal bars
        df = df.T
        
        # Create horizontal bar chart
        df.plot.barh(ax=ax, colormap=palette)
        ax.set_xlabel('Frequency', fontsize=12)
        ax.legend(title='Text', fontsize=10)
    else:
        # Create vertical bar chart
        df.plot.bar(ax=ax, colormap=palette)
        ax.set_ylabel('Frequency', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        ax.legend(title='Text', fontsize=10)
    
    # Set title
    ax.set_title(title, fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig


def plot_word_distribution(bible_dict: Dict[str, Any],
                         word: str,
                         unit: str = 'book',
                         books: Optional[List[str]] = None,
                         title: Optional[str] = None,
                         color: str = 'steelblue',
                         figsize: Tuple[int, int] = (12, 8),
                         filename: Optional[str] = None) -> plt.Figure:
    """
    Plot the distribution of a word across books or chapters.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        word: Word to analyze
        unit: Unit for distribution ('book' or 'chapter')
        books: Optional list of specific books to include
        title: Title for the plot (defaults to "Distribution of '[word]'")
        color: Bar color
        figsize: Figure size (width, height) in inches
        filename: Optional filename to save the figure
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Plot distribution of "faith" across books
        >>> fig = plot_word_distribution(bible, "faith")
    """
    from religious_texts.text_analysis import frequency
    
    # Set default title if not provided
    if title is None:
        title = f"Distribution of '{word}'"
    
    # Get word frequency data
    if unit == 'book':
        # Get frequency by book
        freq_df = frequency.relative_frequency(bible_dict, word, books=books)
        
        if freq_df.empty:
            # Create empty figure
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f"No occurrences of '{word}' found", ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Sort by relative frequency
        freq_df = freq_df.sort_values('relative_frequency', ascending=False)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar chart
        ax.barh(freq_df['book'], freq_df['relative_frequency'], color=color)
        ax.set_xlabel(f'Frequency per 1000 words', fontsize=12)
        ax.tick_params(axis='y', labelsize=10)
        
    elif unit == 'chapter':
        # Initialize data structure
        data = []
        
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
        
        # Process each book and chapter
        for book in books_to_check:
            for chapter_num, verses in bible_dict[book].items():
                # Combine all verses in the chapter
                chapter_text = " ".join(verses.values())
                
                # Count word occurrences (case-insensitive)
                word_count = chapter_text.lower().count(word.lower())
                
                # Normalize by chapter length
                total_words = len(chapter_text.split())
                if total_words > 0:
                    rel_freq = word_count / total_words * 1000
                else:
                    rel_freq = 0
                
                # Add to data
                data.append({
                    'book': book,
                    'chapter': chapter_num,
                    'reference': f"{book} {chapter_num}",
                    'word_count': word_count,
                    'total_words': total_words,
                    'relative_frequency': rel_freq
                })
        
        # Convert to DataFrame
        freq_df = pd.DataFrame(data)
        
        if freq_df.empty:
            # Create empty figure
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f"No occurrences of '{word}' found", ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Sort by relative frequency
        freq_df = freq_df.sort_values('relative_frequency', ascending=False)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar chart (top 50 chapters to avoid overcrowding)
        top_df = freq_df.head(50)
        ax.barh(top_df['reference'], top_df['relative_frequency'], color=color)
        ax.set_xlabel(f'Frequency per 1000 words', fontsize=12)
        ax.tick_params(axis='y', labelsize=10)
        
    else:
        raise ValueError(f"Unknown unit '{unit}'. Choose from 'book' or 'chapter'.")
    
    # Set title
    ax.set_title(title, fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig


def plot_concept_distribution(bible_dict: Dict[str, Any],
                            concepts: Dict[str, List[str]],
                            books: Optional[List[str]] = None,
                            normalize: bool = True,
                            title: str = 'Concept Distribution',
                            figsize: Tuple[int, int] = (12, 10),
                            filename: Optional[str] = None) -> plt.Figure:
    """
    Plot the distribution of concepts across books.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        concepts: Dictionary mapping concept names to lists of related terms
        books: Optional list of specific books to include
        normalize: Whether to normalize frequencies by book length
        title: Title for the plot
        figsize: Figure size (width, height) in inches
        filename: Optional filename to save the figure
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Plot distribution of theological concepts
        >>> concepts = {
        ...     "salvation": ["save", "salvation", "redeem", "deliver"],
        ...     "judgment": ["judge", "judgment", "wrath", "punishment"],
        ...     "love": ["love", "charity", "compassion", "kindness"],
        ...     "faith": ["faith", "believe", "trust", "confidence"]
        ... }
        >>> fig = plot_concept_distribution(bible, concepts)
    """
    # Initialize data structure
    data = []
    
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
        for concept_name, terms in concepts.items():
            # Initialize count
            count = 0
            
            # Count occurrences of each term in the concept
            for term in terms:
                # Use word boundary regex for more accurate counting
                pattern = r'\b' + re.escape(term.lower()) + r'\b'
                matches = re.findall(pattern, book_text_lower)
                count += len(matches)
            
            # Calculate frequency
            if normalize and total_words > 0:
                rel_freq = count / total_words * 1000
            else:
                rel_freq = count
            
            # Add to data
            data.append({
                'book': book,
                'concept': concept_name,
                'count': count,
                'total_words': total_words,
                'relative_frequency': rel_freq
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    if df.empty:
        # Create empty figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No concept occurrences found", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Create figure with subplots for each concept
    num_concepts = len(concepts)
    fig, axes = plt.subplots(num_concepts, 1, figsize=figsize, sharex=True)
    
    # If only one concept, axes is not an array
    if num_concepts == 1:
        axes = [axes]
    
    # Plot each concept
    for i, concept_name in enumerate(concepts.keys()):
        # Filter data for this concept
        concept_df = df[df['concept'] == concept_name]
        
        # Sort by relative frequency
        concept_df = concept_df.sort_values('relative_frequency', ascending=False)
        
        # Get top 15 books for readability
        concept_df = concept_df.head(15)
        
        # Create horizontal bar chart
        axes[i].barh(concept_df['book'], concept_df['relative_frequency'])
        axes[i].set_title(concept_name, fontsize=12)
        
        # Set y-axis labels only for the first subplot
        if i == 0:
            axes[i].set_xlabel('Frequency per 1000 words' if normalize else 'Count')
    
    # Set overall title
    fig.suptitle(title, fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig


def create_wordcloud(text: str,
                   title: Optional[str] = None,
                   stopwords: Optional[Set[str]] = None,
                   mask: Optional[np.ndarray] = None,
                   colormap: str = 'viridis',
                   background_color: str = 'white',
                   figsize: Tuple[int, int] = (10, 10),
                   filename: Optional[str] = None) -> plt.Figure:
    """
    Create a word cloud visualization from text.
    
    Args:
        text: Text to visualize
        title: Optional title for the plot
        stopwords: Optional set of words to exclude
        mask: Optional NumPy array for custom shape
        colormap: Matplotlib colormap name
        background_color: Background color
        figsize: Figure size (width, height) in inches
        filename: Optional filename to save the figure
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Create word cloud for Psalms
        >>> psalms_text = " ".join(
        ...     verse_text 
        ...     for chapter_verses in bible["Psalms"].values() 
        ...     for verse_text in chapter_verses.values()
        ... )
        >>> fig = create_wordcloud(psalms_text, title="Psalms Word Cloud")
    """
    if not HAS_WORDCLOUD:
        # Create figure with error message
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "WordCloud package not available.\nInstall with: pip install wordcloud", 
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Create WordCloud object
    wordcloud = WordCloud(
        width=800, 
        height=800, 
        background_color=background_color,
        stopwords=stopwords,
        colormap=colormap,
        mask=mask,
        max_font_size=200, 
        random_state=42
    )
    
    # Generate word cloud
    wordcloud.generate(text)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display word cloud
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis('off')
    
    # Set title if provided
    if title:
        ax.set_title(title, fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig


def create_multi_wordcloud(texts: Dict[str, str],
                         stopwords: Optional[Set[str]] = None,
                         colormap: str = 'viridis',
                         background_color: str = 'white',
                         figsize: Tuple[int, int] = (15, 10),
                         filename: Optional[str] = None) -> plt.Figure:
    """
    Create multiple word clouds for comparison.
    
    Args:
        texts: Dictionary mapping text names to text content
        stopwords: Optional set of words to exclude
        colormap: Matplotlib colormap name
        background_color: Background color
        figsize: Figure size (width, height) in inches
        filename: Optional filename to save the figure
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Create word clouds for different books
        >>> book_texts = {
        ...     "Genesis": " ".join(verse_text for chapter in bible["Genesis"].values() 
        ...                       for verse_text in chapter.values()),
        ...     "Exodus": " ".join(verse_text for chapter in bible["Exodus"].values() 
        ...                      for verse_text in chapter.values()),
        ...     "Leviticus": " ".join(verse_text for chapter in bible["Leviticus"].values() 
        ...                         for verse_text in chapter.values()),
        ...     "Numbers": " ".join(verse_text for chapter in bible["Numbers"].values() 
        ...                       for verse_text in chapter.values())
        ... }
        >>> fig = create_multi_wordcloud(book_texts)
    """
    if not HAS_WORDCLOUD:
        # Create figure with error message
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "WordCloud package not available.\nInstall with: pip install wordcloud", 
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Determine grid dimensions
    n_texts = len(texts)
    n_cols = min(3, n_texts)
    n_rows = (n_texts + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array for easier indexing
    if n_rows * n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Create word cloud for each text
    for i, (name, text) in enumerate(texts.items()):
        if i < len(axes):
            # Create WordCloud object
            wordcloud = WordCloud(
                width=400, 
                height=400, 
                background_color=background_color,
                stopwords=stopwords,
                colormap=colormap,
                max_font_size=100, 
                random_state=42
            )
            
            # Generate word cloud
            wordcloud.generate(text)
            
            # Display word cloud
            axes[i].imshow(wordcloud, interpolation="bilinear")
            axes[i].set_title(name, fontsize=12)
            axes[i].axis('off')
    
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig
