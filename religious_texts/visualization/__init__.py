"""
Visualization Module

This module provides functions for visualizing biblical text analysis results including:
- Heat maps for argument strength assessment
- Word and concept distribution visualizations
- Timeline representations of key terms
- Network graphs of related concepts
"""

from religious_texts.visualization.heatmaps import (
    create_heatmap,
    create_book_heatmap,
    create_term_heatmap,
    create_concept_heatmap,
    create_chapter_heatmap
)

from religious_texts.visualization.distributions import (
    plot_word_frequency,
    plot_frequency_comparison,
    plot_word_distribution,
    plot_concept_distribution,
    create_wordcloud
)

from religious_texts.visualization.timelines import (
    create_timeline,
    create_term_timeline,
    create_narrative_timeline,
    create_book_timeline,
    plot_chronology
)

from religious_texts.visualization.networks import (
    create_network_graph,
    create_concept_network,
    create_character_network,
    create_verse_network,
    plot_cooccurrence_network
)

__all__ = [
    # Heatmaps
    'create_heatmap',
    'create_book_heatmap',
    'create_term_heatmap',
    'create_concept_heatmap',
    'create_chapter_heatmap',
    
    # Distributions
    'plot_word_frequency',
    'plot_frequency_comparison',
    'plot_word_distribution',
    'plot_concept_distribution',
    'create_wordcloud',
    
    # Timelines
    'create_timeline',
    'create_term_timeline',
    'create_narrative_timeline',
    'create_book_timeline',
    'plot_chronology',
    
    # Networks
    'create_network_graph',
    'create_concept_network',
    'create_character_network',
    'create_verse_network',
    'plot_cooccurrence_network'
]
