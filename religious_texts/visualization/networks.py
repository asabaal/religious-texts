"""
Network Graph Visualization Module

This module provides functions for creating and visualizing network graphs
from biblical texts, including concept networks, character relationships,
and co-occurrence patterns.
"""

import re
import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Try to import optional packages
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    from fa2 import ForceAtlas2
    HAS_FORCEATLAS2 = True
except ImportError:
    HAS_FORCEATLAS2 = False


def create_network_graph(nodes: List[Dict[str, Any]], 
                        edges: List[Dict[str, Any]],
                        title: str = 'Network Graph',
                        node_size_attr: Optional[str] = None,
                        node_color_attr: Optional[str] = None,
                        edge_width_attr: Optional[str] = None,
                        edge_color_attr: Optional[str] = None,
                        layout: str = 'spring',
                        figsize: Tuple[int, int] = (12, 10),
                        filename: Optional[str] = None) -> plt.Figure:
    """
    Create a customizable network graph visualization.
    
    Args:
        nodes: List of node dictionaries with at least 'id' and optional attributes
        edges: List of edge dictionaries with at least 'source' and 'target'
        title: Title for the graph
        node_size_attr: Optional attribute in nodes to determine node size
        node_color_attr: Optional attribute in nodes to determine node color
        edge_width_attr: Optional attribute in edges to determine edge width
        edge_color_attr: Optional attribute in edges to determine edge color
        layout: Network layout algorithm ('spring', 'circular', 'kamada_kawai', etc.)
        figsize: Figure size (width, height) in inches
        filename: Optional filename to save the figure
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> # Create a simple network of biblical characters
        >>> nodes = [
        ...     {'id': 'Abraham', 'type': 'patriarch', 'importance': 10},
        ...     {'id': 'Isaac', 'type': 'patriarch', 'importance': 7},
        ...     {'id': 'Jacob', 'type': 'patriarch', 'importance': 8},
        ...     {'id': 'Joseph', 'type': 'patriarch', 'importance': 9},
        ...     {'id': 'Sarah', 'type': 'matriarch', 'importance': 6},
        ...     {'id': 'Rebekah', 'type': 'matriarch', 'importance': 5},
        ...     {'id': 'Rachel', 'type': 'matriarch', 'importance': 6},
        ...     {'id': 'Leah', 'type': 'matriarch', 'importance': 5}
        ... ]
        >>> edges = [
        ...     {'source': 'Abraham', 'target': 'Isaac', 'relation': 'parent', 'weight': 5},
        ...     {'source': 'Isaac', 'target': 'Jacob', 'relation': 'parent', 'weight': 5},
        ...     {'source': 'Jacob', 'target': 'Joseph', 'relation': 'parent', 'weight': 5},
        ...     {'source': 'Abraham', 'target': 'Sarah', 'relation': 'spouse', 'weight': 3},
        ...     {'source': 'Isaac', 'target': 'Rebekah', 'relation': 'spouse', 'weight': 3},
        ...     {'source': 'Jacob', 'target': 'Rachel', 'relation': 'spouse', 'weight': 3},
        ...     {'source': 'Jacob', 'target': 'Leah', 'relation': 'spouse', 'weight': 2}
        ... ]
        >>> fig = create_network_graph(nodes, edges, node_size_attr='importance', 
        ...                           node_color_attr='type', edge_width_attr='weight',
        ...                           edge_color_attr='relation')
    """
    if not HAS_NETWORKX:
        # Create figure with error message
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "NetworkX package not available.\nInstall with: pip install networkx", 
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for node in nodes:
        node_id = node['id']
        G.add_node(node_id, **{k: v for k, v in node.items() if k != 'id'})
    
    # Add edges with attributes
    for edge in edges:
        source = edge['source']
        target = edge['target']
        G.add_edge(source, target, **{k: v for k, v in edge.items() if k not in ['source', 'target']})
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'shell':
        pos = nx.shell_layout(G)
    elif layout == 'spiral':
        pos = nx.spiral_layout(G)
    elif layout == 'random':
        pos = nx.random_layout(G)
    elif HAS_FORCEATLAS2 and layout == 'force_atlas2':
        # Initialize ForceAtlas2 layout
        forceatlas2 = ForceAtlas2(
            outboundAttractionDistribution=True,
            linLogMode=False,
            adjustSizes=False,
            edgeWeightInfluence=1.0,
            jitterTolerance=1.0,
            barnesHutOptimize=True,
            barnesHutTheta=1.2,
            multiThreaded=False,
            scalingRatio=2.0,
            strongGravityMode=False,
            gravity=1.0
        )
        
        # Convert networkx graph to input format for ForceAtlas2
        positions = {node: (np.random.random(), np.random.random()) for node in G.nodes()}
        
        # Run ForceAtlas2
        positions_fa2 = forceatlas2.forceatlas2_networkx_layout(G, pos=positions, iterations=100)
        
        # Update positions
        pos = positions_fa2
    else:
        # Default to spring layout
        pos = nx.spring_layout(G, seed=42)
    
    # Determine node sizes
    if node_size_attr and node_size_attr in G.nodes[list(G.nodes)[0]]:
        # Get min and max for normalization
        size_values = [G.nodes[node][node_size_attr] for node in G.nodes]
        min_size, max_size = min(size_values), max(size_values)
        
        # Normalize to a reasonable range (300-3000)
        node_sizes = [300 + 2700 * (G.nodes[node][node_size_attr] - min_size) / (max_size - min_size + 1e-6)
                    for node in G.nodes]
    else:
        node_sizes = 500  # Default size
    
    # Determine node colors
    if node_color_attr and node_color_attr in G.nodes[list(G.nodes)[0]]:
        # Check if attribute is categorical or numeric
        attr_values = [G.nodes[node][node_color_attr] for node in G.nodes]
        
        if isinstance(attr_values[0], (int, float)):
            # Numeric attribute - use colormap
            node_colors = [G.nodes[node][node_color_attr] for node in G.nodes]
            cmap = plt.cm.viridis
        else:
            # Categorical attribute - assign colors
            categories = list(set(attr_values))
            cmap = plt.cm.get_cmap('tab10', len(categories))
            color_map = {cat: cmap(i / len(categories)) for i, cat in enumerate(categories)}
            node_colors = [color_map[G.nodes[node][node_color_attr]] for node in G.nodes]
            
            # Create legend for categories
            handles = [mpatches.Patch(color=color_map[cat], label=cat) for cat in categories]
            ax.legend(handles=handles, title=node_color_attr, loc='upper right')
    else:
        node_colors = 'skyblue'  # Default color
    
    # Determine edge widths
    if edge_width_attr and G.edges and edge_width_attr in list(G.edges.values())[0]:
        # Get min and max for normalization
        width_values = [edge[2][edge_width_attr] for edge in G.edges(data=True)]
        min_width, max_width = min(width_values), max(width_values)
        
        # Normalize to a reasonable range (1-5)
        edge_widths = [1 + 4 * (edge[2][edge_width_attr] - min_width) / (max_width - min_width + 1e-6)
                      for edge in G.edges(data=True)]
    else:
        edge_widths = 1.0  # Default width
    
    # Determine edge colors
    if edge_color_attr and G.edges and edge_color_attr in list(G.edges.values())[0]:
        # Check if attribute is categorical or numeric
        attr_values = [edge[2][edge_color_attr] for edge in G.edges(data=True)]
        
        if isinstance(attr_values[0], (int, float)):
            # Numeric attribute - use colormap
            edge_colors = [edge[2][edge_color_attr] for edge in G.edges(data=True)]
        else:
            # Categorical attribute - assign colors
            categories = list(set(attr_values))
            cmap = plt.cm.get_cmap('tab10', len(categories))
            color_map = {cat: cmap(i / len(categories)) for i, cat in enumerate(categories)}
            edge_colors = [color_map[edge[2][edge_color_attr]] for edge in G.edges(data=True)]
            
            # Create legend for categories (if not already created for nodes)
            if node_color_attr != edge_color_attr:
                handles = [mpatches.Patch(color=color_map[cat], label=cat) for cat in categories]
                edge_legend = ax.legend(handles=handles, title=edge_color_attr, loc='lower right')
                
                # Add to existing legend or create a new one
                if node_color_attr:
                    ax.add_artist(edge_legend)
    else:
        edge_colors = 'gray'  # Default color
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos, 
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.7,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=10,
        ax=ax
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.8,
        ax=ax
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_family='sans-serif',
        ax=ax
    )
    
    # Set title
    ax.set_title(title, fontsize=14)
    
    # Remove axis
    ax.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig


def create_concept_network(bible_dict: Dict[str, Any],
                         concepts: Dict[str, List[str]],
                         threshold: float = 0.1,
                         book: Optional[str] = None,
                         layout: str = 'spring',
                         title: str = 'Concept Network',
                         figsize: Tuple[int, int] = (14, 12),
                         filename: Optional[str] = None) -> plt.Figure:
    """
    Create a network graph of related theological concepts based on co-occurrence.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        concepts: Dictionary mapping concept names to lists of related terms
        threshold: Minimum co-occurrence correlation to include as edge
        book: Optional book name to filter by
        layout: Network layout algorithm
        title: Title for the graph
        figsize: Figure size (width, height) in inches
        filename: Optional filename to save the figure
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Create network of theological concepts
        >>> concepts = {
        ...     "salvation": ["save", "salvation", "redeem", "deliver"],
        ...     "judgment": ["judge", "judgment", "wrath", "punishment"],
        ...     "love": ["love", "charity", "compassion", "kindness"],
        ...     "faith": ["faith", "believe", "trust", "confidence"],
        ...     "sin": ["sin", "transgression", "iniquity", "evil"],
        ...     "righteousness": ["righteous", "just", "upright", "holy"],
        ...     "wisdom": ["wisdom", "understanding", "knowledge", "discernment"],
        ...     "prayer": ["pray", "prayer", "supplication", "intercession"]
        ... }
        >>> fig = create_concept_network(bible, concepts)
    """
    if not HAS_NETWORKX:
        # Create figure with error message
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "NetworkX package not available.\nInstall with: pip install networkx", 
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Calculate co-occurrence matrix
    from religious_texts.text_analysis.cooccurrence import concept_cooccurrence
    
    # Get concept co-occurrence
    cooccur_df = concept_cooccurrence(bible_dict, concepts, unit='verse', books=[book] if book else None)
    
    if cooccur_df.empty:
        # Create empty figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No concept co-occurrence data available", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Calculate correlation matrix from co-occurrence counts
    corr_matrix = cooccur_df.corr()
    
    # Create node list with attributes
    nodes = []
    
    # Calculate total frequency for each concept
    concept_freq = {}
    
    for concept_name, terms in concepts.items():
        # Count occurrences of concept terms
        count = 0
        
        for term in terms:
            # Search in the text
            for book_name, chapters in bible_dict.items():
                if book and book_name != book:
                    continue
                    
                for chapter_verses in chapters.values():
                    for verse_text in chapter_verses.values():
                        # Use word boundary regex for more accurate counting
                        pattern = r'\b' + re.escape(term.lower()) + r'\b'
                        matches = re.findall(pattern, verse_text.lower())
                        count += len(matches)
        
        concept_freq[concept_name] = count
    
    # Get max frequency for normalization
    max_freq = max(concept_freq.values()) if concept_freq else 1
    
    # Create nodes with normalized size attribute
    for concept_name in concepts:
        nodes.append({
            'id': concept_name,
            'frequency': concept_freq.get(concept_name, 0),
            'size': 1 + 9 * (concept_freq.get(concept_name, 0) / max_freq),
            'term_count': len(concepts[concept_name])
        })
    
    # Create edge list with correlation weights
    edges = []
    
    for i, concept1 in enumerate(corr_matrix.index):
        for concept2 in corr_matrix.columns[i+1:]:
            correlation = corr_matrix.loc[concept1, concept2]
            
            # Only include edges above threshold
            if correlation >= threshold:
                edges.append({
                    'source': concept1,
                    'target': concept2,
                    'weight': correlation,
                    'width': 1 + 5 * correlation  # Scale for visibility
                })
    
    # Create network graph
    fig = create_network_graph(
        nodes=nodes,
        edges=edges,
        title=title,
        node_size_attr='size',
        edge_width_attr='width',
        edge_color_attr='weight',
        layout=layout,
        figsize=figsize,
        filename=filename
    )
    
    return fig


def create_character_network(bible_dict: Dict[str, Any],
                           characters: Dict[str, List[str]],
                           proximity: int = 10,
                           min_cooccurrence: int = 2,
                           book: Optional[str] = None,
                           layout: str = 'spring',
                           title: Optional[str] = None,
                           figsize: Tuple[int, int] = (14, 12),
                           filename: Optional[str] = None) -> plt.Figure:
    """
    Create a network graph of character relationships based on co-occurrence.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        characters: Dictionary mapping character names to name variations
        proximity: Maximum verse distance to count as co-occurrence
        min_cooccurrence: Minimum number of co-occurrences to include edge
        book: Optional book name to filter by
        layout: Network layout algorithm
        title: Title for the graph (defaults to "Character Network in [book]" or "Biblical Character Network")
        figsize: Figure size (width, height) in inches
        filename: Optional filename to save the figure
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Create network of Gospel characters
        >>> gospel_characters = {
        ...     "Jesus": ["Jesus", "Christ", "Son of God", "Son of Man"],
        ...     "Peter": ["Peter", "Simon", "Cephas"],
        ...     "John": ["John", "the disciple whom Jesus loved"],
        ...     "James": ["James", "son of Zebedee"],
        ...     "Andrew": ["Andrew"],
        ...     "Philip": ["Philip"],
        ...     "Thomas": ["Thomas", "Didymus"],
        ...     "Matthew": ["Matthew", "Levi"],
        ...     "Judas": ["Judas", "Iscariot"],
        ...     "Mary": ["Mary", "mother of Jesus"],
        ...     "Mary Magdalene": ["Mary Magdalene"],
        ...     "Martha": ["Martha"],
        ...     "Lazarus": ["Lazarus"],
        ...     "Pilate": ["Pilate"],
        ...     "Herod": ["Herod"]
        ... }
        >>> fig = create_character_network(bible, gospel_characters, 
        ...                               book="John", title="Character Network in John")
    """
    if not HAS_NETWORKX:
        # Create figure with error message
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "NetworkX package not available.\nInstall with: pip install networkx", 
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Set default title if not provided
    if title is None:
        if book:
            title = f"Character Network in {book}"
        else:
            title = "Biblical Character Network"
    
    # Get entity co-occurrence
    character_mentions = defaultdict(list)
    character_count = Counter()
    
    # Process each verse to find character mentions
    for book_name, chapters in bible_dict.items():
        if book and book_name != book:
            continue
            
        for chapter_num, verses in chapters.items():
            for verse_num, verse_text in verses.items():
                # Check for character mentions
                verse_text_lower = verse_text.lower()
                mentioned_chars = set()
                
                for char_name, variations in characters.items():
                    for variation in variations:
                        if variation.lower() in verse_text_lower:
                            mentioned_chars.add(char_name)
                            character_count[char_name] += 1
                            break
                
                # Add verse reference for each character mentioned
                for char in mentioned_chars:
                    character_mentions[char].append((book_name, chapter_num, verse_num))
    
    # Skip if no character mentions found
    if not character_mentions:
        # Create empty figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No character mentions found", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Calculate co-occurrences within proximity
    cooccurrences = Counter()
    
    for char1, mentions1 in character_mentions.items():
        for char2, mentions2 in character_mentions.items():
            if char1 >= char2:  # Skip self and duplicates
                continue
                
            # Count verses where both characters appear within proximity
            count = 0
            
            for book1, chapter1, verse1 in mentions1:
                for book2, chapter2, verse2 in mentions2:
                    # Simple proximity: same book, same chapter, verses close together
                    if (book1 == book2 and 
                        chapter1 == chapter2 and
                        abs(verse1 - verse2) <= proximity):
                        count += 1
            
            if count >= min_cooccurrence:
                cooccurrences[(char1, char2)] = count
    
    # Create node list
    nodes = []
    for char, count in character_count.items():
        nodes.append({
            'id': char,
            'frequency': count,
            'size': math.sqrt(count)  # Scale for better visualization
        })
    
    # Create edge list
    edges = []
    for (char1, char2), count in cooccurrences.items():
        edges.append({
            'source': char1,
            'target': char2,
            'weight': count,
            'width': math.sqrt(count)  # Scale for better visualization
        })
    
    # Create network graph
    fig = create_network_graph(
        nodes=nodes,
        edges=edges,
        title=title,
        node_size_attr='size',
        edge_width_attr='width',
        layout=layout,
        figsize=figsize,
        filename=filename
    )
    
    return fig


def create_verse_network(bible_dict: Dict[str, Any],
                       book: str,
                       threshold: float = 0.3,
                       max_verses: int = 100,
                       layout: str = 'spring',
                       title: Optional[str] = None,
                       figsize: Tuple[int, int] = (14, 12),
                       filename: Optional[str] = None) -> plt.Figure:
    """
    Create a network of verses connected by similar language and themes.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        book: Book to analyze
        threshold: Minimum similarity to include edge
        max_verses: Maximum number of verses to include (for performance)
        layout: Network layout algorithm
        title: Title for the graph (defaults to "Verse Network in [book]")
        figsize: Figure size (width, height) in inches
        filename: Optional filename to save the figure
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Create network of verses in Romans
        >>> fig = create_verse_network(bible, "Romans", max_verses=50)
    """
    if not HAS_NETWORKX:
        # Create figure with error message
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "NetworkX package not available.\nInstall with: pip install networkx", 
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Check if book exists
    if book not in bible_dict:
        # Create empty figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Book '{book}' not found", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Set default title if not provided
    if title is None:
        title = f"Verse Network in {book}"
    
    # Collect verses
    verses = []
    
    for chapter_num, chapter_verses in bible_dict[book].items():
        for verse_num, verse_text in chapter_verses.items():
            verses.append({
                'id': f"{chapter_num}:{verse_num}",
                'chapter': chapter_num,
                'verse': verse_num,
                'text': verse_text,
                'words': len(verse_text.split())
            })
    
    # Limit number of verses for performance
    if len(verses) > max_verses:
        # Sample verses across chapters
        sampled_verses = []
        verse_count = 0
        
        # Get all chapter numbers
        chapters = sorted(bible_dict[book].keys())
        
        # Calculate verses per chapter to sample
        verses_per_chapter = max(1, max_verses // len(chapters))
        
        # Sample from each chapter
        for chapter_num in chapters:
            chapter_verses = [v for v in verses if v['chapter'] == chapter_num]
            
            # If chapter has fewer verses than needed, take all
            if len(chapter_verses) <= verses_per_chapter:
                sampled_verses.extend(chapter_verses)
                verse_count += len(chapter_verses)
            else:
                # Sample evenly
                step = len(chapter_verses) / verses_per_chapter
                indices = [int(i * step) for i in range(verses_per_chapter)]
                sampled_verses.extend([chapter_verses[i] for i in indices])
                verse_count += verses_per_chapter
            
            # Stop if we've reached our limit
            if verse_count >= max_verses:
                break
        
        verses = sampled_verses[:max_verses]
    
    # Create nodes with attributes
    nodes = [{
        'id': verse['id'],
        'chapter': verse['chapter'],
        'verse': verse['verse'],
        'text': verse['text'][:30] + '...' if len(verse['text']) > 30 else verse['text'],
        'size': verse['words']
    } for verse in verses]
    
    # Calculate similarity between verses
    edges = []
    
    # Convert verses to word sets for faster comparison
    verse_words = {}
    for verse in verses:
        # Tokenize and convert to lowercase
        words = set(word.lower() for word in verse['text'].split())
        verse_words[verse['id']] = words
    
    # Calculate Jaccard similarity between verse pairs
    for i, verse1 in enumerate(verses):
        for verse2 in verses[i+1:]:
            words1 = verse_words[verse1['id']]
            words2 = verse_words[verse2['id']]
            
            # Calculate Jaccard similarity: intersection / union
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            similarity = intersection / union if union > 0 else 0
            
            # Only include edges above threshold
            if similarity >= threshold:
                edges.append({
                    'source': verse1['id'],
                    'target': verse2['id'],
                    'weight': similarity,
                    'width': similarity * 3  # Scale for visibility
                })
    
    # Skip if no edges found
    if not edges:
        # Create empty figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No verse connections found above threshold", 
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Create network graph
    fig = create_network_graph(
        nodes=nodes,
        edges=edges,
        title=title,
        node_size_attr='size',
        node_color_attr='chapter',
        edge_width_attr='width',
        edge_color_attr='weight',
        layout=layout,
        figsize=figsize,
        filename=filename
    )
    
    return fig


def plot_cooccurrence_network(cooccur_df: pd.DataFrame,
                            threshold: float = 0.0,
                            layout: str = 'spring',
                            title: str = 'Co-occurrence Network',
                            figsize: Tuple[int, int] = (12, 10),
                            filename: Optional[str] = None) -> plt.Figure:
    """
    Create a network visualization from a co-occurrence DataFrame.
    
    Args:
        cooccur_df: Co-occurrence DataFrame with items as both rows and columns
        threshold: Minimum co-occurrence value to include as edge
        layout: Network layout algorithm
        title: Title for the graph
        figsize: Figure size (width, height) in inches
        filename: Optional filename to save the figure
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> from religious_texts.text_analysis import frequency, cooccurrence
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Get divine name co-occurrence
        >>> divine_names = ["god", "lord", "jesus", "christ", "spirit"]
        >>> cooccur = cooccurrence.word_cooccurrence(bible, divine_names)
        >>> # Visualize as network
        >>> fig = plot_cooccurrence_network(cooccur)
    """
    if not HAS_NETWORKX:
        # Create figure with error message
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "NetworkX package not available.\nInstall with: pip install networkx", 
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Check if DataFrame is valid
    if cooccur_df.empty:
        # Create empty figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No co-occurrence data available", 
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Ensure DataFrame is square (same row and column names)
    if not all(col in cooccur_df.index for col in cooccur_df.columns):
        # Create empty figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Co-occurrence DataFrame must have same row and column names", 
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Create node list
    nodes = []
    
    # Calculate total co-occurrence for each node
    totals = cooccur_df.sum()
    
    for item in cooccur_df.index:
        nodes.append({
            'id': item,
            'total': totals[item],
            'size': math.sqrt(totals[item]) * 100  # Scale for visibility
        })
    
    # Create edge list
    edges = []
    
    for i, item1 in enumerate(cooccur_df.index):
        for item2 in cooccur_df.columns[i+1:]:
            # Get co-occurrence value
            value = cooccur_df.loc[item1, item2]
            
            # Only include edges above threshold
            if value > threshold:
                edges.append({
                    'source': item1,
                    'target': item2,
                    'weight': value,
                    'width': math.sqrt(value)  # Scale for visibility
                })
    
    # Create network graph
    fig = create_network_graph(
        nodes=nodes,
        edges=edges,
        title=title,
        node_size_attr='size',
        edge_width_attr='width',
        edge_color_attr='weight',
        layout=layout,
        figsize=figsize,
        filename=filename
    )
    
    return fig
