"""
Named Entity Recognition Module

This module provides functions for recognizing and analyzing named entities
in biblical texts, such as people, places, divine names, and theological concepts.
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union, Any, Tuple, Set

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy

# Biblical entity dictionaries
BIBLICAL_PEOPLE = {
    "old_testament_figures": [
        "Adam", "Eve", "Noah", "Abraham", "Sarah", "Isaac", "Rebekah", "Jacob", "Joseph", "Moses",
        "Aaron", "Joshua", "Deborah", "Gideon", "Samuel", "Saul", "David", "Solomon", "Elijah",
        "Elisha", "Isaiah", "Jeremiah", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos", "Jonah"
    ],
    "new_testament_figures": [
        "Jesus", "Mary", "Joseph", "John the Baptist", "Peter", "Andrew", "James", "John",
        "Matthew", "Thomas", "Paul", "Barnabas", "Timothy", "Silas", "Luke", "Mark",
        "Nicodemus", "Mary Magdalene", "Lazarus", "Martha", "Pilate", "Herod"
    ],
    "gospel_characters": [
        "Jesus", "Mary", "Joseph", "John the Baptist", "Peter", "Andrew", "James", "John",
        "Matthew", "Thomas", "Philip", "Nathanael", "Nicodemus", "Mary Magdalene", 
        "Martha", "Lazarus", "Zacchaeus", "Pilate", "Herod", "Caiaphas"
    ],
    "apostles": [
        "Peter", "Andrew", "James", "John", "Philip", "Bartholomew", "Thomas", 
        "Matthew", "James (Alphaeus)", "Thaddaeus", "Simon", "Judas Iscariot", "Paul"
    ]
}

BIBLICAL_PLACES = {
    "regions": [
        "Judea", "Samaria", "Galilee", "Egypt", "Babylonia", "Assyria", "Persia", "Greece",
        "Mesopotamia", "Canaan", "Edom", "Moab", "Philistia", "Rome", "Macedonia", "Achaia", "Asia"
    ],
    "cities": [
        "Jerusalem", "Bethlehem", "Nazareth", "Capernaum", "Jericho", "Babylon", "Rome",
        "Corinth", "Ephesus", "Athens", "Damascus", "Alexandria", "Antioch", "Caesarea",
        "Tyre", "Sidon", "Nineveh", "Ur", "Shechem", "Samaria"
    ],
    "geographical_features": [
        "Jordan River", "Sea of Galilee", "Dead Sea", "Mediterranean", "Mount Sinai",
        "Mount Carmel", "Mount of Olives", "Mount Zion", "Mount Hermon", "Wilderness"
    ],
    "structures": [
        "Temple", "Synagogue", "Tabernacle", "Ark", "Altar", "Holy of Holies", "Holy Place",
        "Palace", "Prison", "Upper Room", "Tomb", "Garden of Gethsemane"
    ]
}

DIVINE_NAMES = {
    "deity_names": [
        "God", "LORD", "Lord", "YHWH", "Yahweh", "Jehovah", "Elohim", "El", "El Shaddai",
        "Adonai", "Holy One", "Most High", "Ancient of Days", "Creator", "Father"
    ],
    "christ_names": [
        "Jesus", "Christ", "Messiah", "Son of God", "Son of Man", "Emmanuel", "Word",
        "Logos", "Lamb of God", "Lion of Judah", "Alpha and Omega", "King of Kings",
        "Lord of Lords", "Savior", "Redeemer", "Good Shepherd"
    ],
    "spirit_names": [
        "Holy Spirit", "Spirit of God", "Counselor", "Comforter", "Advocate",
        "Spirit of Truth", "Spirit of the Lord"
    ]
}

THEOLOGICAL_CONCEPTS = {
    "salvation_terms": [
        "salvation", "save", "redemption", "redeem", "justification", "justify",
        "sanctification", "sanctify", "atonement", "forgiveness", "reconciliation"
    ],
    "faith_terms": [
        "faith", "believe", "trust", "hope", "confidence", "assurance", "conviction"
    ],
    "sin_terms": [
        "sin", "transgression", "iniquity", "wickedness", "evil", "unrighteousness",
        "disobedience", "rebellion", "trespass", "offense"
    ],
    "judgment_terms": [
        "judgment", "condemnation", "wrath", "punishment", "vengeance", "justice",
        "hell", "damnation", "destruction", "fire", "justice", "law", "commandment"
    ],
    "grace_terms": [
        "grace", "mercy", "compassion", "lovingkindness", "favor", "goodness",
        "kindness", "forgiveness", "pardon", "blessing"
    ]
}

def load_nlp_model(model_name: str = "en_core_web_sm") -> Any:
    """
    Load a spaCy NLP model for entity recognition.
    
    Args:
        model_name: Name of the spaCy model to load
        
    Returns:
        Loaded spaCy NLP model
        
    Note:
        If the specified model is not available, will attempt to download it.
        Requires spaCy and the appropriate model to be installed.
    """
    try:
        import spacy
        try:
            return spacy.load(model_name)
        except OSError:
            # If model not found, attempt to download it
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
            return spacy.load(model_name)
    except ImportError:
        raise ImportError("spaCy is required for named entity recognition. Install it with 'pip install spacy'.")
    except Exception as e:
        raise Exception(f"Error loading spaCy model: {str(e)}")

def extract_entities_spacy(text: str, entity_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Extract named entities from text using spaCy NLP.
    
    Args:
        text: Text to analyze
        entity_types: Optional list of entity types to include (e.g., ['PERSON', 'GPE'])
        
    Returns:
        List of dictionaries with entity information
        
    Example:
        >>> # Extract people and places from a passage
        >>> text = "Jesus went to Jerusalem with his disciples Peter and John."
        >>> entities = extract_entities_spacy(text, entity_types=['PERSON', 'GPE'])
        >>> for entity in entities:
        ...     print(f"{entity['text']} ({entity['label']})")
    """
    # Load spaCy model
    nlp = load_nlp_model()
    
    # Process text
    doc = nlp(text)
    
    # Extract entities
    entities = []
    
    for ent in doc.ents:
        # Filter by entity type if specified
        if entity_types and ent.label_ not in entity_types:
            continue
        
        # Add to results
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
            "sentence": ent.sent.text
        })
    
    return entities

def extract_biblical_entities(text: str, 
                             entity_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Extract biblical entities from text using dictionary-based approach.
    
    Args:
        text: Text to analyze
        entity_types: Optional list of entity types to include 
                      (e.g., ['people', 'places', 'divine_names'])
        
    Returns:
        List of dictionaries with entity information
        
    Example:
        >>> # Extract divine names from a passage
        >>> text = "The LORD spoke to Moses on Mount Sinai."
        >>> entities = extract_biblical_entities(text, entity_types=['divine_names'])
        >>> for entity in entities:
        ...     print(f"{entity['text']} ({entity['type']})")
    """
    # Normalize entity types
    if not entity_types:
        entity_types = ['people', 'places', 'divine_names', 'theological_concepts']
    
    entity_types = [et.lower() for et in entity_types]
    
    # Prepare dictionaries to search
    entity_dicts = {}
    
    if 'people' in entity_types:
        entity_dicts['people'] = {}
        for category, names in BIBLICAL_PEOPLE.items():
            for name in names:
                entity_dicts['people'][name] = category
    
    if 'places' in entity_types:
        entity_dicts['places'] = {}
        for category, names in BIBLICAL_PLACES.items():
            for name in names:
                entity_dicts['places'][name] = category
    
    if 'divine_names' in entity_types:
        entity_dicts['divine_names'] = {}
        for category, names in DIVINE_NAMES.items():
            for name in names:
                entity_dicts['divine_names'][name] = category
    
    if 'theological_concepts' in entity_types:
        entity_dicts['theological_concepts'] = {}
        for category, terms in THEOLOGICAL_CONCEPTS.items():
            for term in terms:
                entity_dicts['theological_concepts'][term] = category
    
    # Split text into sentences
    sentences = sent_tokenize(text)
    
    # Extract entities
    entities = []
    
    for sentence in sentences:
        for entity_type, entity_dict in entity_dicts.items():
            for entity_name, category in entity_dict.items():
                # Search for entity in sentence
                matches = list(re.finditer(r'\b' + re.escape(entity_name) + r'\b', sentence))
                
                for match in matches:
                    entities.append({
                        "text": match.group(0),
                        "type": entity_type,
                        "category": category,
                        "start": match.start(),
                        "end": match.end(),
                        "sentence": sentence
                    })
    
    # Sort by position in text
    entities.sort(key=lambda e: e["start"])
    
    return entities

def extract_entities_from_passage(text: str, 
                                use_spacy: bool = True,
                                biblical_entity_types: Optional[List[str]] = None,
                                spacy_entity_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Extract entities from a passage using both spaCy and dictionary-based approaches.
    
    Args:
        text: Text to analyze
        use_spacy: Whether to use spaCy NLP model
        biblical_entity_types: Optional list of biblical entity types to include
        spacy_entity_types: Optional list of spaCy entity types to include
        
    Returns:
        List of dictionaries with entity information
        
    Example:
        >>> # Extract all entity types from a passage
        >>> text = "Jesus went to Jerusalem with his disciples Peter and John."
        >>> entities = extract_entities_from_passage(text)
        >>> for entity in entities:
        ...     print(f"{entity['text']} ({entity['type']})")
    """
    entities = []
    
    # Extract biblical entities
    biblical_entities = extract_biblical_entities(text, entity_types=biblical_entity_types)
    entities.extend(biblical_entities)
    
    # Extract spaCy entities if requested
    if use_spacy:
        spacy_entities = extract_entities_spacy(text, entity_types=spacy_entity_types)
        
        # Convert spaCy entities to match biblical entity format
        for entity in spacy_entities:
            entities.append({
                "text": entity["text"],
                "type": "spacy",
                "category": entity["label"],
                "start": entity["start"],
                "end": entity["end"],
                "sentence": entity["sentence"]
            })
    
    # Sort by position in text
    entities.sort(key=lambda e: e["start"])
    
    # Add unique ID for each entity
    for i, entity in enumerate(entities):
        entity["id"] = i+1
    
    return entities

def analyze_entity_distribution(bible_dict: Dict[str, Any],
                              entity_type: str,
                              specific_entities: Optional[List[str]] = None,
                              books: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Analyze the distribution of entities across biblical texts.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        entity_type: Type of entity to analyze ('people', 'places', 'divine_names', 'theological_concepts')
        specific_entities: Optional list of specific entities to track
        books: Optional list of books to include
        
    Returns:
        DataFrame with entity distribution analysis
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Analyze distribution of apostles in the Gospels
        >>> apostle_distribution = analyze_entity_distribution(
        ...     bible,
        ...     entity_type='people',
        ...     specific_entities=BIBLICAL_PEOPLE['apostles'],
        ...     books=["Matthew", "Mark", "Luke", "John"]
        ... )
    """
    # Validate entity type
    if entity_type not in ['people', 'places', 'divine_names', 'theological_concepts', 'spacy']:
        raise ValueError(f"Invalid entity type: {entity_type}. " 
                        f"Must be one of ['people', 'places', 'divine_names', 'theological_concepts', 'spacy']")
    
    # Get entity dictionary based on type
    if entity_type == 'people':
        entity_dict = {name: category for category, names in BIBLICAL_PEOPLE.items() for name in names}
    elif entity_type == 'places':
        entity_dict = {name: category for category, names in BIBLICAL_PLACES.items() for name in names}
    elif entity_type == 'divine_names':
        entity_dict = {name: category for category, names in DIVINE_NAMES.items() for name in names}
    elif entity_type == 'theological_concepts':
        entity_dict = {name: category for category, names in THEOLOGICAL_CONCEPTS.items() for name in names}
    else:  # spacy
        entity_dict = None
    
    # Filter to specific entities if provided
    if specific_entities and entity_dict:
        entity_dict = {name: entity_dict.get(name) for name in specific_entities if name in entity_dict}
    
    # Initialize counts
    entity_counts = defaultdict(lambda: defaultdict(int))
    book_total_verses = defaultdict(int)
    
    # Process each book
    for book_name, chapters in bible_dict.items():
        # Skip if not in requested books
        if books and book_name not in books:
            continue
        
        # Count verses in book
        book_verses = 0
        
        for chapter_num, verses in chapters.items():
            book_verses += len(verses)
            
            for verse_num, verse_text in verses.items():
                if not verse_text:
                    continue
                
                # Extract entities from verse
                if entity_type != 'spacy':
                    entities = extract_biblical_entities(verse_text, entity_types=[entity_type])
                else:
                    entities = extract_entities_spacy(verse_text)
                
                # Count entities
                for entity in entities:
                    entity_name = entity["text"]
                    
                    # Skip if not in specific entities list
                    if specific_entities and entity_name not in specific_entities:
                        continue
                    
                    # Count occurrence
                    entity_counts[book_name][entity_name] += 1
        
        # Store total verse count
        book_total_verses[book_name] = book_verses
    
    # Prepare results for DataFrame
    results = []
    
    for book_name, entity_dict in entity_counts.items():
        for entity_name, count in entity_dict.items():
            # Calculate frequency per 100 verses
            frequency = count / book_total_verses[book_name] * 100
            
            results.append({
                "book": book_name,
                "entity": entity_name,
                "count": count,
                "total_verses": book_total_verses[book_name],
                "frequency_per_100_verses": frequency
            })
    
    # Convert to DataFrame
    if results:
        df = pd.DataFrame(results)
        
        # Sort by entity and frequency
        df = df.sort_values(["entity", "frequency_per_100_verses"], ascending=[True, False])
    else:
        # Create empty DataFrame with expected columns
        columns = ["book", "entity", "count", "total_verses", "frequency_per_100_verses"]
        df = pd.DataFrame(columns=columns)
    
    return df

def analyze_entity_cooccurrence(bible_dict: Dict[str, Any],
                             entity1_type: str,
                             entity1_names: List[str],
                             entity2_type: str,
                             entity2_names: List[str],
                             books: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Analyze co-occurrence patterns between different types of entities.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        entity1_type: Type of first entity ('people', 'places', 'divine_names', 'theological_concepts')
        entity1_names: List of specific entities of first type to track
        entity2_type: Type of second entity
        entity2_names: List of specific entities of second type to track
        books: Optional list of books to include
        
    Returns:
        DataFrame with entity co-occurrence analysis
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Analyze when Jesus is mentioned with specific places
        >>> cooccurrence = analyze_entity_cooccurrence(
        ...     bible,
        ...     entity1_type='people',
        ...     entity1_names=['Jesus'],
        ...     entity2_type='places',
        ...     entity2_names=['Jerusalem', 'Galilee', 'Nazareth', 'Capernaum'],
        ...     books=["Matthew", "Mark", "Luke", "John"]
        ... )
    """
    # Validate entity types
    valid_types = ['people', 'places', 'divine_names', 'theological_concepts']
    if entity1_type not in valid_types or entity2_type not in valid_types:
        raise ValueError(f"Invalid entity type. Must be one of {valid_types}")
    
    # Initialize co-occurrence counts
    cooccurrence_counts = defaultdict(lambda: defaultdict(int))
    reference_examples = defaultdict(lambda: defaultdict(list))
    
    # Process each book
    for book_name, chapters in bible_dict.items():
        # Skip if not in requested books
        if books and book_name not in books:
            continue
        
        for chapter_num, verses in chapters.items():
            for verse_num, verse_text in verses.items():
                if not verse_text:
                    continue
                
                reference = f"{book_name} {chapter_num}:{verse_num}"
                
                # Extract entities of both types
                entities1 = extract_biblical_entities(verse_text, entity_types=[entity1_type])
                entities2 = extract_biblical_entities(verse_text, entity_types=[entity2_type])
                
                # Filter to requested entities
                entities1 = [e for e in entities1 if e["text"] in entity1_names]
                entities2 = [e for e in entities2 if e["text"] in entity2_names]
                
                # Skip if either entity type not found
                if not entities1 or not entities2:
                    continue
                
                # Record co-occurrences
                for entity1 in entities1:
                    for entity2 in entities2:
                        entity1_name = entity1["text"]
                        entity2_name = entity2["text"]
                        
                        # Update count
                        cooccurrence_counts[entity1_name][entity2_name] += 1
                        
                        # Store example (up to 5 per pair)
                        if len(reference_examples[entity1_name][entity2_name]) < 5:
                            reference_examples[entity1_name][entity2_name].append({
                                "reference": reference,
                                "text": verse_text
                            })
    
    # Prepare results for DataFrame
    results = []
    
    for entity1_name, entity2_dict in cooccurrence_counts.items():
        for entity2_name, count in entity2_dict.items():
            results.append({
                "entity1": entity1_name,
                "entity1_type": entity1_type,
                "entity2": entity2_name,
                "entity2_type": entity2_type,
                "cooccurrence_count": count,
                "example_references": ", ".join([ex["reference"] for ex in reference_examples[entity1_name][entity2_name]]),
                "first_example_text": reference_examples[entity1_name][entity2_name][0]["text"] if reference_examples[entity1_name][entity2_name] else ""
            })
    
    # Convert to DataFrame
    if results:
        df = pd.DataFrame(results)
        
        # Sort by co-occurrence count (descending)
        df = df.sort_values("cooccurrence_count", ascending=False)
    else:
        # Create empty DataFrame with expected columns
        columns = ["entity1", "entity1_type", "entity2", "entity2_type", "cooccurrence_count", 
                  "example_references", "first_example_text"]
        df = pd.DataFrame(columns=columns)
    
    return df

def extract_entity_relationships(text: str, 
                               relation_patterns: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Extract relationships between entities in text based on pattern matching.
    
    Args:
        text: Text to analyze
        relation_patterns: Optional list of relationship patterns to match
        
    Returns:
        List of dictionaries with entity relationship information
        
    Example:
        >>> # Extract relationships from a text
        >>> text = "Jesus spoke to his disciples. Peter said to Jesus, 'You are the Christ.'"
        >>> relationships = extract_entity_relationships(text)
        >>> for rel in relationships:
        ...     print(f"{rel['entity1']} {rel['relation']} {rel['entity2']}")
    """
    # Default relation patterns if none provided
    if not relation_patterns:
        relation_patterns = [
            {
                "name": "speech",
                "patterns": [
                    r"(\w+) (?:said|spoke) to (\w+)",
                    r"(\w+) (?:told|asked) (\w+)"
                ],
                "relation": "spoke to"
            },
            {
                "name": "movement",
                "patterns": [
                    r"(\w+) (?:went|traveled|journeyed) to (\w+)",
                    r"(\w+) (?:arrived at|reached|entered) (\w+)"
                ],
                "relation": "went to"
            },
            {
                "name": "family",
                "patterns": [
                    r"(\w+) (?:was the|is the) (?:son|daughter|father|mother|brother|sister) of (\w+)",
                    r"(\w+)'s (?:son|daughter|father|mother|brother|sister) (\w+)"
                ],
                "relation": "related to"
            },
            {
                "name": "action",
                "patterns": [
                    r"(\w+) (?:healed|blessed|taught|called|chose|sent) (\w+)",
                    r"(\w+) (?:followed|obeyed|worshiped|served) (\w+)"
                ],
                "relation": "acted upon"
            }
        ]
    
    # Extract entities first
    entities = extract_entities_from_passage(text)
    
    # Create a mapping of entity names to their types
    entity_map = {entity["text"]: entity["type"] for entity in entities}
    
    # Extract relationships
    relationships = []
    
    # Process each sentence
    sentences = sent_tokenize(text)
    
    for sentence in sentences:
        # Try each relation pattern
        for relation_dict in relation_patterns:
            relation_type = relation_dict["name"]
            relation_label = relation_dict["relation"]
            
            for pattern in relation_dict["patterns"]:
                matches = re.finditer(pattern, sentence)
                
                for match in matches:
                    entity1 = match.group(1)
                    entity2 = match.group(2)
                    
                    # Skip if either entity not recognized
                    if entity1 not in entity_map or entity2 not in entity_map:
                        continue
                    
                    # Add relationship
                    relationships.append({
                        "entity1": entity1,
                        "entity1_type": entity_map.get(entity1),
                        "relation_type": relation_type,
                        "relation": relation_label,
                        "entity2": entity2,
                        "entity2_type": entity_map.get(entity2),
                        "sentence": sentence
                    })
    
    return relationships

def get_entity_network(bible_dict: Dict[str, Any],
                     entity_type: str,
                     entity_names: Optional[List[str]] = None,
                     relation_types: Optional[List[str]] = None,
                     books: Optional[List[str]] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract a network of entities and their relationships for network analysis.
    
    Args:
        bible_dict: Bible dictionary with structure {book: {chapter: {verse: text}}}
        entity_type: Type of entities to include in network
        entity_names: Optional list of specific entities to include
        relation_types: Optional list of relationship types to include
        books: Optional list of books to include
        
    Returns:
        Tuple of (nodes, edges) lists for network visualization
        
    Example:
        >>> bible = loaders.load_text("kjv.txt")
        >>> # Get apostle relationship network from the Gospels
        >>> nodes, edges = get_entity_network(
        ...     bible,
        ...     entity_type='people',
        ...     entity_names=BIBLICAL_PEOPLE['apostles'],
        ...     books=["Matthew", "Mark", "Luke", "John"]
        ... )
    """
    # Initialize network data
    nodes = {}  # Maps entity name to node attributes
    edges = []  # List of relationship edges
    
    # Determine which books to include
    if books:
        book_subset = {book: bible_dict[book] for book in books if book in bible_dict}
    else:
        book_subset = bible_dict
    
    # Process each book
    for book_name, chapters in book_subset.items():
        for chapter_num, verses in chapters.items():
            # Process each verse
            for verse_num, verse_text in verses.items():
                if not verse_text:
                    continue
                
                reference = f"{book_name} {chapter_num}:{verse_num}"
                
                # Extract entities
                entities = extract_biblical_entities(verse_text, entity_types=[entity_type])
                
                # Filter to requested entities if specified
                if entity_names:
                    entities = [e for e in entities if e["text"] in entity_names]
                
                # Skip if no entities found
                if not entities:
                    continue
                
                # Add nodes for entities
                for entity in entities:
                    entity_name = entity["text"]
                    
                    if entity_name not in nodes:
                        nodes[entity_name] = {
                            "id": entity_name,
                            "type": entity_type,
                            "category": entity.get("category"),
                            "count": 1,
                            "references": [reference]
                        }
                    else:
                        nodes[entity_name]["count"] += 1
                        if reference not in nodes[entity_name]["references"]:
                            nodes[entity_name]["references"].append(reference)
                
                # Extract entity relationships
                if len(entities) > 1:
                    relationships = extract_entity_relationships(verse_text)
                    
                    # Filter to requested relation types if specified
                    if relation_types:
                        relationships = [r for r in relationships if r["relation_type"] in relation_types]
                    
                    # Add edges for relationships
                    for rel in relationships:
                        entity1 = rel["entity1"]
                        entity2 = rel["entity2"]
                        relation = rel["relation"]
                        
                        # Skip if either entity not in our node list
                        if entity1 not in nodes or entity2 not in nodes:
                            continue
                        
                        # Check if edge already exists
                        existing_edge = next((e for e in edges 
                                           if e["source"] == entity1 and e["target"] == entity2 
                                           and e["relation"] == relation), None)
                        
                        if existing_edge:
                            # Update existing edge
                            existing_edge["weight"] += 1
                            existing_edge["references"].append(reference)
                        else:
                            # Add new edge
                            edges.append({
                                "source": entity1,
                                "target": entity2,
                                "relation": relation,
                                "relation_type": rel["relation_type"],
                                "weight": 1,
                                "references": [reference]
                            })
    
    # Convert nodes dictionary to list
    node_list = list(nodes.values())
    
    return node_list, edges
