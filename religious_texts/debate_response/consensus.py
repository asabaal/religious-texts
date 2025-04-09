"""
Scholarly Consensus Module

This module provides tools for assessing scholarly consensus on biblical
interpretation questions and related theological claims.
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union, Any, Tuple, Set

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize

# Define scholarly tradition categories
SCHOLARLY_TRADITIONS = {
    "evangelical": {
        "description": "Conservative Protestant tradition emphasizing biblical authority",
        "representatives": ["D.A. Carson", "Craig Blomberg", "N.T. Wright", "Wayne Grudem", 
                          "John Piper", "Kevin DeYoung", "Timothy Keller", "Alister McGrath"]
    },
    "critical": {
        "description": "Historical-critical scholarship with emphasis on historical context and development",
        "representatives": ["Bart Ehrman", "John Dominic Crossan", "Paula Fredriksen", 
                          "Amy-Jill Levine", "Rudolf Bultmann", "James D.G. Dunn", "E.P. Sanders"]
    },
    "catholic": {
        "description": "Roman Catholic theological tradition with emphasis on church teaching",
        "representatives": ["Raymond Brown", "Joseph Ratzinger", "Hans Küng", "Luke Timothy Johnson",
                          "John P. Meier", "Francis J. Moloney", "Gerald O'Collins"]
    },
    "orthodox": {
        "description": "Eastern Orthodox theological tradition",
        "representatives": ["John Behr", "Andrew Louth", "John Meyendorff", 
                          "Kallistos Ware", "Vladimir Lossky", "Georges Florovsky"]
    },
    "jewish": {
        "description": "Jewish scholarship on Hebrew Bible/Old Testament",
        "representatives": ["Jacob Neusner", "Jon Levenson", "Amy-Jill Levine", 
                          "Daniel Boyarin", "Adele Berlin", "Marc Zvi Brettler"]
    },
    "textual": {
        "description": "Specialized focus on textual criticism and manuscript evidence",
        "representatives": ["Bruce Metzger", "Kurt Aland", "Eldon Jay Epp", 
                          "David Parker", "Emanuel Tov", "Philip Comfort"]
    }
}

# Define known positions on key theological debates
THEOLOGICAL_POSITIONS = {
    "christology": {
        "trinitarian": {
            "description": "Jesus shares the same divine nature as God the Father",
            "key_concepts": ["trinity", "hypostatic union", "one substance", "incarnation", 
                           "fully God and fully man", "pre-existence"]
        },
        "unitarian": {
            "description": "Jesus is distinct from God the Father who alone is the one true God",
            "key_concepts": ["strict monotheism", "divine agency", "subordination", 
                           "functional divinity", "representative", "exaltation"]
        },
        "adoptionist": {
            "description": "Jesus became divine at some point (baptism, resurrection) but wasn't originally",
            "key_concepts": ["adoption", "became Son", "exaltation", "appointed", 
                           "conferred status", "at baptism", "at resurrection"]
        }
    },
    "soteriology": {
        "substitutionary": {
            "description": "Christ's death was a substitute punishment for human sin",
            "key_concepts": ["penal substitution", "satisfaction", "propitiation", 
                           "paid penalty", "took our punishment", "wrath of God"]
        },
        "exemplary": {
            "description": "Christ's death primarily serves as moral example of love and sacrifice",
            "key_concepts": ["moral influence", "example", "demonstration of love", 
                           "moral transformation", "inspire", "follow his path"]
        },
        "ransom": {
            "description": "Christ's death was a ransom payment to free humanity from bondage",
            "key_concepts": ["ransom", "redemption", "bondage", "slavery to sin", 
                           "liberation", "freedom", "captivity", "price paid"]
        },
        "victory": {
            "description": "Christ's death and resurrection defeated evil powers (Christus Victor)",
            "key_concepts": ["victory", "triumph", "defeat", "powers", "principalities", 
                           "Satan", "evil", "conquer", "overcome"]
        }
    },
    "inspiration": {
        "verbal_plenary": {
            "description": "Every word of the original biblical texts was inspired by God",
            "key_concepts": ["verbal inspiration", "plenary", "inerrant", "infallible", 
                           "God-breathed", "every word", "wholly reliable"]
        },
        "dynamic": {
            "description": "The concepts and message were inspired but not necessarily every word",
            "key_concepts": ["dynamic", "thought inspiration", "message", "concepts", 
                           "truth", "reliable message", "human elements"]
        },
        "encounter": {
            "description": "Scripture records human encounters with God rather than dictated content",
            "key_concepts": ["encounter", "witness", "testimony", "experience", 
                           "record", "human perspective", "interpretation"]
        }
    }
}

def measure_scholarly_consensus(topic: str, 
                               include_traditions: Optional[List[str]] = None,
                               method: str = "count") -> Dict[str, Any]:
    """
    Measure the level of scholarly consensus on a biblical or theological topic.
    
    Args:
        topic: The topic or question to analyze
        include_traditions: Optional list of scholarly traditions to include
        method: Method for measuring consensus ("count", "weighted", or "network")
        
    Returns:
        Dictionary with consensus measurement results
        
    Example:
        >>> # Measure consensus on the authorship of Hebrews
        >>> consensus = measure_scholarly_consensus(
        ...     "Who wrote the Epistle to the Hebrews?",
        ...     include_traditions=["evangelical", "critical", "catholic"]
        ... )
        >>> print(consensus["consensus_level"])
    """
    # Set default traditions if not specified
    if include_traditions is None:
        include_traditions = list(SCHOLARLY_TRADITIONS.keys())
    
    # Filter traditions
    traditions = {t: info for t, info in SCHOLARLY_TRADITIONS.items() if t in include_traditions}
    
    # Initialize result
    result = {
        "topic": topic,
        "traditions": list(traditions.keys()),
        "positions": [],
        "consensus_level": None,
        "confidence": 0.0,
        "dominant_view": None,
        "alternative_views": [],
        "methodological_factors": []
    }
    
    # This is a placeholder implementation that would be replaced by a real
    # analysis of scholarly literature or expert opinions in a production system
    
    # For demonstration, we'll handle some common topics with predefined data
    topic_lower = topic.lower()
    
    # Map of topics to known consensus data
    known_topics = {
        "authorship hebrews": {
            "positions": [
                {"tradition": "evangelical", "view": "Pauline association but not direct authorship", "percentage": 60},
                {"tradition": "evangelical", "view": "Pauline authorship", "percentage": 20},
                {"tradition": "evangelical", "view": "Apollos authorship", "percentage": 10},
                {"tradition": "evangelical", "view": "Other candidate", "percentage": 10},
                {"tradition": "critical", "view": "Unknown author", "percentage": 70},
                {"tradition": "critical", "view": "Apollos authorship", "percentage": 15},
                {"tradition": "critical", "view": "Priscilla authorship", "percentage": 10},
                {"tradition": "critical", "view": "Other candidate", "percentage": 5},
                {"tradition": "catholic", "view": "Pauline association but not direct authorship", "percentage": 65},
                {"tradition": "catholic", "view": "Unknown author", "percentage": 25},
                {"tradition": "catholic", "view": "Other candidate", "percentage": 10}
            ],
            "consensus_level": "Medium",
            "dominant_view": "Pauline association but not direct authorship",
            "factors": ["Different methodological approaches", "Limited historical evidence"]
        },
        "synoptic problem": {
            "positions": [
                {"tradition": "evangelical", "view": "Two-Source hypothesis (Mark First, Q)", "percentage": 65},
                {"tradition": "evangelical", "view": "Matthew first", "percentage": 20},
                {"tradition": "evangelical", "view": "Farrer hypothesis (no Q)", "percentage": 10},
                {"tradition": "evangelical", "view": "Other hypothesis", "percentage": 5},
                {"tradition": "critical", "view": "Two-Source hypothesis (Mark First, Q)", "percentage": 80},
                {"tradition": "critical", "view": "Farrer hypothesis (no Q)", "percentage": 15},
                {"tradition": "critical", "view": "Other hypothesis", "percentage": 5},
                {"tradition": "catholic", "view": "Two-Source hypothesis (Mark First, Q)", "percentage": 75},
                {"tradition": "catholic", "view": "Matthew first", "percentage": 15},
                {"tradition": "catholic", "view": "Other hypothesis", "percentage": 10}
            ],
            "consensus_level": "High",
            "dominant_view": "Two-Source hypothesis (Mark First, Q)",
            "factors": ["Text-critical evidence", "Linguistic patterns", "Editorial tendencies"]
        },
        "jesus divinity": {
            "positions": [
                {"tradition": "evangelical", "view": "Full divinity explicitly taught", "percentage": 95},
                {"tradition": "evangelical", "view": "Other view", "percentage": 5},
                {"tradition": "critical", "view": "Later theological development", "percentage": 60},
                {"tradition": "critical", "view": "Limited divine claims in earliest traditions", "percentage": 30},
                {"tradition": "critical", "view": "Other view", "percentage": 10},
                {"tradition": "catholic", "view": "Full divinity explicitly taught", "percentage": 90},
                {"tradition": "catholic", "view": "Other view", "percentage": 10},
                {"tradition": "orthodox", "view": "Full divinity explicitly taught", "percentage": 95},
                {"tradition": "orthodox", "view": "Other view", "percentage": 5}
            ],
            "consensus_level": "Low",
            "dominant_view": None,
            "factors": ["Theological commitments", "Historical-critical methodologies", "Source dating and authenticity"]
        }
    }
    
    # Check if we have data for this topic
    topic_match = None
    for key, data in known_topics.items():
        if key in topic_lower:
            topic_match = key
            break
    
    if topic_match:
        # Get all positions for included traditions
        positions = [p for p in known_topics[topic_match]["positions"] 
                    if p["tradition"] in include_traditions]
        
        result["positions"] = positions
        result["consensus_level"] = known_topics[topic_match]["consensus_level"]
        result["dominant_view"] = known_topics[topic_match]["dominant_view"]
        result["methodological_factors"] = known_topics[topic_match]["factors"]
        
        # Calculate confidence based on agreement and tradition diversity
        if positions:
            # Extract unique views
            views = set(p["view"] for p in positions)
            
            # Calculate agreement score - percentage of scholars holding dominant view
            if result["dominant_view"]:
                dominant_positions = [p for p in positions if p["view"] == result["dominant_view"]]
                dominant_percentage = sum(p["percentage"] for p in dominant_positions) / len(include_traditions)
                agreement_score = dominant_percentage / 100
            else:
                agreement_score = 0.5  # Medium agreement
            
            # Calculate tradition diversity - ratio of included to total traditions
            tradition_diversity = len(set(p["tradition"] for p in positions)) / len(SCHOLARLY_TRADITIONS)
            
            # Calculate confidence
            result["confidence"] = (agreement_score * 0.7) + (tradition_diversity * 0.3)
            
            # Set alternative views
            if result["dominant_view"]:
                result["alternative_views"] = list(views - {result["dominant_view"]})
            else:
                result["alternative_views"] = list(views)[:3]  # Limit to top 3 alternative views
    else:
        # For unknown topics, provide a general assessment based on type of question
        if "authorship" in topic_lower or "who wrote" in topic_lower:
            result["consensus_level"] = "Unknown"
            result["methodological_factors"] = ["Historical attribution evidence", "Stylistic analysis", "Theological content comparison"]
            result["explanation"] = "Authorship questions typically involve analysis of internal evidence (style, theology) and external attestation (early church references)."
        
        elif "meaning" in topic_lower or "interpret" in topic_lower:
            result["consensus_level"] = "Unknown"
            result["methodological_factors"] = ["Context analysis", "Linguistic factors", "Historical background", "Canonical considerations"]
            result["explanation"] = "Interpretive questions are analyzed using linguistic, historical, literary, and theological approaches, often with varying levels of agreement across traditions."
        
        elif "date" in topic_lower or "when" in topic_lower:
            result["consensus_level"] = "Unknown"
            result["methodological_factors"] = ["Historical references", "Archaeological context", "Literary dependencies", "Theological development"]
            result["explanation"] = "Dating questions rely on internal references, external historical correlations, and text-critical considerations."
        
        else:
            result["explanation"] = "This specific topic is not indexed in the consensus database. For an accurate assessment, please consult recent academic literature on this subject."
    
    return result

def identify_academic_positions(topic: str, 
                               sources: Optional[List[Dict[str, str]]] = None) -> pd.DataFrame:
    """
    Identify and categorize academic positions on a biblical or theological topic.
    
    Args:
        topic: The topic or question to analyze
        sources: Optional list of dictionaries with source information
        
    Returns:
        DataFrame with academic positions
        
    Example:
        >>> # Identify positions on the historical Jesus
        >>> positions = identify_academic_positions(
        ...     "What can we know about the historical Jesus?",
        ...     sources=[
        ...         {"author": "E.P. Sanders", "work": "The Historical Figure of Jesus", "quote": "..."},
        ...         {"author": "N.T. Wright", "work": "Jesus and the Victory of God", "quote": "..."}
        ...     ]
        ... )
    """
    # Initialize columns for DataFrame
    columns = ["scholar", "tradition", "position", "confidence", "key_claims", "methodology", "sources"]
    
    # This is a placeholder implementation that would be replaced by a real
    # analysis of provided sources and scholarly literature in a production system
    
    # If sources are provided, extract information from them
    if sources:
        rows = []
        
        for source in sources:
            # Extract author name
            author = source.get("author", "Unknown")
            
            # Attempt to match author to known tradition
            tradition = "unknown"
            for trad_name, trad_info in SCHOLARLY_TRADITIONS.items():
                if author in trad_info["representatives"]:
                    tradition = trad_name
                    break
            
            # Extract position from quote (simplified approach)
            quote = source.get("quote", "")
            position = "Position not extracted from quote"
            
            if quote:
                # Very simple position extraction - first sentence of quote
                sentences = sent_tokenize(quote)
                if sentences:
                    position = sentences[0]
            
            # Add placeholder values for other columns
            row = {
                "scholar": author,
                "tradition": tradition,
                "position": position,
                "confidence": 0.7,  # Default medium-high confidence
                "key_claims": [],
                "methodology": "Not analyzed",
                "sources": source.get("work", "Unknown source")
            }
            
            rows.append(row)
        
        if rows:
            return pd.DataFrame(rows, columns=columns)
    
    # For demonstration, we'll handle some common topics with predefined data
    topic_lower = topic.lower()
    
    # Map of topics to known positions
    known_topics = {
        "historical jesus": [
            {"scholar": "E.P. Sanders", "tradition": "critical", "position": "Jesus was an apocalyptic prophet who proclaimed the kingdom of God", 
             "confidence": 0.9, "key_claims": ["Jewish context", "Kingdom of God", "Eschatological expectations"], 
             "methodology": "Historical-critical", "sources": "The Historical Figure of Jesus (1993)"},
            {"scholar": "John Dominic Crossan", "tradition": "critical", "position": "Jesus was a peasant Jewish Cynic philosopher", 
             "confidence": 0.8, "key_claims": ["Social context", "Wisdom teachings", "Kingdom as present reality"], 
             "methodology": "Historical-cultural", "sources": "The Historical Jesus (1991)"},
            {"scholar": "N.T. Wright", "tradition": "evangelical", "position": "Jesus was a Jewish prophet announcing God's kingdom and his own role within it", 
             "confidence": 0.9, "key_claims": ["Jewish context", "Kingdom of God", "Messianic identity"], 
             "methodology": "Historical-narrative", "sources": "Jesus and the Victory of God (1996)"},
            {"scholar": "Luke Timothy Johnson", "tradition": "catholic", "position": "The 'historical Jesus' cannot be separated from the 'Christ of faith'", 
             "confidence": 0.7, "key_claims": ["Limits of historical method", "Resurrection experience", "Living Lord"], 
             "methodology": "Historical-theological", "sources": "The Real Jesus (1996)"}
        ],
        "paul conversion": [
            {"scholar": "James D.G. Dunn", "tradition": "critical", "position": "Paul's 'conversion' was a prophetic call within Judaism, not a conversion away from Judaism", 
             "confidence": 0.8, "key_claims": ["Remained Jewish", "Prophetic calling", "Gentile mission"], 
             "methodology": "Historical-exegetical", "sources": "The Theology of Paul the Apostle (1998)"},
            {"scholar": "N.T. Wright", "tradition": "evangelical", "position": "Paul's conversion was a radical reframing of his Jewish worldview around Jesus as Messiah", 
             "confidence": 0.9, "key_claims": ["Jewish context", "Christological shift", "Apocalyptic revelation"], 
             "methodology": "Historical-narrative", "sources": "Paul: A Biography (2018)"},
            {"scholar": "Alan Segal", "tradition": "jewish", "position": "Paul's conversion resembled Jewish mystical transformation experiences", 
             "confidence": 0.7, "key_claims": ["Jewish mysticism", "Throne visions", "Transformation"], 
             "methodology": "Historical-comparative", "sources": "Paul the Convert (1990)"},
            {"scholar": "Douglas Campbell", "tradition": "critical", "position": "Paul's Damascus road experience was an apocalyptic unveiling of God's nature in Christ", 
             "confidence": 0.8, "key_claims": ["Apocalyptic revelation", "Theological reframing", "Divine disclosure"], 
             "methodology": "Theological-critical", "sources": "The Deliverance of God (2009)"}
        ],
        "revelation authorship": [
            {"scholar": "Craig Koester", "tradition": "critical", "position": "Revelation was written by a Christian prophet named John, not the apostle", 
             "confidence": 0.8, "key_claims": ["Separate author", "Prophetic authority", "Stylistic differences"], 
             "methodology": "Historical-critical", "sources": "Revelation and the End of All Things (2001)"},
            {"scholar": "Grant Osborne", "tradition": "evangelical", "position": "John the Apostle wrote Revelation, with possible use of an amanuensis", 
             "confidence": 0.7, "key_claims": ["Apostolic authorship", "Early church testimony", "Theological consistency"], 
             "methodology": "Historical-grammatical", "sources": "Revelation (Baker Exegetical Commentary, 2002)"},
            {"scholar": "David Aune", "tradition": "critical", "position": "Revelation likely underwent several stages of composition by different authors", 
             "confidence": 0.6, "key_claims": ["Multiple authors", "Redaction history", "Composite text"], 
             "methodology": "Historical-critical", "sources": "Revelation (Word Biblical Commentary, 1997)"},
            {"scholar": "Richard Bauckham", "tradition": "evangelical", "position": "A single author named John who was not the apostle but a Christian prophet", 
             "confidence": 0.8, "key_claims": ["Literary unity", "Jewish apocalyptic genre", "Prophetic authority"], 
             "methodology": "Literary-historical", "sources": "The Theology of the Book of Revelation (1993)"}
        ]
    }
    
    # Check if we have data for this topic
    topic_match = None
    for key, data in known_topics.items():
        if key in topic_lower:
            topic_match = key
            break
    
    if topic_match:
        return pd.DataFrame(known_topics[topic_match], columns=columns)
    else:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=columns)

def consensus_evolution(topic: str,
                       time_period: str = "20th-21st century",
                       traditions: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Track the evolution of scholarly consensus on a theological or biblical topic over time.
    
    Args:
        topic: The topic or question to analyze
        time_period: Time period to analyze ("20th-21st century", "early church", etc.)
        traditions: Optional list of traditions to include
        
    Returns:
        Dictionary with consensus evolution data
        
    Example:
        >>> # Track evolution of views on the synoptic problem
        >>> evolution = consensus_evolution(
        ...     "The synoptic problem",
        ...     time_period="19th-21st century",
        ...     traditions=["critical", "evangelical"]
        ... )
    """
    # Set default traditions if not specified
    if traditions is None:
        traditions = list(SCHOLARLY_TRADITIONS.keys())
    
    # Initialize result
    result = {
        "topic": topic,
        "time_period": time_period,
        "traditions": traditions,
        "evolution_pattern": None,
        "timeline": [],
        "turning_points": [],
        "current_consensus": None,
        "factors_of_change": []
    }
    
    # This is a placeholder implementation that would be replaced by a real
    # analysis of historical scholarly positions in a production system
    
    # For demonstration, we'll handle some common topics with predefined data
    topic_lower = topic.lower()
    
    # Map of topics to known consensus evolution data
    known_topics = {
        "synoptic problem": {
            "evolution_pattern": "Convergence",
            "timeline": [
                {"period": "Early 19th century", "dominant_view": "Independence hypothesis/Augustinian hypothesis (Matthew first)", 
                 "percentage": 80, "key_scholars": ["Traditional view"]},
                {"period": "Late 19th century", "dominant_view": "Two-Source hypothesis (Mark first, Q source)", 
                 "percentage": 40, "key_scholars": ["H.J. Holtzmann", "B.H. Streeter"]},
                {"period": "Mid 20th century", "dominant_view": "Two-Source hypothesis", 
                 "percentage": 70, "key_scholars": ["Rudolf Bultmann", "Vincent Taylor"]},
                {"period": "Late 20th century", "dominant_view": "Two-Source hypothesis with variations", 
                 "percentage": 75, "key_scholars": ["Graham Stanton", "E.P. Sanders"]},
                {"period": "Early 21st century", "dominant_view": "Two-Source hypothesis with challenges from Farrer hypothesis", 
                 "percentage": 65, "key_scholars": ["Mark Goodacre", "John Kloppenborg"]}
            ],
            "turning_points": [
                {"period": "1830s", "event": "Proposal of Marcan priority by Karl Lachmann"},
                {"period": "1890s", "event": "Formulation of Two-Source hypothesis by H.J. Holtzmann"},
                {"period": "1920s", "event": "Refinement of Two-Source hypothesis by B.H. Streeter"},
                {"period": "1970s-80s", "event": "Revival of Farrer hypothesis (Mark without Q) by Michael Goulder"}
            ],
            "current_consensus": "Two-Source hypothesis remains dominant but with significant minority views",
            "factors_of_change": ["Text-critical advances", "Literary analysis", "Methodological developments"]
        },
        "pauline authorship": {
            "evolution_pattern": "Divergence",
            "timeline": [
                {"period": "Pre-18th century", "dominant_view": "Traditional authorship (13 epistles)", 
                 "percentage": 95, "key_scholars": ["Church tradition"]},
                {"period": "19th century", "dominant_view": "Disputed authorship (questions about Pastorals and others)", 
                 "percentage": 60, "key_scholars": ["F.C. Baur", "Early critical scholars"]},
                {"period": "Early 20th century", "dominant_view": "Core 7/disputed 6 model emerging", 
                 "percentage": 50, "key_scholars": ["Adolf Jülicher", "Alfred Loisy"]},
                {"period": "Mid-late 20th century", "dominant_view": "7 undisputed/6 disputed or pseudepigraphical", 
                 "percentage": 70, "key_scholars": ["Raymond Brown", "J.D.G. Dunn"]},
                {"period": "Early 21st century", "dominant_view": "Split between critical (7 authentic) and evangelical (13 authentic) views", 
                 "percentage": 65, "key_scholars": ["Luke Timothy Johnson", "D.A. Carson"]}
            ],
            "turning_points": [
                {"period": "1830s", "event": "F.C. Baur's proposal of only 4 authentic letters"},
                {"period": "1900-1930", "event": "Critical consensus forming around 7 authentic letters"},
                {"period": "1950s-60s", "event": "Development of more nuanced pseudepigraphy theories"},
                {"period": "1980s-present", "event": "Growing tradition-based divergence in scholarship"}
            ],
            "current_consensus": "Strong tradition-based divide with most critical scholars accepting 7 authentic letters and many evangelical scholars defending 13",
            "factors_of_change": ["Historical-critical method", "Linguistic analysis", "Theological considerations", "Different views of pseudepigraphy"]
        },
        "historical jesus": {
            "evolution_pattern": "Cyclical",
            "timeline": [
                {"period": "19th century", "dominant_view": "Liberal 'Lives of Jesus' (ethical teacher)", 
                 "percentage": 70, "key_scholars": ["D.F. Strauss", "Ernest Renan"]},
                {"period": "Early 20th century", "dominant_view": "Eschatological prophet", 
                 "percentage": 65, "key_scholars": ["Albert Schweitzer", "Johannes Weiss"]},
                {"period": "Mid 20th century", "dominant_view": "Existential teacher/demythologized Christ", 
                 "percentage": 50, "key_scholars": ["Rudolf Bultmann", "Form critics"]},
                {"period": "Late 20th century", "dominant_view": "Jewish apocalyptic prophet (3rd Quest)", 
                 "percentage": 60, "key_scholars": ["E.P. Sanders", "John P. Meier"]},
                {"period": "Early 21st century", "dominant_view": "Jewish prophet in social context with multiple interpretive approaches", 
                 "percentage": 55, "key_scholars": ["N.T. Wright", "Bart Ehrman", "Dale Allison"]}
            ],
            "turning_points": [
                {"period": "1835-1906", "event": "First Quest for the Historical Jesus"},
                {"period": "1906", "event": "Albert Schweitzer's critique of the First Quest"},
                {"period": "1953-1970s", "event": "Second Quest (New Quest) for the Historical Jesus"},
                {"period": "1980s-present", "event": "Third Quest emphasizing Jewish context"}
            ],
            "current_consensus": "Jesus was a Jewish apocalyptic prophet with both Jewish context and enigmatic elements that resist easy categorization",
            "factors_of_change": ["Archaeological discoveries", "Dead Sea Scrolls", "Social-scientific methods", "Jewish-Christian dialogue"]
        }
    }
    
    # Check if we have data for this topic
    topic_match = None
    for key, data in known_topics.items():
        if key in topic_lower:
            topic_match = key
            break
    
    if topic_match:
        # Add data to result
        for key, value in known_topics[topic_match].items():
            result[key] = value
    else:
        # Provide generic evolution pattern based on type of question
        if "authorship" in topic_lower or "who wrote" in topic_lower:
            result["evolution_pattern"] = "Typical divergence from traditional attributions"
            result["timeline"] = [
                {"period": "Pre-critical era", "dominant_view": "Traditional authorship attribution", "percentage": 90},
                {"period": "Early historical-critical era", "dominant_view": "Challenges to traditional attribution", "percentage": 60},
                {"period": "Modern era", "dominant_view": "Tradition-dependent views", "percentage": 50}
            ]
            result["factors_of_change"] = ["Rise of historical-critical methods", "Linguistic analysis advances", "Archaeological context"]
            
        elif "meaning" in topic_lower or "interpret" in topic_lower:
            result["evolution_pattern"] = "Complex interaction of methods and traditions"
            result["timeline"] = [
                {"period": "Pre-critical era", "dominant_view": "Traditional/ecclesiastical interpretation", "percentage": 85},
                {"period": "Critical era", "dominant_view": "Historical-grammatical approaches", "percentage": 70},
                {"period": "Modern era", "dominant_view": "Multiple hermeneutical approaches", "percentage": 55}
            ]
            result["factors_of_change"] = ["Hermeneutical developments", "Postmodern critiques", "Interdisciplinary approaches"]
            
        elif "date" in topic_lower or "when" in topic_lower:
            result["evolution_pattern"] = "Refinement through evidence"
            result["timeline"] = [
                {"period": "Pre-critical era", "dominant_view": "Traditional dating", "percentage": 90},
                {"period": "Critical era", "dominant_view": "Revised critical dating", "percentage": 65},
                {"period": "Modern era", "dominant_view": "Evidence-based range of dates", "percentage": 70}
            ]
            result["factors_of_change"] = ["Archaeological discoveries", "Comparative literary analysis", "Historical correlations"]
            
        else:
            result["explanation"] = "This specific topic is not indexed in the consensus evolution database. For an accurate historical assessment, please consult specialized academic literature on this subject."
    
    return result

def expert_distribution(topic: str,
                       view: str,
                       filter_by: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Analyze the distribution of expert opinions on a specific view or position.
    
    Args:
        topic: The topic or question to analyze
        view: The specific view or position to analyze
        filter_by: Optional dictionary to filter experts (by tradition, era, etc.)
        
    Returns:
        DataFrame with expert distribution data
        
    Example:
        >>> # Analyze support for Q document hypothesis
        >>> distribution = expert_distribution(
        ...     "Synoptic Problem",
        ...     "Two-Source Hypothesis (Q)",
        ...     filter_by={"traditions": ["critical", "evangelical"], "era": "contemporary"}
        ... )
    """
    # Initialize columns for DataFrame
    columns = ["scholar", "tradition", "view", "confidence", "reasoning", "key_work", "year"]
    
    # This is a placeholder implementation that would be replaced by a real
    # analysis of expert opinions in a production system
    
    # Set default filter if not provided
    if filter_by is None:
        filter_by = {}
    
    # For demonstration, we'll handle some common topics with predefined data
    topic_lower = topic.lower()
    view_lower = view.lower()
    
    # Map of topics and views to known expert distributions
    known_distributions = {
        "synoptic problem": {
            "two-source": [
                {"scholar": "James D.G. Dunn", "tradition": "critical", "view": "Two-Source Hypothesis", 
                 "confidence": 0.9, "reasoning": "Markan priority and Q best explain the evidence", 
                 "key_work": "Christianity in the Making, Vol. 1", "year": 2003},
                {"scholar": "John S. Kloppenborg", "tradition": "critical", "view": "Two-Source Hypothesis", 
                 "confidence": 0.95, "reasoning": "Q can be stratified into distinct layers of tradition", 
                 "key_work": "The Formation of Q", "year": 1987},
                {"scholar": "Christopher M. Tuckett", "tradition": "critical", "view": "Two-Source Hypothesis", 
                 "confidence": 0.9, "reasoning": "Minor agreements don't invalidate two-source model", 
                 "key_work": "Q and the History of Early Christianity", "year": 1996},
                {"scholar": "Darrell L. Bock", "tradition": "evangelical", "view": "Two-Source Hypothesis", 
                 "confidence": 0.8, "reasoning": "Best explains the synoptic data while allowing for reliability", 
                 "key_work": "Luke (BECNT)", "year": 1994},
                {"scholar": "Scot McKnight", "tradition": "evangelical", "view": "Two-Source Hypothesis", 
                 "confidence": 0.7, "reasoning": "Most plausible explanation of synoptic relationships", 
                 "key_work": "Interpreting the Synoptic Gospels", "year": 1988}
            ],
            "farrer": [
                {"scholar": "Mark Goodacre", "tradition": "critical", "view": "Farrer Hypothesis (No Q)", 
                 "confidence": 0.9, "reasoning": "Luke used Matthew directly, eliminating need for Q", 
                 "key_work": "The Case Against Q", "year": 2002},
                {"scholar": "Michael Goulder", "tradition": "critical", "view": "Farrer Hypothesis (No Q)", 
                 "confidence": 0.95, "reasoning": "Luke's redactional creativity explains double tradition", 
                 "key_work": "Luke: A New Paradigm", "year": 1989},
                {"scholar": "Eric Eve", "tradition": "critical", "view": "Farrer Hypothesis (No Q)", 
                 "confidence": 0.8, "reasoning": "Problems with Q reconstruction are resolved without Q", 
                 "key_work": "Behind the Gospels", "year": 2013},
                {"scholar": "John C. Poirier", "tradition": "critical", "view": "Farrer Hypothesis (No Q)", 
                 "confidence": 0.85, "reasoning": "Q hypothesis creates more problems than it solves", 
                 "key_work": "Articles in JBL and NTS", "year": 2008}
            ]
        },
        "pastoral epistles": {
            "non-pauline": [
                {"scholar": "Raymond E. Brown", "tradition": "catholic", "view": "Non-Pauline authorship", 
                 "confidence": 0.9, "reasoning": "Linguistic and theological differences from undisputed letters", 
                 "key_work": "An Introduction to the New Testament", "year": 1997},
                {"scholar": "Luke Timothy Johnson", "tradition": "catholic", "view": "Non-Pauline authorship", 
                 "confidence": 0.7, "reasoning": "Different vocabulary but theology compatible with Paul", 
                 "key_work": "The First and Second Letters to Timothy", "year": 2001},
                {"scholar": "Bart D. Ehrman", "tradition": "critical", "view": "Non-Pauline authorship", 
                 "confidence": 0.95, "reasoning": "Clear pseudepigraphy reflecting later church situation", 
                 "key_work": "Forgery and Counterforgery", "year": 2012},
                {"scholar": "Margaret Y. MacDonald", "tradition": "critical", "view": "Non-Pauline authorship", 
                 "confidence": 0.9, "reasoning": "Reflects post-Pauline community development", 
                 "key_work": "The Pauline Churches", "year": 1988},
                {"scholar": "I. Howard Marshall", "tradition": "evangelical", "view": "Non-Pauline authorship", 
                 "confidence": 0.7, "reasoning": "Pauline legacy through authorized secretary/co-author", 
                 "key_work": "ICC Commentary on the Pastoral Epistles", "year": 1999}
            ],
            "pauline": [
                {"scholar": "William D. Mounce", "tradition": "evangelical", "view": "Pauline authorship", 
                 "confidence": 0.8, "reasoning": "Differences explicable through secretary, audience, and context", 
                 "key_work": "Pastoral Epistles (WBC)", "year": 2000},
                {"scholar": "Andreas Köstenberger", "tradition": "evangelical", "view": "Pauline authorship", 
                 "confidence": 0.85, "reasoning": "Historical problems with pseudepigraphy theory", 
                 "key_work": "Commentary on 1-2 Timothy & Titus", "year": 2021},
                {"scholar": "Donald Guthrie", "tradition": "evangelical", "view": "Pauline authorship", 
                 "confidence": 0.9, "reasoning": "External evidence and weakness of counter-arguments", 
                 "key_work": "The Pastoral Epistles (TNTC)", "year": 1990},
                {"scholar": "George W. Knight III", "tradition": "evangelical", "view": "Pauline authorship", 
                 "confidence": 0.9, "reasoning": "Linguistic objections overstated, consistent with late Paul", 
                 "key_work": "The Pastoral Epistles (NIGTC)", "year": 1992}
            ]
        }
    }
    
    # Find matching distribution
    data = None
    for topic_key, views in known_distributions.items():
        if topic_key in topic_lower:
            for view_key, experts in views.items():
                if view_key in view_lower:
                    data = experts
                    break
            break
    
    if data:
        # Apply filters if specified
        if filter_by:
            filtered_data = data.copy()
            
            # Filter by tradition
            if "traditions" in filter_by:
                filtered_data = [d for d in filtered_data if d["tradition"] in filter_by["traditions"]]
            
            # Filter by era/time period
            if "era" in filter_by and filter_by["era"] == "contemporary":
                # Consider contemporary as past 30 years
                current_year = 2025  # as of the model's knowledge cutoff
                filtered_data = [d for d in filtered_data if d["year"] > current_year - 30]
            
            # Filter by confidence level
            if "min_confidence" in filter_by:
                filtered_data = [d for d in filtered_data if d["confidence"] >= filter_by["min_confidence"]]
            
            return pd.DataFrame(filtered_data, columns=columns)
        else:
            return pd.DataFrame(data, columns=columns)
    else:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=columns)

def track_theological_trends(topic: str,
                            start_year: int = 1900,
                            end_year: int = 2024,
                            interval: int = 20) -> Dict[str, Any]:
    """
    Track trends in theological positions on a topic over time.
    
    Args:
        topic: The topic to analyze
        start_year: Starting year for trend analysis
        end_year: Ending year for trend analysis
        interval: Year interval for data points
        
    Returns:
        Dictionary with theological trend data
        
    Example:
        >>> # Track trends in interpretation of Romans 9-11
        >>> trends = track_theological_trends(
        ...     "Interpretation of Romans 9-11",
        ...     start_year=1950,
        ...     end_year=2020,
        ...     interval=10
        ... )
    """
    # Ensure valid time range
    if start_year >= end_year:
        return {"error": "Start year must be before end year"}
    
    # Calculate year intervals
    years = list(range(start_year, end_year + 1, interval))
    if years[-1] != end_year:
        years.append(end_year)
    
    # Initialize result
    result = {
        "topic": topic,
        "time_range": f"{start_year}-{end_year}",
        "years": years,
        "positions": [],
        "trends": [],
        "tradition_shifts": {},
        "key_publications": []
    }
    
    # This is a placeholder implementation that would be replaced by a real
    # analysis of theological literature trends in a production system
    
    # For demonstration, we'll handle some common topics with predefined data
    topic_lower = topic.lower()
    
    # Map of topics to known trends
    known_trends = {
        "romans 9-11": {
            "positions": [
                {"name": "Double Predestination", "description": "God actively predestines both salvation and damnation"},
                {"name": "Single Predestination", "description": "God predestines salvation but not damnation"},
                {"name": "Corporate Election", "description": "Election concerns groups not individuals"},
                {"name": "New Perspective", "description": "Focus on Israel's role in salvation history, not individual election"}
            ],
            "trends": [
                {"position": "Double Predestination", "values": [80, 70, 55, 40, 30, 25, 20]},
                {"position": "Single Predestination", "values": [15, 20, 30, 35, 30, 25, 20]},
                {"position": "Corporate Election", "values": [5, 10, 15, 20, 25, 30, 30]},
                {"position": "New Perspective", "values": [0, 0, 0, 5, 15, 20, 30]}
            ],
            "tradition_shifts": {
                "evangelical": "From individual predestination to more diverse views including corporate election",
                "critical": "Increasing emphasis on social and historical context of Israel",
                "catholic": "Greater emphasis on universal salvation possibilities"
            },
            "key_publications": [
                {"year": 1918, "author": "Karl Barth", "work": "Romans Commentary (1st edition)", "impact": "Challenged liberal Protestant readings"},
                {"year": 1957, "author": "John Murray", "work": "The Epistle to the Romans", "impact": "Reformed double predestination interpretation"},
                {"year": 1980, "author": "E.P. Sanders", "work": "Paul and Palestinian Judaism", "impact": "Reframed Paul in Jewish context"},
                {"year": 1994, "author": "N.T. Wright", "work": "The Climax of the Covenant", "impact": "Advanced New Perspective reading of Romans 9-11"},
                {"year": 2004, "author": "John Piper", "work": "The Justification of God", "impact": "Defense of Calvinist reading of Romans 9"},
                {"year": 2010, "author": "Michael Gorman", "work": "Inhabiting the Cruciform God", "impact": "Participationist reading"}
            ]
        },
        "atonement": {
            "positions": [
                {"name": "Penal Substitution", "description": "Christ bore the punishment for sin in our place"},
                {"name": "Christus Victor", "description": "Christ's death and resurrection defeated evil powers"},
                {"name": "Moral Influence", "description": "Christ's death exemplifies God's love to inspire moral change"},
                {"name": "Recapitulation", "description": "Christ retraces and redeems human experience"}
            ],
            "trends": [
                {"position": "Penal Substitution", "values": [40, 55, 70, 75, 65, 55, 45]},
                {"position": "Christus Victor", "values": [30, 20, 15, 10, 15, 25, 30]},
                {"position": "Moral Influence", "values": [25, 20, 10, 10, 15, 15, 15]},
                {"position": "Recapitulation", "values": [5, 5, 5, 5, 5, 5, 10]}
            ],
            "tradition_shifts": {
                "evangelical": "Strong emphasis on penal substitution, though slightly decreasing",
                "critical": "Shift toward Christus Victor and away from substitutionary models",
                "catholic": "Renewed interest in recapitulation and broader models",
                "orthodox": "Consistent emphasis on Christus Victor and recapitulation"
            },
            "key_publications": [
                {"year": 1931, "author": "Gustaf Aulén", "work": "Christus Victor", "impact": "Revived interest in early church 'classic' view"},
                {"year": 1955, "author": "John Stott", "work": "The Cross of Christ", "impact": "Articulated penal substitution for evangelicals"},
                {"year": 1974, "author": "Jürgen Moltmann", "work": "The Crucified God", "impact": "Social and political dimensions of atonement"},
                {"year": 1986, "author": "J.I. Packer", "work": "What Did the Cross Achieve?", "impact": "Defense of penal substitution"},
                {"year": 2006, "author": "Joel Green & Mark Baker", "work": "Recovering the Scandal of the Cross", "impact": "Multi-faceted view of atonement"},
                {"year": 2011, "author": "Fleming Rutledge", "work": "The Crucifixion", "impact": "Integration of substitution and Christus Victor"}
            ]
        },
        "genesis 1-11": {
            "positions": [
                {"name": "Young Earth Creationism", "description": "Literal 6-day creation, earth <10,000 years old"},
                {"name": "Old Earth Creationism", "description": "Day-age or gap interpretations, compatible with ancient earth"},
                {"name": "Theistic Evolution", "description": "Evolution as God's method of creation"},
                {"name": "Literary/Theological Reading", "description": "Focus on theological message, not scientific details"}
            ],
            "trends": [
                {"position": "Young Earth Creationism", "values": [40, 45, 35, 30, 35, 30, 25]},
                {"position": "Old Earth Creationism", "values": [30, 30, 35, 40, 35, 30, 25]},
                {"position": "Theistic Evolution", "values": [20, 15, 20, 20, 20, 25, 30]},
                {"position": "Literary/Theological Reading", "values": [10, 10, 10, 10, 10, 15, 20]}
            ],
            "tradition_shifts": {
                "evangelical": "Growing diversity from YEC dominance to acceptance of multiple views",
                "critical": "Consistent emphasis on literary and theological readings",
                "catholic": "Movement from concordist models toward theistic evolution"
            },
            "key_publications": [
                {"year": 1923, "author": "William Jennings Bryan", "work": "In His Image", "impact": "Defense of creation against evolution"},
                {"year": 1961, "author": "John Whitcomb & Henry Morris", "work": "The Genesis Flood", "impact": "Launched modern young-earth creationism"},
                {"year": 1984, "author": "Hugh Ross", "work": "The Fingerprint of God", "impact": "Old-earth creationism for evangelicals"},
                {"year": 1992, "author": "Phyllis Trible", "work": "God and the Rhetoric of Sexuality", "impact": "Literary-feminist reading of Genesis"},
                {"year": 2003, "author": "Francis Collins", "work": "The Language of God", "impact": "Evangelical case for theistic evolution"},
                {"year": 2009, "author": "John Walton", "work": "The Lost World of Genesis One", "impact": "Ancient cosmology interpretation"}
            ]
        }
    }
    
    # Check if we have data for this topic
    topic_match = None
    for key, data in known_trends.items():
        if key in topic_lower:
            topic_match = key
            break
    
    if topic_match:
        # Add positions and trends
        result["positions"] = known_trends[topic_match]["positions"]
        
        # Adjust trend data to match requested years
        trends = []
        for position_trend in known_trends[topic_match]["trends"]:
            # Create interpolated values for the requested years
            original_years = list(range(1900, 2021, 20))
            original_values = position_trend["values"]
            
            # Use simple linear interpolation to estimate values
            interpolated_values = []
            for year in years:
                # Find position in original timeline
                if year <= original_years[0]:
                    interpolated_values.append(original_values[0])
                elif year >= original_years[-1]:
                    interpolated_values.append(original_values[-1])
                else:
                    # Linear interpolation
                    for i in range(len(original_years) - 1):
                        if original_years[i] <= year < original_years[i+1]:
                            ratio = (year - original_years[i]) / (original_years[i+1] - original_years[i])
                            value = original_values[i] + ratio * (original_values[i+1] - original_values[i])
                            interpolated_values.append(round(value))
                            break
            
            trends.append({
                "position": position_trend["position"],
                "values": interpolated_values
            })
        
        result["trends"] = trends
        
        # Add tradition shifts and key publications
        result["tradition_shifts"] = known_trends[topic_match]["tradition_shifts"]
        
        # Filter publications to match the requested time range
        key_publications = [pub for pub in known_trends[topic_match]["key_publications"] 
                           if start_year <= pub["year"] <= end_year]
        result["key_publications"] = key_publications
    else:
        # Create generic trend data
        positions = [
            {"name": "Traditional View", "description": "Conservative or traditional interpretation"},
            {"name": "Moderate View", "description": "Balanced approach between traditional and critical"},
            {"name": "Critical View", "description": "Historical-critical or progressive interpretation"},
            {"name": "Alternative View", "description": "Emerging or minority interpretation"}
        ]
        
        # Generate made-up trend data
        trends = [
            {"position": "Traditional View", "values": [70, 60, 55, 45, 40, 35, 30]},
            {"position": "Moderate View", "values": [20, 25, 30, 35, 35, 35, 35]},
            {"position": "Critical View", "values": [10, 15, 15, 20, 25, 25, 25]},
            {"position": "Alternative View", "values": [0, 0, 0, 0, 0, 5, 10]}
        ]
        
        # Adjust to match requested years
        for trend in trends:
            original_years = list(range(1900, 2021, 20))
            original_values = trend["values"]
            
            # Use simple linear interpolation
            interpolated_values = []
            for year in years:
                if year <= original_years[0]:
                    interpolated_values.append(original_values[0])
                elif year >= original_years[-1]:
                    interpolated_values.append(original_values[-1])
                else:
                    for i in range(len(original_years) - 1):
                        if original_years[i] <= year < original_years[i+1]:
                            ratio = (year - original_years[i]) / (original_years[i+1] - original_years[i])
                            value = original_values[i] + ratio * (original_values[i+1] - original_values[i])
                            interpolated_values.append(round(value))
                            break
            
            trend["values"] = interpolated_values
        
        result["positions"] = positions
        result["trends"] = trends
        result["tradition_shifts"] = {
            "evangelical": "Gradual shift from exclusively traditional to more diverse views",
            "critical": "Consistent emphasis on historical and literary approaches",
            "catholic": "Movement toward greater integration of tradition and modern scholarship"
        }
        result["key_publications"] = []
    
    return result

def create_scholarly_position(tradition: str, position: str, 
                            sources: List[Dict[str, str]], confidence: float) -> Dict[str, Any]:
    """
    Create a structured record of a scholarly position on a biblical interpretation.
    
    Args:
        tradition: Scholarly tradition (e.g., 'evangelical', 'critical')
        position: Brief statement of the position
        sources: List of dictionaries with source information
        confidence: Confidence level from 0-1 representing scholarly consensus
        
    Returns:
        Dictionary representing the scholarly position
        
    Example:
        >>> # Create a position record for Johannine authorship
        >>> sources = [
        ...     {"author": "D.A. Carson", "work": "The Gospel According to John", "quote": "..."},
        ...     {"author": "Craig Blomberg", "work": "The Historical Reliability of John's Gospel", "quote": "..."}
        ... ]
        >>> position = create_scholarly_position(
        ...     "evangelical", 
        ...     "The apostle John wrote the Fourth Gospel",
        ...     sources,
        ...     0.8
        ... )
    """
    if tradition not in SCHOLARLY_TRADITIONS:
        # Create a custom tradition entry if not in predefined list
        tradition_data = {
            "description": f"Custom tradition: {tradition}",
            "representatives": []
        }
    else:
        tradition_data = SCHOLARLY_TRADITIONS[tradition]
    
    # Return structured position
    return {
        "tradition": tradition,
        "tradition_info": tradition_data,
        "position": position,
        "sources": sources,
        "confidence": confidence,
        "counter_positions": [],
        "related_debates": []
    }

def assess_consensus_level(positions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Assess the level of scholarly consensus across multiple positions.
    
    Args:
        positions: List of position dictionaries created with create_scholarly_position
        
    Returns:
        Dictionary with consensus assessment
        
    Example:
        >>> # Assess consensus on Johannine authorship
        >>> evangelical_position = create_scholarly_position(...)
        >>> critical_position = create_scholarly_position(...)
        >>> consensus = assess_consensus_level([evangelical_position, critical_position])
        >>> print(consensus["overall_consensus"])
    """
    # Count positions by tradition
    tradition_counts = Counter([p["tradition"] for p in positions])
    
    # Count confident positions vs. non-confident
    confidence_threshold = 0.7
    confident_positions = sum(1 for p in positions if p["confidence"] >= confidence_threshold)
    non_confident_positions = len(positions) - confident_positions
    
    # Calculate weighted confidence average
    if positions:
        avg_confidence = sum(p["confidence"] for p in positions) / len(positions)
    else:
        avg_confidence = 0
    
    # Analyze position text for agreement
    all_position_texts = [p["position"].lower() for p in positions]
    
    # Very simple text similarity check
    # (A more sophisticated approach would use NLP similarity metrics)
    similar_positions = 0
    for i, pos1 in enumerate(all_position_texts):
        for pos2 in all_position_texts[i+1:]:
            # Calculate simple word overlap
            words1 = set(word_tokenize(pos1))
            words2 = set(word_tokenize(pos2))
            
            if len(words1) > 0 and len(words2) > 0:
                overlap = len(words1.intersection(words2)) / min(len(words1), len(words2))
                
                if overlap > 0.5:  # More than 50% word overlap
                    similar_positions += 1
    
    # Calculate consensus metrics
    tradition_diversity = len(tradition_counts) / len(SCHOLARLY_TRADITIONS)
    position_similarity = similar_positions / (len(positions) * (len(positions) - 1) / 2) if len(positions) > 1 else 0
    
    # Determine overall consensus level
    # High consensus: High confidence, similar positions across diverse traditions
    # Medium consensus: Medium confidence or similarity, some tradition diversity
    # Low consensus: Low confidence, dissimilar positions, tradition homogeneity
    
    # Start with confidence-based score
    consensus_score = avg_confidence
    
    # Adjust for position similarity
    consensus_score += position_similarity * 0.3
    
    # Adjust for tradition diversity (more diverse = more significant consensus)
    if tradition_diversity > 0.5:
        consensus_score += 0.1
    elif tradition_diversity < 0.2:
        consensus_score -= 0.1
    
    # Convert to qualitative level
    if consensus_score >= 0.8:
        consensus_level = "High"
    elif consensus_score >= 0.5:
        consensus_level = "Medium"
    else:
        consensus_level = "Low"
    
    # Prepare result
    result = {
        "positions_count": len(positions),
        "traditions_represented": list(tradition_counts.keys()),
        "tradition_diversity": tradition_diversity,
        "average_confidence": avg_confidence,
        "position_similarity": position_similarity,
        "consensus_score": consensus_score,
        "overall_consensus": consensus_level,
        "explanation": "",
        "majority_position": None
    }
    
    # Identify majority position if it exists
    if position_similarity > 0.5 and avg_confidence > 0.6:
        # Find most common tradition
        if tradition_counts:
            majority_tradition = tradition_counts.most_common(1)[0][0]
            
            # Find highest confidence position from that tradition
            matching_positions = [p for p in positions if p["tradition"] == majority_tradition]
            if matching_positions:
                majority_position = max(matching_positions, key=lambda p: p["confidence"])
                result["majority_position"] = majority_position["position"]
    
    # Generate explanation
    if consensus_level == "High":
        result["explanation"] = "There is strong scholarly consensus across multiple traditions. "
    elif consensus_level == "Medium":
        result["explanation"] = "There is moderate scholarly consensus, though some disagreement exists. "
    else:
        result["explanation"] = "There is little scholarly consensus, with significant disagreements. "
    
    if result["majority_position"]:
        result["explanation"] += f"The majority position is: {result['majority_position']}"
    else:
        result["explanation"] += "No clear majority position could be identified."
    
    return result

def map_position_to_theological_framework(position_text: str) -> Dict[str, float]:
    """
    Map a position statement to known theological frameworks with confidence scores.
    
    Args:
        position_text: Text describing the position
        
    Returns:
        Dictionary mapping theological frameworks to confidence scores
        
    Example:
        >>> # Map a christological position
        >>> mapping = map_position_to_theological_framework(
        ...     "Jesus is of the same substance (homoousios) with the Father, fully divine"
        ... )
        >>> print(mapping)
        {'trinitarian': 0.95, 'unitarian': 0.05, 'adoptionist': 0.0}
    """
    results = {}
    position_lower = position_text.lower()
    
    # Process each theological category
    for category, positions in THEOLOGICAL_POSITIONS.items():
        category_scores = {}
        
        # Check each position in this category
        for position_name, position_data in positions.items():
            score = 0
            matched_concepts = []
            
            # Check for key concepts
            for concept in position_data["key_concepts"]:
                if concept.lower() in position_lower:
                    score += 1
                    matched_concepts.append(concept)
            
            # Normalize score (0-1)
            if position_data["key_concepts"]:
                normalized_score = score / len(position_data["key_concepts"])
            else:
                normalized_score = 0
            
            category_scores[position_name] = {
                "score": normalized_score,
                "matched_concepts": matched_concepts
            }
        
        # Add to results
        results[category] = category_scores
    
    # Flatten results for easier consumption
    flat_results = {}
    for category, positions in results.items():
        for position, data in positions.items():
            flat_results[f"{category}_{position}"] = data["score"]
    
    return flat_results

def get_scholarly_consensus(question: str, include_traditions: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Get scholarly consensus information for a biblical interpretation question.
    
    Args:
        question: The interpretive question to assess
        include_traditions: Optional list of traditions to include
        
    Returns:
        Dictionary with consensus information
        
    Example:
        >>> # Get consensus on Johannine authorship
        >>> consensus = get_scholarly_consensus(
        ...     "Did the Apostle John write the Gospel of John?",
        ...     include_traditions=["evangelical", "critical", "catholic"]
        ... )
    """
    result = {
        "question": question,
        "traditions_included": include_traditions if include_traditions else list(SCHOLARLY_TRADITIONS.keys()),
        "consensus_level": None,
        "positions": [],
        "explanation": "",
        "related_questions": []
    }
    
    # This is a placeholder implementation
    # In a real implementation, this would query a database of scholarly positions
    # or use NLP to analyze scholarly literature
    
    # For demonstration, we'll include some hard-coded responses for common questions
    question_lower = question.lower()
    
    # Pattern match to some common interpretive questions
    matched_question = False
    
    # Q1: Johannine authorship
    if (("who wrote" in question_lower and "john" in question_lower) or
       ("authorship" in question_lower and "john" in question_lower) or
       ("johannine authorship" in question_lower)):
        
        matched_question = True
        result["positions"] = [
            {
                "tradition": "evangelical",
                "position": "The Apostle John wrote the Fourth Gospel, possibly with some editorial assistance",
                "confidence": 0.85,
                "key_scholars": ["D.A. Carson", "Craig Blomberg", "Andreas Köstenberger"],
                "common_arguments": ["Internal evidence of eyewitness details", "Early church testimony", "Theological consistency"]
            },
            {
                "tradition": "critical",
                "position": "The Gospel of John was composed by a Johannine community, not directly by the Apostle John",
                "confidence": 0.75,
                "key_scholars": ["Raymond Brown", "Rudolf Bultmann", "Bart Ehrman"],
                "common_arguments": ["Stylistic differences from Revelation", "Late compositional date", "Theological development"]
            },
            {
                "tradition": "catholic",
                "position": "The Gospel has roots in Johannine testimony but reached final form through disciples",
                "confidence": 0.80,
                "key_scholars": ["Raymond Brown", "Francis J. Moloney", "Luke Timothy Johnson"],
                "common_arguments": ["Combination of apostolic source with later editing", "Church tradition", "Internal testimony"]
            }
        ]
        
        result["consensus_level"] = "Low"
        result["explanation"] = "There is significant disagreement between evangelical and critical scholars on Johannine authorship, with Catholic scholarship often taking a middle position. Evangelical scholars generally maintain apostolic authorship, while critical scholars favor a community composition model."
        
        result["related_questions"] = [
            "When was the Gospel of John written?",
            "What is the relationship between the Gospel of John and the Johannine Epistles?",
            "Is the Beloved Disciple the same as John the Apostle?"
        ]
    
    # Q2: Jesus's divinity in Mark
    elif (("divinity" in question_lower and "mark" in question_lower) or
         ("mark" in question_lower and "divine" in question_lower and "jesus" in question_lower)):
        
        matched_question = True
        result["positions"] = [
            {
                "tradition": "evangelical",
                "position": "Mark presents Jesus as divine, albeit more implicitly than John",
                "confidence": 0.70,
                "key_scholars": ["James Edwards", "Larry Hurtado", "Darrell Bock"],
                "common_arguments": ["Jesus forgives sins", "Son of Man authority", "Commands nature", "Receives worship"]
            },
            {
                "tradition": "critical",
                "position": "Mark presents Jesus as an exalted human figure but not as fully divine",
                "confidence": 0.65,
                "key_scholars": ["Bart Ehrman", "John Dominic Crossan", "Paula Fredriksen"],
                "common_arguments": ["Absence of pre-existence claims", "Human limitations emphasized", "Adoptionist elements in early manuscripts"]
            },
            {
                "tradition": "orthodox",
                "position": "Mark's Christology is high but expressed through action rather than titles",
                "confidence": 0.75,
                "key_scholars": ["John Behr", "Andrew Louth"],
                "common_arguments": ["Divine actions", "Narrative structure", "Old Testament allusions"]
            }
        ]
        
        result["consensus_level"] = "Medium"
        result["explanation"] = "Scholars generally agree that Mark's Christology is lower or less explicit than John's, but disagree on whether Mark presents Jesus as truly divine or as an exalted human figure. The disagreement often follows theological tradition lines."
        
        result["related_questions"] = [
            "What is the significance of Jesus forgiving sins in Mark?",
            "How does Mark's Christology compare to John's?",
            "What does the title 'Son of Man' mean in Mark?"
        ]
    
    # Q3: Two Powers in Heaven in First Century Judaism
    elif "two powers" in question_lower and ("heaven" in question_lower or "judaism" in question_lower):
        
        matched_question = True
        result["positions"] = [
            {
                "tradition": "evangelical",
                "position": "Some strands of first-century Judaism recognized a 'second power' figure alongside God",
                "confidence": 0.60,
                "key_scholars": ["Michael Heiser", "Larry Hurtado", "Daniel Boyarin"],
                "common_arguments": ["Angel of the Lord traditions", "Second Temple literature", "Early 'two powers' heresy"]
            },
            {
                "tradition": "jewish",
                "position": "The 'two powers heresy' was a later rabbinic category, not a first-century concept",
                "confidence": 0.75,
                "key_scholars": ["Alan Segal", "Daniel Boyarin", "Peter Schäfer"],
                "common_arguments": ["Anachronistic reading of later rabbinic concerns", "Strict monotheism in first-century Judaism", "Later development of the concept"]
            },
            {
                "tradition": "critical",
                "position": "There was diversity in Jewish conceptions of divine agency, but not a 'two powers' doctrine",
                "confidence": 0.70,
                "key_scholars": ["James D.G. Dunn", "E.P. Sanders", "Paula Fredriksen"],
                "common_arguments": ["Various mediator figures existed", "Principal angel traditions", "Divine attribute personification"]
            }
        ]
        
        result["consensus_level"] = "Low"
        result["explanation"] = "There is significant scholarly disagreement about whether 'two powers in heaven' was a concept in first-century Judaism. Most scholars acknowledge mediator figures and divine agents in Jewish tradition, but disagree about whether this constituted a 'two powers' concept that Christians could have drawn upon."
        
        result["related_questions"] = [
            "What is the rabbinic 'two powers in heaven' heresy?",
            "How did Jewish angelology influence early Christology?",
            "What divine mediator figures existed in Second Temple Judaism?"
        ]
    
    # If no match found
    if not matched_question:
        result["explanation"] = "This specific question does not have pre-recorded scholarly consensus information in the database. For accurate assessment, please consult recent academic publications on this topic."
    
    return result

def compare_scholarly_positions(position1: Dict[str, Any], position2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two scholarly positions to identify points of agreement and disagreement.
    
    Args:
        position1: First scholarly position dictionary
        position2: Second scholarly position dictionary
        
    Returns:
        Dictionary with comparison results
        
    Example:
        >>> # Compare evangelical and critical positions on Johannine authorship
        >>> evangelical = create_scholarly_position(...)
        >>> critical = create_scholarly_position(...)
        >>> comparison = compare_scholarly_positions(evangelical, critical)
    """
    # Initialize result structure
    result = {
        "position1": {
            "tradition": position1["tradition"],
            "position": position1["position"]
        },
        "position2": {
            "tradition": position2["tradition"],
            "position": position2["position"]
        },
        "tradition_relationship": "",
        "position_similarity": 0,
        "common_concepts": [],
        "unique_concepts1": [],
        "unique_concepts2": [],
        "methodological_differences": [],
        "summary": ""
    }
    
    # Compare traditions
    tradition1 = position1["tradition"]
    tradition2 = position2["tradition"]
    
    if tradition1 == tradition2:
        result["tradition_relationship"] = "Same tradition"
    else:
        # Identify tradition relationships
        if tradition1 in ["evangelical", "orthodox", "catholic"] and tradition2 in ["evangelical", "orthodox", "catholic"]:
            result["tradition_relationship"] = "Both Christian traditions with different emphases"
        elif tradition1 in ["evangelical", "orthodox", "catholic"] and tradition2 == "critical":
            result["tradition_relationship"] = "Traditional Christian vs. historical-critical approach"
        elif tradition2 in ["evangelical", "orthodox", "catholic"] and tradition1 == "critical":
            result["tradition_relationship"] = "Historical-critical approach vs. traditional Christian"
        elif tradition1 == "jewish" or tradition2 == "jewish":
            result["tradition_relationship"] = "Jewish and Christian interpretive traditions"
        else:
            result["tradition_relationship"] = "Different scholarly traditions"
    
    # Compare position content
    text1 = position1["position"].lower()
    text2 = position2["position"].lower()
    
    # Tokenize and extract key concepts (non-stopwords)
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    
    tokens1 = [w for w in word_tokenize(text1) if w.isalnum() and w not in stop_words]
    tokens2 = [w for w in word_tokenize(text2) if w.isalnum() and w not in stop_words]
    
    # Count term frequencies
    terms1 = Counter(tokens1)
    terms2 = Counter(tokens2)
    
    # Find common and unique terms
    common_terms = set(terms1.keys()).intersection(set(terms2.keys()))
    unique_terms1 = set(terms1.keys()) - common_terms
    unique_terms2 = set(terms2.keys()) - common_terms
    
    # Calculate similarity (Jaccard coefficient)
    if terms1 or terms2:
        similarity = len(common_terms) / len(set(terms1.keys()).union(set(terms2.keys())))
    else:
        similarity = 0
    
    result["position_similarity"] = similarity
    
    # Extract most important terms based on frequency
    result["common_concepts"] = sorted(common_terms, key=lambda t: terms1[t] + terms2[t], reverse=True)[:5]
    result["unique_concepts1"] = sorted(unique_terms1, key=lambda t: terms1[t], reverse=True)[:5]
    result["unique_concepts2"] = sorted(unique_terms2, key=lambda t: terms2[t], reverse=True)[:5]
    
    # Identify likely methodological differences
    methodological_markers = {
        "historical": ["historical", "history", "evidence", "sources", "manuscript"],
        "theological": ["theological", "doctrine", "systematic", "church teaching", "tradition"],
        "literary": ["literary", "narrative", "genre", "style", "structure"],
        "linguistic": ["linguistic", "grammar", "syntax", "semantics", "words"],
        "socio-cultural": ["culture", "social", "context", "background", "society"]
    }
    
    methods1 = []
    methods2 = []
    
    for method, markers in methodological_markers.items():
        if any(marker in text1 for marker in markers):
            methods1.append(method)
        if any(marker in text2 for marker in markers):
            methods2.append(method)
    
    # Add specific methodological differences
    differences = []
    
    if "historical" in methods1 and "theological" in methods2:
        differences.append("Historical-critical vs. theological approach")
    elif "theological" in methods1 and "historical" in methods2:
        differences.append("Theological vs. historical-critical approach")
    
    if "literary" in methods1 and not "literary" in methods2:
        differences.append("Literary analysis emphasis in position 1 but not position 2")
    elif "literary" in methods2 and not "literary" in methods1:
        differences.append("Literary analysis emphasis in position 2 but not position 1")
    
    if "socio-cultural" in methods1 and not "socio-cultural" in methods2:
        differences.append("Socio-cultural context emphasis in position 1 but not position 2")
    elif "socio-cultural" in methods2 and not "socio-cultural" in methods1:
        differences.append("Socio-cultural context emphasis in position 2 but not position 1")
    
    result["methodological_differences"] = differences if differences else ["No clear methodological differences identified"]
    
    # Generate summary
    if similarity > 0.7:
        agreement_level = "high level of agreement"
    elif similarity > 0.4:
        agreement_level = "moderate agreement with significant differences"
    else:
        agreement_level = "substantial disagreement"
    
    result["summary"] = f"These positions from {tradition1} and {tradition2} traditions show a {agreement_level}. "
    
    if result["common_concepts"]:
        result["summary"] += f"They share focus on concepts like {', '.join(result['common_concepts'][:3])}. "
    
    if result["methodological_differences"] and result["methodological_differences"][0] != "No clear methodological differences identified":
        result["summary"] += f"There are methodological differences: {result['methodological_differences'][0]}."
    
    return result

def rate_claim_scholarly_support(claim: str, category: Optional[str] = None) -> Dict[str, Any]:
    """
    Rate the level of scholarly support for a claim about biblical interpretation.
    
    Args:
        claim: The claim to evaluate
        category: Optional category to help with classification (e.g., 'christology', 'authorship')
        
    Returns:
        Dictionary with scholarly support assessment
        
    Example:
        >>> # Evaluate scholarly support for a claim about Paul's authorship
        >>> assessment = rate_claim_scholarly_support(
        ...     "Paul did not write the Pastoral Epistles (1-2 Timothy, Titus)",
        ...     category="authorship"
        ... )
        >>> print(assessment["support_level"])
    """
    # Initialize result
    result = {
        "claim": claim,
        "category": category,
        "support_level": None,
        "theological_alignment": {},
        "explanation": "",
        "mainstream_position": "",
        "minority_position": ""
    }
    
    # This is a placeholder implementation
    # In a real implementation, this would use NLP to compare the claim against
    # a database of scholarly positions or analyze scholarly literature
    
    # For demonstration, we'll include some hard-coded responses for common claims
    claim_lower = claim.lower()
    
    # Map claim to theological frameworks
    if category:
        # If category is provided, only check relevant frameworks
        framework_scores = {}
        for position_category, positions in THEOLOGICAL_POSITIONS.items():
            if position_category == category or category == "all":
                for position_name, position_data in positions.items():
                    score = 0
                    for concept in position_data["key_concepts"]:
                        if concept.lower() in claim_lower:
                            score += 1
                    
                    if position_data["key_concepts"]:
                        normalized_score = score / len(position_data["key_concepts"])
                    else:
                        normalized_score = 0
                    
                    framework_scores[f"{position_category}_{position_name}"] = normalized_score
    else:
        # Otherwise check all frameworks
        framework_scores = map_position_to_theological_framework(claim)
    
    # Add to result
    result["theological_alignment"] = framework_scores
    
    # Pattern match to some common claims
    matched_claim = False
    
    # Claim 1: Pastoral Epistles authorship
    if "pastoral" in claim_lower and "paul" in claim_lower and ("wrote" in claim_lower or "authorship" in claim_lower):
        matched_claim = True
        
        if "not write" in claim_lower or "didn't write" in claim_lower or "did not write" in claim_lower:
            # Claim that Paul did not write the Pastorals
            result["support_level"] = "High"
            result["explanation"] = "There is strong scholarly consensus that the Pastoral Epistles (1-2 Timothy, Titus) were not written directly by Paul. This view dominates in critical scholarship and is accepted by many moderate evangelical scholars, based on vocabulary differences, historical inconsistencies with Acts, and developed church structures."
            result["mainstream_position"] = "The Pastoral Epistles were written by a follower of Paul after his death, drawing on Pauline tradition."
            result["minority_position"] = "Paul directly authored the Pastoral Epistles during his final years."
        else:
            # Claim that Paul wrote the Pastorals
            result["support_level"] = "Low"
            result["explanation"] = "The majority of critical scholars and many moderate evangelical scholars consider the Pastoral Epistles to be deutero-Pauline (not written directly by Paul). The claim of direct Pauline authorship has limited support in current scholarship, though it remains the traditional view in some conservative circles."
            result["mainstream_position"] = "The Pastoral Epistles were written by a follower of Paul after his death, drawing on Pauline tradition."
            result["minority_position"] = "Paul directly authored the Pastoral Epistles during his final years."
    
    # Claim 2: Markan Priority
    elif "mark" in claim_lower and "first" in claim_lower and "gospel" in claim_lower:
        matched_claim = True
        
        result["support_level"] = "Very High"
        result["explanation"] = "Markan priority (the view that Mark was the first Gospel written and was used as a source by Matthew and Luke) has overwhelming scholarly support across all traditions. It forms the basis of the widely accepted Two-Source or Four-Source Hypothesis in Synoptic studies."
        result["mainstream_position"] = "Mark was written first and used as a source by both Matthew and Luke."
        result["minority_position"] = "Matthew was written first (Augustinian hypothesis) or the Gospels were written independently."
    
    # Claim 3: Pre-existence in John
    elif "john" in claim_lower and "pre-existence" in claim_lower and "jesus" in claim_lower:
        matched_claim = True
        
        result["support_level"] = "Very High"
        result["explanation"] = "There is overwhelming scholarly consensus that the Gospel of John presents Jesus as pre-existent (existing before his human birth). This is accepted across critical, evangelical, Catholic, and Orthodox scholarship, based primarily on the Prologue (John 1:1-18) and statements like John 8:58."
        result["mainstream_position"] = "The Gospel of John explicitly teaches Jesus' pre-existence as the divine Word/Logos."
        result["minority_position"] = "John's language about pre-existence should be understood figuratively or as later theological development."
    
    # Claim 4: Documentary Hypothesis
    elif (("documentary" in claim_lower or "jedp" in claim_lower) and 
          ("pentateuch" in claim_lower or "torah" in claim_lower or "moses" in claim_lower)):
        matched_claim = True
        
        if "false" in claim_lower or "incorrect" in claim_lower or "wrong" in claim_lower:
            # Claim against Documentary Hypothesis
            result["support_level"] = "Low"
            result["explanation"] = "While the classic JEDP Documentary Hypothesis has been significantly modified in current scholarship, some form of the documentary approach to Pentateuchal composition has strong support among critical scholars. Complete rejection of documentary approaches has limited support mainly in conservative evangelical circles."
            result["mainstream_position"] = "The Pentateuch was composed from multiple sources/traditions, though the classic JEDP model has been extensively revised."
            result["minority_position"] = "The Pentateuch was essentially composed by Moses as a unified work in the Late Bronze Age."
        else:
            # Claim supporting Documentary Hypothesis
            result["support_level"] = "Medium"
            result["explanation"] = "Modern scholarly approaches to Pentateuchal criticism generally accept some form of the documentary approach, though the classic JEDP model has been significantly modified. There is substantial scholarly disagreement about the dating, extent, and relationship of the sources."
            result["mainstream_position"] = "The Pentateuch was composed from multiple sources/traditions, though the classic JEDP model has been extensively revised."
            result["minority_position"] = "The Pentateuch was essentially composed by Moses as a unified work in the Late Bronze Age."
    
    # Claim 5: Two Powers in Heaven
    elif "two powers" in claim_lower and "judaism" in claim_lower:
        matched_claim = True
        
        result["support_level"] = "Low to Medium"
        result["explanation"] = "The claim that 'two powers in heaven' was a recognized concept in first-century Judaism has moderate support from some scholars (e.g., Alan Segal, Daniel Boyarin) but is disputed by others who consider it anachronistic. Most scholars acknowledge divine mediator figures in Judaism but disagree about whether this constituted a 'two powers' concept."
        result["mainstream_position"] = "First-century Judaism had various divine mediator figures and exalted agents, but 'two powers' as a concept is primarily known from later rabbinic literature as a heresy."
        result["minority_position"] = "A 'two powers' concept existed in some strands of first-century Judaism that influenced early Christology."
    
    # If no match found
    if not matched_claim:
        # Default to analyzing theological alignment
        if framework_scores:
            # Find the highest scoring framework
            max_framework = max(framework_scores.items(), key=lambda x: x[1])
            
            if max_framework[1] > 0.5:
                # High alignment with a known theological position
                framework_name = max_framework[0]
                category, position = framework_name.split('_', 1)
                
                result["explanation"] = f"This claim aligns closely with the {position} position on {category}. "
                
                # Add support level based on which position it aligns with
                # (This is a simplified approach - in reality would need to assess actual scholarly support)
                if position in ["trinitarian", "verbal_plenary", "substitutionary"]:
                    result["support_level"] = "Medium to High in evangelical circles, Lower in critical scholarship"
                elif position in ["dynamic", "encounter", "exemplary", "victory"]:
                    result["support_level"] = "Medium to High in critical and mainline scholarship, Lower in conservative circles"
                else:
                    result["support_level"] = "Varies significantly across scholarly traditions"
            else:
                # Low alignment with known positions
                result["explanation"] = "This claim does not strongly align with standard positions in biblical scholarship. "
                result["support_level"] = "Indeterminate"
        else:
            # No theological alignment detected
            result["explanation"] = "This specific claim does not match known patterns for scholarly assessment. For accurate evaluation, please consult recent academic publications on this topic."
            result["support_level"] = "Indeterminate"
    
    return result

# Pre-defined scholarly consensus on common debate topics
COMMON_CONSENSUS = {
    "johannine_authorship": {
        "question": "Who wrote the Gospel of John?",
        "consensus_level": "Low",
        "explanation": "There is significant disagreement between evangelical and critical scholars on Johannine authorship. Evangelical scholars generally maintain apostolic authorship (the Apostle John), while critical scholars favor a community composition model or attribute it to a different John (the Elder).",
        "positions": {
            "evangelical": "The Apostle John wrote the Fourth Gospel, possibly with some editorial assistance",
            "critical": "The Gospel of John was composed by a Johannine community, not directly by the Apostle John",
            "catholic": "The Gospel has roots in Johannine testimony but reached final form through disciples"
        }
    },
    "trinity_development": {
        "question": "When and how did the doctrine of the Trinity develop?",
        "consensus_level": "Medium",
        "explanation": "Scholars generally agree that the formal doctrine of the Trinity was not explicitly formulated until the 4th century, but disagree on the extent to which trinitarian concepts were present in the New Testament and early church.",
        "positions": {
            "evangelical": "The Trinity is implicitly present in the New Testament and was progressively clarified, not invented later",
            "critical": "The Trinity was a later theological development that went beyond the original New Testament concepts",
            "catholic/orthodox": "The Trinity was implicit in apostolic teaching and developed organically through the church's reflection"
        }
    },
    "synoptic_problem": {
        "question": "What is the relationship between the Synoptic Gospels?",
        "consensus_level": "High",
        "explanation": "There is strong scholarly consensus for Markan priority (Mark was written first) and the existence of Q (a sayings source used by Matthew and Luke), though some scholars support alternative models.",
        "positions": {
            "mainstream": "Two/Four-Source Hypothesis: Mark was written first, with Matthew and Luke using Mark and Q (plus unique material)",
            "minority": "Augustinian Hypothesis (Matthew first) or Farrer Hypothesis (no Q source)"
        }
    },
    "pauline_authorship": {
        "question": "Which epistles were actually written by Paul?",
        "consensus_level": "Medium",
        "explanation": "There is strong scholarly consensus that 7 epistles are authentic Pauline works, with others being disputed or considered pseudepigraphical.",
        "positions": {
            "undisputed": "Romans, 1-2 Corinthians, Galatians, Philippians, 1 Thessalonians, and Philemon are undisputed Pauline works",
            "disputed": "Ephesians, Colossians, and 2 Thessalonians have mixed scholarly opinion",
            "widely_considered_pseudepigraphical": "Pastoral Epistles (1-2 Timothy, Titus) are widely considered to be written in Paul's name by later authors"
        }
    },
    "historical_jesus": {
        "question": "What can we know about the historical Jesus?",
        "consensus_level": "Medium",
        "explanation": "There is significant scholarly consensus on Jesus's existence, Jewish context, baptism by John, teaching activity, and crucifixion under Pontius Pilate, but disagreement on many details of his life and teachings.",
        "positions": {
            "mainstream": "Jesus was an apocalyptic Jewish prophet who proclaimed the kingdom of God, gathered disciples, created controversy, and was crucified",
            "evangelical": "The Gospels provide substantially accurate portrayals of Jesus's life and teachings",
            "critical": "The Gospels contain theological interpretations that must be carefully distinguished from historical elements"
        }
    }
}
