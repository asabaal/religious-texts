"""
Debate Response Module

This module provides tools for analyzing and responding to claims about biblical
content in debate contexts, including statistical validation, interpretation
comparison, contextual analysis, and scholarly consensus measurement.
"""

from religious_texts.debate_response.validators import (
    validate_word_frequency_claim,
    validate_cooccurrence_claim,
    validate_theological_claim,
    statistical_confidence,
    check_distribution_claim
)

from religious_texts.debate_response.comparisons import (
    compare_interpretations,
    interpretation_consistency,
    alternative_readings,
    compare_translations,
    translation_bias_analysis
)

from religious_texts.debate_response.context import (
    analyze_quote_context,
    broader_literary_context,
    historical_context,
    verse_contextual_meaning,
    cross_reference_analysis
)

from religious_texts.debate_response.consensus import (
    measure_scholarly_consensus,
    identify_academic_positions,
    consensus_evolution,
    expert_distribution,
    track_theological_trends
)

__all__ = [
    # Claim validation
    'validate_word_frequency_claim',
    'validate_cooccurrence_claim',
    'validate_theological_claim',
    'statistical_confidence',
    'check_distribution_claim',
    
    # Interpretation comparison
    'compare_interpretations',
    'interpretation_consistency',
    'alternative_readings',
    'compare_translations',
    'translation_bias_analysis',
    
    # Context analysis
    'analyze_quote_context',
    'broader_literary_context',
    'historical_context',
    'verse_contextual_meaning',
    'cross_reference_analysis',
    
    # Scholarly consensus
    'measure_scholarly_consensus',
    'identify_academic_positions',
    'consensus_evolution',
    'expert_distribution',
    'track_theological_trends'
]
