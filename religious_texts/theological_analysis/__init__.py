"""
Theological Analysis Module

This module provides specialized tools for theological analysis of biblical texts,
focusing on divine name usage, speech attribution, worship language contexts,
authority claims, and parallel passage analysis.
"""

from religious_texts.theological_analysis.divine_names import (
    divine_name_usage,
    divine_name_distribution, 
    divine_name_context_analysis,
    divine_title_analysis
)

from religious_texts.theological_analysis.speech import (
    identify_speech_segments,
    speech_attribution_analysis,
    speech_act_distribution,
    speech_comparison_by_speaker
)

from religious_texts.theological_analysis.worship import (
    identify_worship_contexts,
    worship_language_analysis,
    worship_term_distribution,
    prayer_pattern_analysis
)

from religious_texts.theological_analysis.authority import (
    identify_authority_claims,
    authority_language_analysis,
    delegation_pattern_analysis,
    command_distribution
)

from religious_texts.theological_analysis.parallel import (
    identify_parallel_passages,
    compare_parallel_passages,
    analyze_narrative_differences,
    synoptic_analysis
)

__all__ = [
    # Divine name analysis
    'divine_name_usage',
    'divine_name_distribution',
    'divine_name_context_analysis',
    'divine_title_analysis',
    
    # Speech attribution analysis
    'identify_speech_segments',
    'speech_attribution_analysis',
    'speech_act_distribution',
    'speech_comparison_by_speaker',
    
    # Worship language analysis
    'identify_worship_contexts',
    'worship_language_analysis',
    'worship_term_distribution',
    'prayer_pattern_analysis',
    
    # Authority claim analysis
    'identify_authority_claims',
    'authority_language_analysis',
    'delegation_pattern_analysis',
    'command_distribution',
    
    # Parallel passage analysis
    'identify_parallel_passages',
    'compare_parallel_passages',
    'analyze_narrative_differences',
    'synoptic_analysis'
]
