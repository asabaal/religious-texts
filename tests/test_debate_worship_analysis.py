"""
Tests for the specialized debate worship analysis module.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from religious_texts.specialized.debate_worship_analysis import (
    analyze_debate_claim,
    analyze_debate_worship_terms,
    create_worship_evidence_matrix,
    analyze_debate_passages_context,
    DEBATE_CLAIMS
)

class TestDebateWorshipAnalysis(unittest.TestCase):
    """Test the specialized debate worship analysis functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock Bible dictionary for testing
        self.mock_bible = {
            "Matthew": {
                2: {
                    11: "And going into the house, they saw the child with Mary his mother, and they fell down and worshiped him."
                },
                4: {
                    10: "Then Jesus said to him, 'Be gone, Satan! For it is written, \"You shall worship the Lord your God and him only shall you serve.\"'"
                },
                28: {
                    17: "And when they saw him they worshiped him, but some doubted."
                }
            },
            "John": {
                9: {
                    38: "He said, 'Lord, I believe,' and he worshiped him."
                },
                12: {
                    20: "Now among those who went up to worship at the feast were some Greeks."
                }
            },
            "Romans": {
                1: {
                    9: "For God is my witness, whom I serve with my spirit in the gospel of his Son.",
                    25: "They exchanged the truth about God for a lie and worshiped and served the creature rather than the Creator."
                }
            },
            "Hebrews": {
                1: {
                    6: "And again, when he brings the firstborn into the world, he says, 'Let all God's angels worship him.'"
                }
            },
            "Revelation": {
                22: {
                    8: "I, John, am the one who heard and saw these things. And when I heard and saw them, I fell down to worship at the feet of the angel who showed them to me.",
                    9: "But he said to me, 'You must not do that! I am a fellow servant with you and your brothers the prophets, and with those who keep the words of this book. Worship God.'"
                }
            }
        }
    
    def test_analyze_debate_claim(self):
        """Test the analysis of a specific debate claim."""
        # Test proskuneo exclusivity claim
        analysis = analyze_debate_claim(self.mock_bible, "proskuneo_exclusivity")
        
        # Verify analysis structure
        self.assertIn("claim", analysis, "Analysis should include the claim text")
        self.assertIn("counter", analysis, "Analysis should include the counter text")
        self.assertIn("related_terms", analysis, "Analysis should include related terms")
        self.assertIn("key_passages", analysis, "Analysis should include key passages")
        self.assertIn("term_occurrences", analysis, "Analysis should include term occurrences")
        self.assertIn("support_statistics", analysis, "Analysis should include support statistics")
        
        # Test with an invalid claim key
        with self.assertRaises(ValueError):
            analyze_debate_claim(self.mock_bible, "invalid_claim")
    
    def test_analyze_debate_worship_terms(self):
        """Test the analysis of debate worship terms."""
        # Define test passages
        proskuneo_passages = ["Matthew 2:11", "Matthew 28:17", "John 9:38"]
        latreo_passages = ["Matthew 4:10", "Romans 1:9"]
        
        # Run analysis
        analysis = analyze_debate_worship_terms(
            self.mock_bible,
            proskuneo_passages=proskuneo_passages,
            latreo_passages=latreo_passages
        )
        
        # Verify analysis structure
        self.assertIn("proskuneo_analysis", analysis, "Analysis should include proskuneo_analysis")
        self.assertIn("latreo_analysis", analysis, "Analysis should include latreo_analysis")
        self.assertIn("comparison", analysis, "Analysis should include comparison")
        self.assertIn("debate_significance", analysis, "Analysis should include debate_significance")
        
        # Check proskuneo analysis
        proskuneo = analysis["proskuneo_analysis"]
        self.assertEqual(len(proskuneo["passages"]), len(proskuneo_passages), 
                        f"Should include {len(proskuneo_passages)} proskuneo passages")
        
        # Check latreo analysis
        latreo = analysis["latreo_analysis"]
        self.assertEqual(len(latreo["passages"]), len(latreo_passages), 
                        f"Should include {len(latreo_passages)} latreo passages")
        
        # Test with default passages
        default_analysis = analyze_debate_worship_terms(self.mock_bible)
        self.assertIn("proskuneo_analysis", default_analysis, "Default analysis should include proskuneo_analysis")
    
    def test_create_worship_evidence_matrix(self):
        """Test the creation of worship evidence matrix."""
        # Create matrix
        df = create_worship_evidence_matrix(self.mock_bible)
        
        # Verify DataFrame structure
        self.assertIsInstance(df, pd.DataFrame, "Result should be a DataFrame")
        
        # Check for expected columns
        expected_columns = ["claim", "evidence_category", "support_count", "counter_count", 
                           "neutral_count", "total_passages", "claim_key"]
        for col in expected_columns:
            self.assertIn(col, df.columns, f"DataFrame should include {col} column")
        
        # Verify all claims are included
        self.assertEqual(len(df), len(DEBATE_CLAIMS), 
                        f"Matrix should include all {len(DEBATE_CLAIMS)} claims")
    
    def test_analyze_debate_passages_context(self):
        """Test the analysis of debate passages context."""
        # Define test passages
        passages = ["Matthew 2:11", "John 9:38", "Revelation 22:8"]
        
        # Run analysis
        analysis = analyze_debate_passages_context(self.mock_bible, passages)
        
        # Verify analysis structure
        self.assertIn("passages", analysis, "Analysis should include passages")
        self.assertIn("total_passages", analysis, "Analysis should include total_passages")
        self.assertIn("worship_category_counts", analysis, "Analysis should include worship_category_counts")
        self.assertIn("recipient_category_counts", analysis, "Analysis should include recipient_category_counts")
        self.assertIn("theme_counts", analysis, "Analysis should include theme_counts")
        
        # Check number of passages
        self.assertEqual(analysis["total_passages"], len(passages), 
                        f"Should analyze all {len(passages)} passages")
        
        # Test with larger context
        large_context = analyze_debate_passages_context(self.mock_bible, passages, context_verses=5)
        self.assertIn("passages", large_context, "Large context analysis should include passages")

if __name__ == "__main__":
    unittest.main()
