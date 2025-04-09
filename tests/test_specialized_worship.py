"""
Tests for the specialized worship analysis module.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from religious_texts.specialized.worship_analysis import (
    extract_worship_instances,
    analyze_proskuneo_usage,
    compare_proskuneo_latreo,
    analyze_worship_recipients,
    analyze_worship_contexts,
    analyze_worship_in_debate_passages
)

class TestSpecializedWorship(unittest.TestCase):
    """Test the specialized worship analysis functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock Bible dictionary for testing
        self.mock_bible = {
            "Matthew": {
                1: {
                    1: "The genealogy of Jesus Christ, the son of David, the son of Abraham.",
                    2: "Abraham was the father of Isaac, and Isaac the father of Jacob."
                },
                2: {
                    11: "And going into the house, they saw the child with Mary his mother, and they fell down and worshiped him."
                },
                4: {
                    10: "Then Jesus said to him, 'Be gone, Satan! For it is written, \"You shall worship the Lord your God and him only shall you serve.\"'"
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
            "Revelation": {
                7: {
                    15: "Therefore they are before the throne of God, and serve him day and night in his temple."
                },
                22: {
                    3: "No longer will there be anything accursed, but the throne of God and of the Lamb will be in it, and his servants will worship him."
                }
            }
        }
    
    def test_extract_worship_instances(self):
        """Test the extraction of worship instances."""
        # Test extraction of all worship instances
        df = extract_worship_instances(self.mock_bible)
        self.assertGreater(len(df), 0, "Should extract at least one worship instance")
        
        # Test filtering by worship category
        proskuneo_df = extract_worship_instances(self.mock_bible, worship_category="proskuneo")
        self.assertGreater(len(proskuneo_df), 0, "Should extract proskuneo instances")
        
        # Test filtering by recipient category
        jesus_df = extract_worship_instances(self.mock_bible, recipient_category="jesus_terms")
        self.assertGreater(len(jesus_df), 0, "Should extract instances with Jesus as recipient")
        
        # Test filtering by books
        matthew_df = extract_worship_instances(self.mock_bible, books=["Matthew"])
        self.assertTrue(all(book == "Matthew" for book in matthew_df["book"]), 
                        "Should only include instances from Matthew")
    
    def test_analyze_proskuneo_usage(self):
        """Test the analysis of proskuneo usage."""
        analysis = analyze_proskuneo_usage(self.mock_bible)
        
        # Verify the analysis structure
        self.assertIn("total_instances", analysis, "Analysis should include total_instances")
        self.assertIn("recipient_distribution", analysis, "Analysis should include recipient_distribution")
        self.assertIn("book_distribution", analysis, "Analysis should include book_distribution")
        self.assertIn("examples", analysis, "Analysis should include examples")
        
        # Test with specific books
        gospel_analysis = analyze_proskuneo_usage(self.mock_bible, books=["Matthew", "John"])
        self.assertIn("gospel_distribution", gospel_analysis, "Analysis should include gospel_distribution")
    
    def test_compare_proskuneo_latreo(self):
        """Test the comparison of proskuneo and latreo terms."""
        comparison = compare_proskuneo_latreo(self.mock_bible)
        
        # Verify the comparison structure
        self.assertIn("proskuneo_instances", comparison, "Comparison should include proskuneo_instances")
        self.assertIn("latreo_instances", comparison, "Comparison should include latreo_instances")
        self.assertIn("proskuneo_recipients", comparison, "Comparison should include proskuneo_recipients")
        self.assertIn("latreo_recipients", comparison, "Comparison should include latreo_recipients")
        self.assertIn("overlap_count", comparison, "Comparison should include overlap_count")
    
    def test_analyze_worship_recipients(self):
        """Test the analysis of worship recipients."""
        df = analyze_worship_recipients(self.mock_bible, recipient="jesus_terms")
        
        # Verify the result is a DataFrame
        self.assertIsInstance(df, pd.DataFrame, "Result should be a DataFrame")
        
        # Test with specific worship categories
        categories_df = analyze_worship_recipients(
            self.mock_bible, 
            worship_categories=["proskuneo"], 
            recipient="jesus_terms"
        )
        if not categories_df.empty:
            self.assertTrue(all(cat == "proskuneo" for cat in categories_df["worship_category"]),
                           "Should only include proskuneo instances")
    
    def test_analyze_worship_contexts(self):
        """Test the analysis of worship contexts."""
        contexts = analyze_worship_contexts(self.mock_bible)
        
        # Verify the analysis structure
        self.assertIn("total_instances", contexts, "Analysis should include total_instances")
        self.assertIn("common_contexts", contexts, "Analysis should include common_contexts")
        self.assertIn("recipient_contexts", contexts, "Analysis should include recipient_contexts")
        
        # Test with specific context window
        narrow_contexts = analyze_worship_contexts(self.mock_bible, context_window=1)
        self.assertIn("common_contexts", narrow_contexts, "Analysis should include common_contexts")
    
    def test_analyze_worship_in_debate_passages(self):
        """Test the analysis of worship in debate passages."""
        debate_passages = ["Matthew 2:11", "John 9:38", "Romans 1:25", "Revelation 22:3"]
        analysis = analyze_worship_in_debate_passages(self.mock_bible, debate_passages)
        
        # Verify the analysis structure
        self.assertIn("passages", analysis, "Analysis should include passages")
        self.assertIn("proskuneo_count", analysis, "Analysis should include proskuneo_count")
        self.assertIn("latreo_count", analysis, "Analysis should include latreo_count")
        self.assertIn("jesus_worship_count", analysis, "Analysis should include jesus_worship_count")
        
        # Check that all valid passages are included
        self.assertEqual(len(analysis["passages"]), len(debate_passages),
                         "Analysis should include all valid debate passages")

if __name__ == "__main__":
    unittest.main()
