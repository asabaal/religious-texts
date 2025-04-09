"""
Tests for the worship language analysis module in theological_analysis.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from religious_texts.theological_analysis.worship import (
    identify_worship_contexts,
    worship_language_analysis,
    worship_term_distribution,
    prayer_pattern_analysis,
    WORSHIP_TERMS,
    PRAYER_PATTERNS,
    WORSHIP_BOOKS
)

class TestWorship(unittest.TestCase):
    """Test the worship language analysis functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock Bible dictionary for testing
        self.mock_bible = {
            "Psalms": {
                1: {
                    1: "Blessed is the man who walks not in the counsel of the wicked.",
                    2: "But his delight is in the law of the LORD, and on his law he meditates day and night."
                },
                100: {
                    1: "Make a joyful noise to the LORD, all the earth!",
                    2: "Serve the LORD with gladness! Come into his presence with singing!",
                    3: "Know that the LORD, he is God! It is he who made us, and we are his; we are his people, and the sheep of his pasture.",
                    4: "Enter his gates with thanksgiving, and his courts with praise! Give thanks to him; bless his name!"
                }
            },
            "Matthew": {
                6: {
                    5: "And when you pray, you must not be like the hypocrites. For they love to stand and pray in the synagogues and at the street corners, that they may be seen by others.",
                    6: "But when you pray, go into your room and shut the door and pray to your Father who is in secret.",
                    7: "And when you pray, do not heap up empty phrases as the Gentiles do, for they think that they will be heard for their many words.",
                    9: "Pray then like this: 'Our Father in heaven, hallowed be your name.'",
                    10: "Your kingdom come, your will be done, on earth as it is in heaven."
                }
            },
            "Leviticus": {
                1: {
                    1: "The LORD called Moses and spoke to him from the tent of meeting, saying,",
                    2: "Speak to the people of Israel and say to them, When any one of you brings an offering to the LORD, you shall bring your offering of livestock from the herd or from the flock.",
                    3: "If his offering is a burnt offering from the herd, he shall offer a male without blemish. He shall bring it to the entrance of the tent of meeting, that he may be accepted before the LORD."
                }
            }
        }
    
    def test_identify_worship_contexts(self):
        """Test the identification of worship contexts."""
        # Test identification of worship contexts in all books
        df = identify_worship_contexts(self.mock_bible)
        self.assertGreater(len(df), 0, "Should identify at least one worship context")
        
        # Test filtering by book
        psalms_df = identify_worship_contexts(self.mock_bible, book="Psalms")
        self.assertTrue(all(book == "Psalms" for book in psalms_df["book"]), 
                       "Should only include contexts from Psalms")
        
        # Test filtering by chapter
        chapter_df = identify_worship_contexts(self.mock_bible, book="Psalms", chapter=100)
        self.assertTrue(all(chapter == 100 for chapter in chapter_df["chapter"]),
                       "Should only include contexts from Psalms 100")
        
        # Test filtering by categories
        categories = ["praise", "prayer"]
        categories_df = identify_worship_contexts(self.mock_bible, categories=categories)
        for _, row in categories_df.iterrows():
            found_categories = row["categories"].split(", ")
            self.assertTrue(any(cat in categories for cat in found_categories),
                           "Should only include specified worship categories")
    
    def test_worship_language_analysis(self):
        """Test the analysis of worship language distribution."""
        # Test analysis by book
        book_df = worship_language_analysis(self.mock_bible, unit="book")
        self.assertGreater(len(book_df), 0, "Should analyze at least one book")
        self.assertIn("total_worship_terms", book_df.columns, "Should include total_worship_terms column")
        
        # Test analysis by chapter
        chapter_df = worship_language_analysis(self.mock_bible, unit="chapter")
        self.assertGreater(len(chapter_df), 0, "Should analyze at least one chapter")
        self.assertIn("chapter", chapter_df.columns, "Should include chapter column")
        
        # Test filtering by book
        psalms_df = worship_language_analysis(self.mock_bible, book="Psalms", unit="chapter")
        self.assertTrue(all(book == "Psalms" for book in psalms_df["book"]), 
                       "Should only include analysis for Psalms")
        
        # Test without normalization
        unnormalized_df = worship_language_analysis(self.mock_bible, normalize=False)
        self.assertNotIn("worship_density", unnormalized_df.columns, 
                        "Should not include normalized columns when normalize=False")
    
    def test_worship_term_distribution(self):
        """Test the analysis of specific worship term distribution."""
        # Define test terms
        terms = ["praise", "worship", "sacrifice", "prayer"]
        
        # Test with all books
        df = worship_term_distribution(self.mock_bible, terms)
        self.assertGreater(len(df), 0, "Should analyze at least one book")
        for term in terms:
            self.assertIn(term, df.columns, f"Should include {term} column")
        
        # Test with specific books
        books = ["Psalms", "Leviticus"]
        books_df = worship_term_distribution(self.mock_bible, terms, books=books)
        self.assertTrue(all(book in books for book in books_df["book"]), 
                       "Should only include specified books")
        
        # Test without normalization
        unnormalized_df = worship_term_distribution(self.mock_bible, terms, normalize=False)
        self.assertIn("total_term_count", unnormalized_df.columns, "Should include total_term_count column")
        for term in terms:
            normalized_term = f"{term}_normalized"
            self.assertNotIn(normalized_term, unnormalized_df.columns,
                            f"Should not include {normalized_term} when normalize=False")
    
    def test_prayer_pattern_analysis(self):
        """Test the analysis of prayer patterns."""
        # Test with all books
        df = prayer_pattern_analysis(self.mock_bible)
        self.assertGreater(len(df), 0, "Should analyze at least one prayer")
        self.assertIn("pattern_count", df.columns, "Should include pattern_count column")
        
        # Test filtering by book
        matthew_df = prayer_pattern_analysis(self.mock_bible, book="Matthew")
        self.assertTrue(all(book == "Matthew" for book in matthew_df["book"]), 
                       "Should only include prayers from Matthew")
        
        # Test filtering by chapter
        chapter_df = prayer_pattern_analysis(self.mock_bible, book="Matthew", chapter=6)
        self.assertTrue(all(chapter == 6 for chapter in chapter_df["chapter"]),
                       "Should only include prayers from Matthew 6")
        
        # Verify prayer pattern flags
        for pattern in PRAYER_PATTERNS:
            flag_column = f"has_{pattern}"
            self.assertIn(flag_column, df.columns, f"Should include {flag_column} column")

if __name__ == "__main__":
    unittest.main()
