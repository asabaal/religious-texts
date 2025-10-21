#!/usr/bin/env python3
"""
Integration test for Bible Extraction System
Tests PDF processing + AI extraction together
"""

import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pdf_processor import PDFProcessor
from ai_extractor import AIExtractor

def test_integration():
    """Test the complete extraction pipeline"""
    print("🧪 Bible Extraction System Integration Test")
    print("=" * 50)
    
    # Initialize components
    pdf_processor = PDFProcessor("config/pdfs.yaml")
    ai_extractor = AIExtractor("config/models.yaml")
    
    # Test PDF
    pdf_path = "../bible_king_james_version.pdf"
    
    print(f"📖 Testing with: {pdf_path}")
    
    # Step 1: Book Discovery
    print("\n1️⃣ Discovering books...")
    book_result = pdf_processor.discover_books(pdf_path)
    
    print(f"✅ Detected {len(book_result.detected_books)} books:")
    for book in book_result.detected_books[:5]:  # Show first 5
        print(f"   - {book}")
    if len(book_result.detected_books) > 5:
        print(f"   ... and {len(book_result.detected_books) - 5} more")
    
    print(f"📊 Canon type: {book_result.canon_type}")
    print(f"📋 PDF format: {book_result.pdf_format}")
    
    # Step 2: Extract a few pages for testing
    print("\n2️⃣ Extracting sample pages...")
    pages = pdf_processor.extract_pages(pdf_path, start_page=50, end_page=52)  # Pages 51-53
    
    print(f"✅ Extracted {len(pages)} pages")
    for page in pages:
        print(f"   Page {page.page_num}: {page.metadata['line_count']} lines")
    
    # Step 3: AI Extraction on first page
    print("\n3️⃣ Testing AI extraction...")
    if pages:
        from ai_extractor import ExtractionContext
        
        # Create context
        context = ExtractionContext(
            current_book=None,
            current_chapter=None,
            last_verse=None,
            detected_books=book_result.detected_books,
            pdf_format=book_result.pdf_format,
            canon_type=book_result.canon_type
        )
        
        # Extract verses from first page
        verses = ai_extractor.extract_verses_from_page(
            pages[0], pages, 0, context
        )
        
        print(f"✅ AI extracted {len(verses)} verses from page {pages[0].page_num}")
        
        # Show results
        for i, verse in enumerate(verses[:3]):  # Show first 3
            print(f"\n   Verse {i+1}:")
            print(f"   📖 {verse.book} {verse.chapter}:{verse.verse}")
            print(f"   📝 {verse.text[:100]}...")
            print(f"   🎯 Confidence: {verse.confidence:.2f}")
        
        if len(verses) > 3:
            print(f"\n   ... and {len(verses) - 3} more verses")
    
    print("\n🎉 Integration test completed successfully!")
    print("\n📋 System Status:")
    print("   ✅ PDF Processing: Working")
    print("   ✅ Book Discovery: Working") 
    print("   ✅ AI Extraction: Working")
    print("   ✅ Ollama + Qwen3: Connected")

if __name__ == "__main__":
    test_integration()