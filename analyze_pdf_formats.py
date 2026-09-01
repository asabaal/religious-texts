#!/usr/bin/env python3
"""
Analyze formatting patterns of all three Bible PDFs to determine extraction difficulty
"""

import pdfplumber
import re

def analyze_pdf_format(pdf_path, sample_pages=[50, 100, 150, 200, 250]):
    """Analyze specific pages from a PDF to understand formatting patterns"""
    print(f"\n{'='*60}")
    print(f"📖 ANALYZING: {pdf_path}")
    print(f"{'='*60}")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = pdf.pages.__len__()
            print(f"📄 Total pages: {total_pages}")
            
            for page_num in sample_pages:
                if page_num >= total_pages:
                    continue
                    
                print(f"\n--- PAGE {page_num} ---")
                page = pdf.pages[page_num]
                text = page.extract_text()
                
                if text:
                    lines = text.split('\n')
                    for i, line in enumerate(lines[:15]):  # First 15 lines per page
                        # Show line with analysis
                        clean_line = line.strip()
                        if clean_line:
                            # Detect patterns
                            has_number_at_start = bool(re.match(r'^\d+', clean_line))
                            has_inline_number = bool(re.search(r'\b\d+\b', clean_line))
                            is_all_caps = clean_line.isupper() and len(clean_line) > 3
                            is_standalone_number = clean_line.isdigit()
                            has_colon_reference = bool(re.search(r'\d+:\d+', clean_line))
                            
                            pattern_indicators = []
                            if has_number_at_start: pattern_indicators.append("START_NUM")
                            if has_inline_number: pattern_indicators.append("INLINE_NUM") 
                            if is_all_caps: pattern_indicators.append("ALL_CAPS")
                            if is_standalone_number: pattern_indicators.append("STANDALONE_NUM")
                            if has_colon_reference: pattern_indicators.append("REF_FORMAT")
                            
                            indicators = f"[{', '.join(pattern_indicators)}]" if pattern_indicators else ""
                            print(f"{i+1:2}: {clean_line[:80]} {indicators}")
                
                print()  # Spacing between pages
                
    except Exception as e:
        print(f"❌ Error analyzing {pdf_path}: {e}")

def find_book_chapter_patterns(pdf_path):
    """Look for book and chapter patterns in the PDF"""
    print(f"\n🔍 SEARCHING FOR BOOK/CHAPTER PATTERNS in {pdf_path}:")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            found_books = set()
            found_chapters = []
            
            # Check first 50 pages for patterns
            for page_num in range(min(50, pdf.pages.__len__())):
                page = pdf.pages[page_num]
                text = page.extract_text()
                
                if text:
                    lines = text.split('\n')
                    for line in lines:
                        clean_line = line.strip()
                        
                        # Look for potential book names
                        bible_books = ['Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy',
                                      'Joshua', 'Judges', 'Ruth', 'Samuel', 'Kings', 'Chronicles',
                                      'Matthew', 'Mark', 'Luke', 'John', 'Acts', 'Romans']
                        
                        for book in bible_books:
                            if book in clean_line:
                                found_books.add(book)
                                print(f"  📚 Found book '{book}' in: {clean_line}")
                        
                        # Look for chapter patterns
                        if re.match(r'^\d+$', clean_line) and int(clean_line) <= 150:
                            found_chapters.append((page_num, clean_line))
                            if len(found_chapters) <= 5:  # Show first 5 examples
                                print(f"  🔢 Found chapter '{clean_line}' on page {page_num}")
            
            print(f"\n📊 SUMMARY for {pdf_path}:")
            print(f"  Books found: {sorted(found_books)}")
            print(f"  Chapter examples: {found_chapters[:3]}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    pdf_files = [
        'bible_king_james_version.pdf',
        'english-standard-version-bible.pdf', 
        'new-revised-standard-version.pdf'
    ]
    
    print("🔍 BIBLE PDF FORMAT ANALYSIS")
    print("Analyzing formatting patterns to determine extraction difficulty...")
    
    for pdf_file in pdf_files:
        try:
            analyze_pdf_format(pdf_file)
            find_book_chapter_patterns(pdf_file)
        except Exception as e:
            print(f"❌ Could not analyze {pdf_file}: {e}")
    
    print(f"\n{'='*60}")
    print("🎯 ANALYSIS COMPLETE")
    print("Review the patterns above to determine which PDF is easiest to parse")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()