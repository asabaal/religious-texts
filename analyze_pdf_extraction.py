#!/usr/bin/env python3
"""
Analyze extraction difficulty of 3 Bible PDFs to determine which would be easiest to extract reliably.
"""

import pdfplumber
import re
from pathlib import Path

def analyze_pdf_formatting(pdf_path, sample_pages=(50, 60, 70, 80, 90, 100)):
    """Analyze PDF formatting patterns on sample pages."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {Path(pdf_path).name}")
    print(f"{'='*60}")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"Total pages: {total_pages}")
            
            # Analyze sample pages
            verse_patterns = []
            header_patterns = []
            
            for page_num in sample_pages:
                if page_num >= total_pages:
                    continue
                    
                page = pdf.pages[page_num]
                text = page.extract_text()
                
                if not text:
                    continue
                    
                print(f"\n--- PAGE {page_num + 1} SAMPLE ---")
                print(text[:500] + "..." if len(text) > 500 else text)
                
                # Analyze verse number patterns
                lines = text.split('\n')
                verse_start_lines = 0
                inline_verses = 0
                header_lines = 0
                
                for line in lines[:20]:  # First 20 lines for pattern analysis
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check for verse numbers at start of line
                    if re.match(r'^\d+\s+', line):
                        verse_start_lines += 1
                    # Check for inline verse numbers
                    elif re.search(r'\s\d+\s', line):
                        inline_verses += 1
                    # Check for book/chapter headers
                    if re.search(r'[A-Z][a-z]+\s+\d+:\d+', line) or re.match(r'^[A-Z\s]+$', line):
                        header_lines += 1
                
                verse_patterns.append({
                    'page': page_num + 1,
                    'verse_start_lines': verse_start_lines,
                    'inline_verses': inline_verses,
                    'header_lines': header_lines
                })
            
            # Summarize patterns
            print(f"\n--- FORMATTING ANALYSIS ---")
            avg_verse_start = sum(p['verse_start_lines'] for p in verse_patterns) / len(verse_patterns)
            avg_inline = sum(p['inline_verses'] for p in verse_patterns) / len(verse_patterns)
            avg_headers = sum(p['header_lines'] for p in verse_patterns) / len(verse_patterns)
            
            print(f"Average verse-start lines per page: {avg_verse_start:.1f}")
            print(f"Average inline verse numbers per page: {avg_inline:.1f}")
            print(f"Average header lines per page: {avg_headers:.1f}")
            
            # Determine difficulty
            if avg_verse_start > 5:
                difficulty = "Easy"
                reason = "Verses consistently start lines with numbers"
            elif avg_inline > 5 and avg_verse_start < 2:
                difficulty = "Medium"
                reason = "Verses are inline in paragraphs but consistent"
            else:
                difficulty = "Hard"
                reason = "Inconsistent or mixed formatting patterns"
            
            print(f"\nEXTRACTION DIFFICULTY: {difficulty}")
            print(f"Reasoning: {reason}")
            
            return {
                'pdf_name': Path(pdf_path).name,
                'difficulty': difficulty,
                'reason': reason,
                'avg_verse_start': avg_verse_start,
                'avg_inline': avg_inline,
                'avg_headers': avg_headers,
                'total_pages': total_pages
            }
            
    except Exception as e:
        print(f"Error analyzing {pdf_path}: {e}")
        return None

def main():
    pdf_files = [
        "bible_king_james_version.pdf",
        "new-revised-standard-version.pdf", 
        "english-standard-version-bible.pdf"
    ]
    
    results = []
    
    for pdf_file in pdf_files:
        if Path(pdf_file).exists():
            result = analyze_pdf_formatting(pdf_file)
            if result:
                results.append(result)
        else:
            print(f"File not found: {pdf_file}")
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("EXTRACTION DIFFICULTY COMPARISON")
    print(f"{'='*60}")
    
    for result in results:
        print(f"\n{result['pdf_name']}:")
        print(f"  Difficulty: {result['difficulty']}")
        print(f"  Reason: {result['reason']}")
        print(f"  Verse-start lines: {result['avg_verse_start']:.1f}/page")
        print(f"  Inline verses: {result['avg_inline']:.1f}/page")
        print(f"  Headers: {result['avg_headers']:.1f}/page")
    
    # Recommendation
    if results:
        easiest = min(results, key=lambda x: {
            'Easy': 1, 'Medium': 2, 'Hard': 3
        }[x['difficulty']])
        
        print(f"\n{'='*60}")
        print("RECOMMENDATION")
        print(f"{'='*60}")
        print(f"Most reliable for extraction: {easiest['pdf_name']}")
        print(f"Difficulty: {easiest['difficulty']}")
        print(f"Reason: {easiest['reason']}")

if __name__ == "__main__":
    main()