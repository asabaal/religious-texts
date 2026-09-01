#!/usr/bin/env python3
"""
Detailed analysis of PDF formatting patterns with more comprehensive samples.
"""

import pdfplumber
import re
from pathlib import Path

def get_detailed_sample(pdf_path, page_num, lines=30):
    """Get detailed sample from a specific page."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num >= len(pdf.pages):
                return None
                
            page = pdf.pages[page_num]
            text = page.extract_text()
            
            if not text:
                return None
                
            text_lines = text.split('\n')
            return '\n'.join(text_lines[:lines])
            
    except Exception as e:
        return f"Error: {e}"

def analyze_verse_patterns(text):
    """Analyze verse number patterns in text."""
    lines = text.split('\n') if text else []
    patterns = {
        'verse_start': [],
        'inline_verses': [],
        'potential_headers': [],
        'footnotes': [],
        'paragraph_breaks': []
    }
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            patterns['paragraph_breaks'].append(i)
            continue
            
        # Verse starts line
        if re.match(r'^\d+\s+', line):
            patterns['verse_start'].append((i, line))
        # Inline verse numbers
        elif re.search(r'\s\d+\s', line):
            matches = re.finditer(r'\s(\d+)\s', line)
            for match in matches:
                patterns['inline_verses'].append((i, match.group(1), line))
        # Potential headers (all caps, book names)
        elif re.match(r'^[A-Z\s]{3,}$', line) or re.search(r'[A-Z][a-z]+\s+\d+:\d+', line):
            patterns['potential_headers'].append((i, line))
        # Footnotes (superscript-style numbers)
        elif re.search(r'\[\d+\]|\d+\]|\[\d+', line):
            patterns['footnotes'].append((i, line))
    
    return patterns

def main():
    pdf_files = [
        ("bible_king_james_version.pdf", "KJV"),
        ("new-revised-standard-version.pdf", "NRSV"), 
        ("english-standard-version-bible.pdf", "ESV")
    ]
    
    sample_pages = [50, 75, 100]  # Different pages for variety
    
    for pdf_file, name in pdf_files:
        if not Path(pdf_file).exists():
            print(f"File not found: {pdf_file}")
            continue
            
        print(f"\n{'='*80}")
        print(f"DETAILED ANALYSIS: {name} ({pdf_file})")
        print(f"{'='*80}")
        
        for page_num in sample_pages:
            print(f"\n--- PAGE {page_num + 1} ---")
            sample = get_detailed_sample(pdf_file, page_num, lines=25)
            
            if sample and not sample.startswith("Error:"):
                print(sample)
                print("\n" + "-"*50)
                
                patterns = analyze_verse_patterns(sample)
                print(f"Verse-start lines: {len(patterns['verse_start'])}")
                print(f"Inline verse numbers: {len(patterns['inline_verses'])}")
                print(f"Potential headers: {len(patterns['potential_headers'])}")
                print(f"Footnotes: {len(patterns['footnotes'])}")
                
                if patterns['verse_start']:
                    print("Verse-start examples:")
                    for i, line in patterns['verse_start'][:3]:
                        print(f"  {line}")
                        
                if patterns['inline_verses']:
                    print("Inline verse examples:")
                    for i, num, line in patterns['inline_verses'][:3]:
                        print(f"  Verse {num}: {line}")
                        
                if patterns['potential_headers']:
                    print("Header examples:")
                    for i, line in patterns['potential_headers']:
                        print(f"  {line}")
            else:
                print(f"Could not extract page {page_num + 1}")

if __name__ == "__main__":
    main()