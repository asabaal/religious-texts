#!/usr/bin/env python3
"""
Proper KJV Bible PDF extractor that handles the actual format patterns.
Based on analysis of pages 50-60, the format is:
- Page headers: "50 Genesis 23" (page number + book + chapter)
- Standalone chapter numbers: "24"
- Section headers: "A Wife Obtained for Isaac"
- Verse numbers: Start lines with numbers 1-176
- Paragraph markers: ¶ symbol
- Footnotes: [red], [contention], etc.
"""

import pdfplumber
import re
import json
from typing import Dict, List, Tuple, Optional

class KJVExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.verses = []
        self.current_book = None
        self.current_chapter = None
        self.current_page = None
        
    def is_page_header(self, line: str) -> Optional[Tuple[int, str, int]]:
        """Check if line is a page header like '50 Genesis 23'"""
        match = re.match(r'^(\d+)\s+([A-Za-z]+)\s+(\d+)$', line.strip())
        if match:
            page_num = int(match.group(1))
            book = match.group(2)
            chapter = int(match.group(3))
            return (page_num, book, chapter)
        return None
    
    def is_standalone_chapter(self, line: str) -> Optional[int]:
        """Check if line is a standalone chapter number"""
        if line.strip().isdigit() and 1 <= int(line.strip()) <= 150:
            return int(line.strip())
        return None
    
    def is_verse_number(self, line: str) -> Optional[int]:
        """Check if line starts with a verse number (1-176 for Psalms)"""
        match = re.match(r'^(\d+)\s+', line.strip())
        if match:
            verse_num = int(match.group(1))
            # Reasonable verse number range
            if 1 <= verse_num <= 176:
                return verse_num
        return None
    
    def is_section_header(self, line: str) -> bool:
        """Check if line is a section header (not a verse)"""
        # Section headers typically don't start with numbers under 1000
        # and often contain capitalization patterns
        line_stripped = line.strip()
        
        # If it starts with a number > 100, it's likely a page number or other content
        if re.match(r'^\d{3,}', line_stripped):
            return True
            
        # If it has no numbers and is title-cased
        if not re.search(r'\d', line_stripped) and len(line_stripped) > 0:
            # Check if it looks like a title (multiple words, capital letters)
            words = line_stripped.split()
            if len(words) > 1 and any(word[0].isupper() for word in words):
                return True
                
        return False
    
    def clean_verse_text(self, text: str) -> str:
        """Clean verse text by removing footnotes and normalizing whitespace"""
        # Remove footnotes in brackets
        text = re.sub(r'\s*\[.*?\]', '', text)
        # Remove paragraph markers
        text = text.replace('¶', '')
        # Normalize whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    def extract_page_content(self, page_text: str) -> List[Dict]:
        """Extract verses from a single page of text"""
        lines = page_text.split('\n')
        page_verses = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            # Check for page header
            header_match = self.is_page_header(line)
            if header_match:
                page_num, book, chapter = header_match
                self.current_page = page_num
                self.current_book = book
                self.current_chapter = chapter
                print(f"Page {page_num}: {book} {chapter}")
                i += 1
                continue
            
            # Check for standalone chapter number
            chapter_num = self.is_standalone_chapter(line)
            if chapter_num and self.current_book:
                self.current_chapter = chapter_num
                print(f"  Chapter {chapter_num}")
                i += 1
                continue
            
            # Check for section header
            if self.is_section_header(line):
                print(f"  Section: {line}")
                i += 1
                continue
            
            # Check for verse number
            verse_num = self.is_verse_number(line)
            if verse_num and self.current_book and self.current_chapter:
                # Extract verse text (may span multiple lines)
                verse_text = line
                
                # Look ahead for continuation lines
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    # Stop if we hit another verse, header, or empty line
                    if (not next_line or 
                        self.is_page_header(next_line) or
                        self.is_standalone_chapter(next_line) or
                        self.is_section_header(next_line) or
                        self.is_verse_number(next_line)):
                        break
                    verse_text += ' ' + next_line
                    j += 1
                
                # Clean and store the verse
                cleaned_text = self.clean_verse_text(verse_text)
                # Remove the verse number from the beginning of the text
                cleaned_text = re.sub(r'^\d+\s+', '', cleaned_text)
                if cleaned_text:
                    verse = {
                        'book': self.current_book,
                        'chapter': self.current_chapter,
                        'verse': verse_num,
                        'text': cleaned_text,
                        'page': self.current_page
                    }
                    page_verses.append(verse)
                    print(f"    {verse_num}: {cleaned_text[:50]}...")
                
                i = j
                continue
            
            i += 1
        
        return page_verses
    
    def extract_all(self, max_pages: Optional[int] = None) -> List[Dict]:
        """Extract all verses from the PDF"""
        all_verses = []
        
        with pdfplumber.open(self.pdf_path) as pdf:
            total_pages = len(pdf.pages)
            if max_pages:
                total_pages = min(max_pages, total_pages)
            
            print(f"Extracting from {total_pages} pages...")
            
            for page_num in range(total_pages):
                page = pdf.pages[page_num]
                text = page.extract_text()
                
                if text:
                    page_verses = self.extract_page_content(text)
                    all_verses.extend(page_verses)
        
        return all_verses
    
    def save_to_file(self, verses: List[Dict], output_path: str):
        """Save verses to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(verses, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved {len(verses)} verses to {output_path}")

def main():
    extractor = KJVExtractor('bible_king_james_version.pdf')
    
    # Full extraction
    print("Starting full KJV Bible extraction...")
    verses = extractor.extract_all()  # No max_pages limit
    
    # Save results
    extractor.save_to_file(verses, 'kjv_bible_complete.json')
    
    # Print statistics
    books = sorted(set(v['book'] for v in verses))
    chapters = set((v['book'], v['chapter']) for v in verses)
    print(f"\nFinal Extraction Statistics:")
    print(f"Total verses: {len(verses):,}")
    print(f"Books found: {len(books)}")
    print(f"Chapters found: {len(chapters)}")
    print(f"Books: {books[:10]}{'...' if len(books) > 10 else ''}")
    
    # Show some sample verses from different books
    print(f"\nSample verses:")
    sample_indices = [0, len(verses)//4, len(verses)//2, 3*len(verses)//4, -1]
    for i in sample_indices:
        if 0 <= i < len(verses):
            verse = verses[i]
            print(f"{verse['book']} {verse['chapter']}:{verse['verse']} - {verse['text'][:80]}...")

if __name__ == "__main__":
    main()