#!/usr/bin/env python3
"""
PDF Processor for Bible Extraction System
Handles PDF text extraction with dynamic book detection and metadata analysis
"""

import re
import yaml
import pdfplumber
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PageObject:
    """Represents a single page from the PDF"""
    page_num: int
    raw_text: str
    metadata: Dict

@dataclass
class BookDetectionResult:
    """Results from book discovery phase"""
    detected_books: List[str]
    canon_type: str
    confidence: float
    pdf_format: str

class PDFProcessor:
    """Handles PDF processing and book discovery"""
    
    def __init__(self, config_path: str = "config/pdfs.yaml"):
        self.config = self._load_config(config_path)
        self.canon_data = self._load_canon_data()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load PDF configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config {config_path}: {e}")
            return {}
    
    def _load_canon_data(self) -> Dict:
        """Load canon database"""
        try:
            canon_path = Path("data/canon_databases/books.yaml")
            with open(canon_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load canon data: {e}")
            return {}
    
    def discover_books(self, pdf_path: str, sample_pages: Optional[List[int]] = None) -> BookDetectionResult:
        """
        Discover books in PDF by scanning sample pages
        
        Args:
            pdf_path: Path to PDF file
            sample_pages: List of page numbers to sample (None for auto-selection)
            
        Returns:
            BookDetectionResult with detected books and metadata
        """
        logger.info(f"Starting book discovery for {pdf_path}")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                
                # Select sample pages if not provided
                if sample_pages is None:
                    sample_pages = self._select_sample_pages(total_pages)
                
                logger.info(f"Sampling {len(sample_pages)} pages for book discovery")
                
                # Extract text from sample pages
                sample_texts = []
                for page_num in sample_pages:
                    if page_num < total_pages:
                        page = pdf.pages[page_num]
                        text = page.extract_text()
                        if text:
                            sample_texts.append({
                                'page_num': page_num + 1,
                                'text': text
                            })
                
                # Detect books from samples
                detected_books = self._extract_books_from_samples(sample_texts)
                logger.info(f"Book extraction completed: {type(detected_books)}")
                
                # Identify canon type
                try:
                    canon_type, confidence = self._identify_canon_type(detected_books)
                except Exception as e:
                    logger.error(f"Canon identification failed: {e}")
                    canon_type, confidence = "unknown", 0.0
                
                # Detect PDF format
                pdf_format = self._detect_pdf_format(sample_texts)
                
                logger.info(f"Detected {len(detected_books)} books, canon: {canon_type}, format: {pdf_format}")
                
                return BookDetectionResult(
                    detected_books=detected_books,
                    canon_type=canon_type,
                    confidence=confidence,
                    pdf_format=pdf_format
                )
                
        except Exception as e:
            logger.error(f"Book discovery failed: {e}")
            return BookDetectionResult([], "unknown", 0.0, "unknown")
    
    def _select_sample_pages(self, total_pages: int) -> List[int]:
        """Select representative pages for sampling"""
        if total_pages <= 50:
            return list(range(total_pages))
        
        # Sample evenly throughout the PDF
        sample_count = min(50, total_pages // 10)  # 10% of pages, max 50
        interval = total_pages // sample_count
        
        sample_pages = []
        for i in range(sample_count):
            page_num = i * interval
            sample_pages.append(page_num)
        
        return sample_pages
    
    def _extract_books_from_samples(self, sample_texts: List[Dict]) -> List[str]:
        """Extract book names from sample texts"""
        all_books = set()
        
        # Pattern from config
        patterns = self.config.get('auto_detection', {}).get('book_name_patterns', [])
        
        for sample in sample_texts:
            text = sample['text']
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or len(line) < 3:
                    continue
                
                # Try each pattern
                for pattern in patterns:
                    try:
                        matches = re.findall(pattern, line, re.IGNORECASE)
                        for match in matches:
                            if isinstance(match, tuple):
                                match = match[0]
                            
                            if not isinstance(match, str):
                                continue
                            
                            # Validate against canon databases
                            if self._is_valid_book_name(match):
                                standardized = self._standardize_book_name(match)
                                if isinstance(standardized, str):
                                    all_books.add(standardized)
                    except re.error as e:
                        logger.warning(f"Regex pattern failed: {pattern}, error: {e}")
                        continue
        
        result = sorted(list(all_books))
        logger.info(f"Extracted books: {result}")
        return result
    
    def _is_valid_book_name(self, name: str) -> bool:
        """Check if name matches any book in canon databases"""
        name = name.strip().title()
        
        # Check all canon types
        for canon_type in ['protestant_books', 'catholic_books', 'orthodox_books']:
            if canon_type in self.canon_data:
                for testament in ['old_testament', 'new_testament']:
                    if testament in self.canon_data[canon_type]:
                        books = self.canon_data[canon_type][testament]
                        if name in books:
                            return True
        
        # Check variations
        if 'book_variations' in self.canon_data:
            for standard_name, variations in self.canon_data['book_variations'].items():
                if name in [v.title() for v in variations]:
                    return True
        
        return False
    
    def _standardize_book_name(self, name: str) -> str:
        """Standardize book name to canonical form"""
        name = name.strip().title()
        
        # Check variations and return standard name
        if 'book_variations' in self.canon_data:
            for standard_name, variations in self.canon_data['book_variations'].items():
                if name in [v.title() for v in variations] or name == standard_name:
                    return standard_name
        
        return name
    
    def _looks_like_book_header(self, line: str) -> bool:
        """Check if line looks like a book header"""
        # Common patterns for book headers
        patterns = [
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+\d+$',  # "Genesis 1"
            r'^\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+\d+$',  # "51 Genesis 24"
            r'^[A-Z\s]+$',  # All caps "GENESIS"
        ]
        
        for pattern in patterns:
            if re.match(pattern, line):
                return True
        
        return False
    
    def _extract_book_from_header(self, line: str) -> Optional[str]:
        """Extract book name from header line"""
        # Remove page numbers and chapter numbers
        patterns = [
            r'^\d+\s+([A-Za-z\s]+?)\s+\d+$',  # "51 Genesis 24" -> "Genesis"
            r'^([A-Za-z\s]+?)\s+\d+$',  # "Genesis 1" -> "Genesis"
            r'^([A-Z\s]+)$',  # "GENESIS" -> "Genesis"
        ]
        
        for pattern in patterns:
            match = re.match(pattern, line.strip())
            if match:
                book_name = match.group(1).strip()
                return book_name
        
        return None
    
    def _identify_canon_type(self, detected_books: List[str]) -> Tuple[str, float]:
        """Identify canon type based on detected books"""
        if not detected_books:
            return "unknown", 0.0
        
        # Ensure detected_books is a list of strings
        if isinstance(detected_books, set):
            detected_books = list(detected_books)
        
        # Debug canon data structure
        logger.info(f"Canon data keys: {list(self.canon_data.keys())}")
        
        # Count matches for each canon type
        canon_matches = {}
        
        for canon_type, canon_data in self.canon_data.items():
            if not isinstance(canon_data, dict) or not canon_type.endswith('_books') or canon_type == 'book_variations':
                continue
            
            canon_books = []
            for testament in ['old_testament', 'new_testament']:
                if testament in canon_data and isinstance(canon_data[testament], list):
                    canon_books.extend(canon_data[testament])
            
            matches = len(set(detected_books) & set(canon_books))
            total = len(canon_books)
            confidence = matches / total if total > 0 else 0
            
            canon_matches[canon_type.replace('_books', '')] = {
                'matches': matches,
                'total': total,
                'confidence': confidence
            }
        
        # Debug logging
        logger.info(f"Canon matches: {canon_matches}")
        
        # Find best match
        if canon_matches:
            best_canon = max(canon_matches.items(), key=lambda x: x[1]['confidence'])
            return best_canon[0], best_canon[1]['confidence']
        else:
            return "unknown", 0.0
    
    def _detect_pdf_format(self, sample_texts: List[Dict]) -> str:
        """Detect PDF format based on verse patterns"""
        verse_start_count = 0
        inline_count = 0
        total_lines = 0
        
        for sample in sample_texts:
            text = sample['text']
            lines = text.split('\n')
            
            for line in lines[:20]:  # Check first 20 lines per sample
                line = line.strip()
                if not line:
                    continue
                
                total_lines += 1
                
                # Check for verse-start pattern
                if re.match(r'^\d+\s+', line):
                    verse_start_count += 1
                
                # Check for inline pattern
                if re.search(r'\s\d+\s', line):
                    inline_count += 1
        
        if total_lines == 0:
            return "unknown"
        
        verse_start_ratio = verse_start_count / total_lines
        inline_ratio = inline_count / total_lines
        
        if verse_start_ratio > 0.3:
            return "verse_start_lines"
        elif inline_ratio > 0.2:
            return "inline_verses"
        else:
            return "compact_inline"
    
    def extract_pages(self, pdf_path: str, start_page: int = 0, end_page: Optional[int] = None) -> List[PageObject]:
        """
        Extract pages from PDF
        
        Args:
            pdf_path: Path to PDF file
            start_page: Starting page number (0-based)
            end_page: Ending page number (None for all pages)
            
        Returns:
            List of PageObject instances
        """
        logger.info(f"Extracting pages {start_page} to {end_page or 'end'} from {pdf_path}")
        
        pages = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                if end_page is None:
                    end_page = total_pages
                
                end_page = min(end_page, total_pages)
                
                for page_num in range(start_page, end_page):
                    page = pdf.pages[page_num]
                    text = page.extract_text()
                    
                    if text:
                        metadata = {
                            'pdf_format': self._detect_pdf_format([{'text': text}]),
                            'line_count': len(text.split('\n')),
                            'char_count': len(text),
                            'has_headers': self._has_headers(text)
                        }
                        
                        pages.append(PageObject(
                            page_num=page_num + 1,  # 1-based for user
                            raw_text=text,
                            metadata=metadata
                        ))
                
                logger.info(f"Extracted {len(pages)} pages")
                return pages
                
        except Exception as e:
            logger.error(f"Page extraction failed: {e}")
            return []
    
    def _has_headers(self, text: str) -> bool:
        """Check if text has likely book/chapter headers"""
        lines = text.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            if self._looks_like_book_header(line.strip()):
                return True
        return False

if __name__ == "__main__":
    # Test the processor
    processor = PDFProcessor()
    
    # Test book discovery
    result = processor.discover_books("../bible_king_james_version.pdf")
    print(f"Detected books: {result.detected_books}")
    print(f"Canon type: {result.canon_type}")
    print(f"Confidence: {result.confidence}")
    print(f"PDF format: {result.pdf_format}")