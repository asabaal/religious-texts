#!/usr/bin/env python3
"""
AI Extraction Engine for Bible Extraction System
Uses Ollama + Qwen3 to intelligently extract verses from PDF pages
"""

import json
import yaml
import requests
import logging
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

try:
    from .pdf_processor import PageObject, BookDetectionResult
except ImportError:
    from pdf_processor import PageObject, BookDetectionResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VerseObject:
    """Represents a single extracted verse"""
    book: str
    chapter: int
    verse: int
    text: str
    page: int
    confidence: float
    flags: List[str]
    context: Dict

@dataclass
class ExtractionContext:
    """Context for verse extraction"""
    current_book: Optional[str]
    current_chapter: Optional[int]
    last_verse: Optional[int]
    detected_books: List[str]
    pdf_format: str
    canon_type: str

class AIExtractor:
    """AI-powered verse extraction using Ollama"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = "config/models.yaml"
            # Try relative path if not found
            if not Path(config_path).exists():
                config_path = "../config/models.yaml"
        self.config = self._load_config(config_path)
        self.ollama_base_url = "http://localhost:11434"
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load model configuration"""
        if config_path is None:
            logger.error("Config path is None")
            return {}
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config {config_path}: {e}")
            return {}
    
    def _build_context_window(self, pages: List[PageObject], current_index: int, context_size: int = 3) -> str:
        """Build context window around current page"""
        start_idx = max(0, current_index - context_size // 2)
        end_idx = min(len(pages), current_index + context_size // 2 + 1)
        
        context_parts = []
        for i in range(start_idx, end_idx):
            page = pages[i]
            marker = ">>> CURRENT PAGE <<<" if i == current_index else f"Page {page.page_num}"
            context_parts.append(f"{marker}:\n{page.raw_text}")
        
        return "\n\n".join(context_parts)
    
    def _create_extraction_prompt(self, context: ExtractionContext, page_text: str, context_window: str) -> str:
        """Create prompt for AI extraction"""
        
        system_prompt = self.config.get('prompting', {}).get('system_prompt', 
            "You are a Bible text extraction expert with deep knowledge of Biblical canon, verse structure, and manuscript traditions.")
        
        prompt = f"""{system_prompt}

EXTRACTION CONTEXT:
- Current book: {context.current_book or 'Unknown'}
- Current chapter: {context.current_chapter or 'Unknown'}
- Last verse: {context.last_verse or 'Unknown'}
- PDF format: {context.pdf_format}
- Canon type: {context.canon_type}

DETECTED BOOKS IN THIS BIBLE:
{', '.join(context.detected_books)}

CONTEXT FROM NEARBY PAGES:
{context_window}

CURRENT PAGE TO EXTRACT:
{page_text}

TASK:
Extract all Bible verses from the current page with perfect accuracy. Follow these rules:

1. Identify book/chapter headers and validate against detected books
2. Extract every verse with correct book, chapter, and verse numbers
3. Handle footnotes, section headers, page numbers appropriately
4. Maintain verse sequence continuity
5. Assign confidence scores (0.0-1.0) based on extraction certainty
6. Flag any special conditions (spanning verses, ambiguous references, etc.)

RESPONSE FORMAT:
Return a JSON array of verse objects with this exact structure:
[
  {{
    "book": "Genesis",
    "chapter": 1,
    "verse": 1,
    "text": "In the beginning God created the heaven and the earth.",
    "confidence": 0.95,
    "flags": ["starts_chapter", "complete_sentence"],
    "notes": "Any extraction notes or concerns"
  }}
]

IMPORTANT:
- Only include actual Bible verses, not page numbers or headers
- Use exact book names from the detected books list
- Maintain proper verse numbering sequence
- If no verses found, return an empty array []
- Response must be valid JSON only, no explanations"""

        return prompt
    
    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Make API call to Ollama"""
        try:
            url = f"{self.ollama_base_url}/api/generate"
            
            payload = {
                "model": self.config.get('ollama', {}).get('model', 'qwen2.5:30b'),
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.get('ollama', {}).get('temperature', 0.1),
                    "num_predict": self.config.get('ollama', {}).get('max_tokens', 8000)
                }
            }
            
            logger.info(f"Calling Ollama with model: {payload['model']}")
            
            response = requests.post(
                url, 
                json=payload, 
                timeout=self.config.get('ollama', {}).get('timeout', 120)
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to call Ollama: {e}")
            return None
    
    def _parse_ai_response(self, response: str) -> List[Dict]:
        """Parse AI response into verse objects"""
        if not response:
            return []
        
        try:
            # Try to extract JSON from response
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()
            
            # Parse JSON
            verses_data = json.loads(response)
            
            if isinstance(verses_data, list):
                return verses_data
            else:
                logger.warning(f"Expected list, got {type(verses_data)}")
                return []
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.error(f"Response content: {response[:500]}...")
            return []
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return []
    
    def _validate_verse_object(self, verse_data: Dict, context: ExtractionContext) -> Optional[VerseObject]:
        """Validate and convert verse data to VerseObject"""
        try:
            # Required fields
            book = verse_data.get('book')
            chapter = verse_data.get('chapter')
            verse = verse_data.get('verse')
            text = verse_data.get('text')
            
            if not all([book, chapter is not None, verse is not None, text]):
                logger.warning(f"Missing required fields in verse: {verse_data}")
                return None
            
            # Type checking and validation
            if not isinstance(book, str):
                logger.warning(f"Invalid book type: {type(book)}")
                return None
            
            if not isinstance(text, str):
                logger.warning(f"Invalid text type: {type(text)}")
                return None
            
            # Validate book name
            if book not in context.detected_books:
                logger.warning(f"Book '{book}' not in detected books list")
                # Could still be valid if book detection missed it
            
            # Validate chapter and verse numbers
            if not isinstance(chapter, int) or chapter < 1:
                logger.warning(f"Invalid chapter number: {chapter}")
                return None
            
            if not isinstance(verse, int) or verse < 1:
                logger.warning(f"Invalid verse number: {verse}")
                return None
            
            # Create VerseObject
            return VerseObject(
                book=book,
                chapter=chapter,
                verse=verse,
                text=text.strip(),
                page=0,  # Will be set by caller
                confidence=float(verse_data.get('confidence', 0.8)),
                flags=verse_data.get('flags', []),
                context={
                    'notes': verse_data.get('notes', ''),
                    'extraction_method': 'ai'
                }
            )
            
        except Exception as e:
            logger.error(f"Error validating verse object: {e}")
            return None
    
    def extract_verses_from_page(self, 
                                page: PageObject, 
                                pages: List[PageObject], 
                                page_index: int,
                                context: ExtractionContext) -> List[VerseObject]:
        """
        Extract verses from a single page using AI
        
        Args:
            page: Current page to extract from
            pages: All pages for context
            page_index: Index of current page in pages list
            context: Current extraction context
            
        Returns:
            List of extracted VerseObject instances
        """
        logger.info(f"Extracting verses from page {page.page_num}")
        
        # Build context window
        context_window = self._build_context_window(pages, page_index)
        
        # Create prompt
        prompt = self._create_extraction_prompt(context, page.raw_text, context_window)
        
        # Call AI
        response = self._call_ollama(prompt)
        if not response:
            logger.error(f"No response from AI for page {page.page_num}")
            return []
        
        # Parse response
        verses_data = self._parse_ai_response(response)
        if not verses_data:
            logger.info(f"No verses extracted from page {page.page_num}")
            return []
        
        # Validate and convert to VerseObject
        verses = []
        for verse_data in verses_data:
            verse_obj = self._validate_verse_object(verse_data, context)
            if verse_obj:
                verse_obj.page = page.page_num
                verses.append(verse_obj)
        
        logger.info(f"Extracted {len(verses)} verses from page {page.page_num}")
        return verses
    
    def extract_verses_batch(self, 
                            pages: List[PageObject], 
                            start_index: int,
                            book_detection: BookDetectionResult) -> Tuple[List[VerseObject], ExtractionContext]:
        """
        Extract verses from a batch of pages
        
        Args:
            pages: List of pages to process
            start_index: Starting index in pages list
            book_detection: Results from book discovery phase
            
        Returns:
            Tuple of (extracted verses, updated context)
        """
        logger.info(f"Extracting verses from batch of {len(pages)} pages starting at index {start_index}")
        
        # Initialize context
        context = ExtractionContext(
            current_book=None,
            current_chapter=None,
            last_verse=None,
            detected_books=book_detection.detected_books,
            pdf_format=book_detection.pdf_format,
            canon_type=book_detection.canon_type
        )
        
        all_verses = []
        
        for i, page in enumerate(pages):
            try:
                verses = self.extract_verses_from_page(page, pages, i, context)
                all_verses.extend(verses)
                
                # Update context from last extracted verse
                if verses:
                    last_verse = verses[-1]
                    context.current_book = last_verse.book
                    context.current_chapter = last_verse.chapter
                    context.last_verse = last_verse.verse
                
            except Exception as e:
                logger.error(f"Error processing page {page.page_num}: {e}")
                continue
        
        logger.info(f"Batch extraction complete: {len(all_verses)} verses extracted")
        return all_verses, context

if __name__ == "__main__":
    # Test the extractor
    extractor = AIExtractor()
    
    # Simple test
    test_prompt = "Respond with a JSON array containing one test object: [{'test': 'value'}]"
    response = extractor._call_ollama(test_prompt)
    print(f"Ollama response: {response}")