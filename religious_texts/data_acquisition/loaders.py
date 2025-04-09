"""
Text Loaders Module

This module provides functions for loading biblical texts from various file formats
and sources.
"""

import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup


def load_text(file_path: Union[str, Path], encoding: str = 'utf-8') -> Dict[str, Any]:
    """
    Load a plain text biblical text file and parse its contents.
    
    Args:
        file_path: Path to the text file
        encoding: Character encoding of the file (default: utf-8)
        
    Returns:
        Dictionary containing the parsed biblical text with books as keys
        
    Example:
        >>> bible = load_text('data/kjv.txt')
        >>> genesis = bible['Genesis']
        >>> print(genesis[1][1])  # Genesis 1:1
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        
        # Determine the format based on content and extension
        ext = file_path.suffix.lower()
        
        if ext == '.txt':
            from religious_texts.data_acquisition.parsers import parse_plaintext
            return parse_plaintext(content)
        elif ext == '.xml' or ext == '.osis':
            return load_xml(file_path, encoding)
        elif ext == '.json':
            return load_json(file_path, encoding)
        else:
            # Try to infer the format
            if '<?xml' in content[:100]:
                from religious_texts.data_acquisition.parsers import parse_osis
                return parse_osis(content)
            elif content.strip().startswith('{'):
                return json.loads(content)
            else:
                from religious_texts.data_acquisition.parsers import parse_plaintext
                return parse_plaintext(content)
                
    except Exception as e:
        raise IOError(f"Error loading {file_path}: {str(e)}")


def load_xml(file_path: Union[str, Path], encoding: str = 'utf-8') -> Dict[str, Any]:
    """
    Load a biblical text in XML format (OSIS, USFM, etc.).
    
    Args:
        file_path: Path to the XML file
        encoding: Character encoding of the file (default: utf-8)
        
    Returns:
        Dictionary containing the parsed biblical text
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        
        # Determine XML format
        if '<osis' in content[:1000]:
            from religious_texts.data_acquisition.parsers import parse_osis
            return parse_osis(content)
        elif '<usfm' in content[:1000] or '\\id' in content[:1000]:
            from religious_texts.data_acquisition.parsers import parse_usfm
            return parse_usfm(content)
        elif '<sword' in content[:1000]:
            from religious_texts.data_acquisition.parsers import parse_sword
            return parse_sword(content)
        else:
            # Generic XML parsing
            root = ET.fromstring(content)
            # Extract basic structure
            books = {}
            
            # Basic extraction of books/chapters/verses
            for book in root.findall(".//book") or root.findall(".//div[@type='book']"):
                book_name = book.get('name') or book.get('osisID') or ''
                if not book_name:
                    continue
                    
                chapters = {}
                for chapter in book.findall(".//chapter") or book.findall(".//div[@type='chapter']"):
                    chapter_num = chapter.get('number') or chapter.get('osisID') or ''
                    if not chapter_num and 'n' in chapter.attrib:
                        chapter_num = chapter.get('n')
                    
                    try:
                        chapter_num = int(chapter_num.split('.')[-1])
                    except (ValueError, AttributeError):
                        continue
                        
                    verses = {}
                    for verse in chapter.findall(".//verse") or chapter.findall(".//div[@type='verse']"):
                        verse_num = verse.get('number') or verse.get('osisID') or ''
                        if not verse_num and 'n' in verse.attrib:
                            verse_num = verse.get('n')
                            
                        try:
                            verse_num = int(verse_num.split('.')[-1])
                        except (ValueError, AttributeError):
                            continue
                            
                        verse_text = ''.join(verse.itertext()).strip()
                        verses[verse_num] = verse_text
                        
                    if verses:
                        chapters[chapter_num] = verses
                
                if chapters:
                    books[book_name] = chapters
            
            if not books:
                warnings.warn(f"No structured content found in {file_path}. Using generic XML parser.")
            
            return books
            
    except Exception as e:
        raise IOError(f"Error loading XML file {file_path}: {str(e)}")


def load_json(file_path: Union[str, Path], encoding: str = 'utf-8') -> Dict[str, Any]:
    """
    Load a biblical text in JSON format.
    
    Args:
        file_path: Path to the JSON file
        encoding: Character encoding of the file (default: utf-8)
        
    Returns:
        Dictionary containing the parsed biblical text
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            data = json.load(f)
        
        # Validate expected structure
        if isinstance(data, dict) and any(isinstance(data.get(key), dict) for key in data):
            # Check if the structure looks like a biblical text
            if any(isinstance(book, dict) and any(isinstance(book.get(str(ch)), dict) for ch in range(1, 151)) 
                  for book in data.values()):
                return data
            else:
                warnings.warn("JSON structure doesn't appear to follow expected Bible format.")
                return data
        else:
            warnings.warn("JSON structure doesn't match expected format.")
            return data
            
    except Exception as e:
        raise IOError(f"Error loading JSON file {file_path}: {str(e)}")


def load_csv(file_path: Union[str, Path], book_col: str = 'book', 
             chapter_col: str = 'chapter', verse_col: str = 'verse', 
             text_col: str = 'text', encoding: str = 'utf-8') -> Dict[str, Any]:
    """
    Load a biblical text from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        book_col: Column name for book names (default: 'book')
        chapter_col: Column name for chapter numbers (default: 'chapter')
        verse_col: Column name for verse numbers (default: 'verse')
        text_col: Column name for verse text (default: 'text')
        encoding: Character encoding of the file (default: utf-8)
        
    Returns:
        Dictionary containing the parsed biblical text
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        
        # Validate column names
        required_cols = [book_col, chapter_col, verse_col, text_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
            
        # Convert to structured dictionary
        bible = {}
        
        for book_name, book_df in df.groupby(book_col):
            chapters = {}
            
            for chapter_num, chapter_df in book_df.groupby(chapter_col):
                verses = {}
                
                for _, row in chapter_df.iterrows():
                    verse_num = row[verse_col]
                    verse_text = row[text_col]
                    verses[int(verse_num)] = verse_text
                    
                chapters[int(chapter_num)] = verses
                
            bible[book_name] = chapters
            
        return bible
        
    except Exception as e:
        raise IOError(f"Error loading CSV file {file_path}: {str(e)}")


def download_text(url: str, cache_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Download a biblical text from a URL.
    
    Args:
        url: URL to download the text from
        cache_path: Optional path to save the downloaded text
        
    Returns:
        Dictionary containing the parsed biblical text
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        content = response.text
        
        # Save to cache if requested
        if cache_path:
            cache_path = Path(cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Determine format based on content and URL
        content_type = response.headers.get('Content-Type', '').lower()
        
        if 'json' in content_type:
            return json.loads(content)
        elif 'xml' in content_type or url.endswith(('.xml', '.osis')):
            if '<osis' in content[:1000]:
                from religious_texts.data_acquisition.parsers import parse_osis
                return parse_osis(content)
            elif '<usfm' in content[:1000] or '\\id' in content[:1000]:
                from religious_texts.data_acquisition.parsers import parse_usfm
                return parse_usfm(content)
            else:
                # Generic XML parsing
                root = ET.fromstring(content)
                # (XML parsing logic here - similar to load_xml function)
                # This is a simplified placeholder
                return {"info": "XML content downloaded. Parsing not implemented yet."}
        elif 'html' in content_type or url.endswith(('.html', '.htm')):
            soup = BeautifulSoup(content, 'html.parser')
            # This is a simplistic approach - would need refinement based on specific sites
            text = soup.get_text()
            from religious_texts.data_acquisition.parsers import parse_plaintext
            return parse_plaintext(text)
        else:
            # Try to infer the format
            if content.strip().startswith('{'):
                return json.loads(content)
            elif '<?xml' in content[:100]:
                # Simplified placeholder
                return {"info": "XML content downloaded. Detailed parsing not implemented yet."}
            else:
                from religious_texts.data_acquisition.parsers import parse_plaintext
                return parse_plaintext(content)
                
    except requests.exceptions.RequestException as e:
        raise IOError(f"Error downloading text from {url}: {str(e)}")
