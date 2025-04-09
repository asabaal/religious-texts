"""
Text Parsers Module

This module provides functions for parsing biblical texts in various formats.
"""

import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Union, Any, Tuple

# Regular expressions for parsing plain text Bible formats
BOOK_PATTERN = r'^(\s*)([A-Za-z\s]+)\s*$'
CHAPTER_PATTERN = r'^(\s*)Chapter\s+(\d+)(\s*)$'
VERSE_PATTERN = r'^(\s*)(\d+)\s+(.+)$'
ALT_VERSE_PATTERN = r'^(\s*)(\d+):(\d+)\s+(.+)$'

# Common book name variations
BOOK_ALIASES = {
    # Old Testament
    'genesis': 'Genesis',
    'gen': 'Genesis',
    'exodus': 'Exodus',
    'exo': 'Exodus',
    'ex': 'Exodus',
    'leviticus': 'Leviticus',
    'lev': 'Leviticus',
    'numbers': 'Numbers',
    'num': 'Numbers',
    'deuteronomy': 'Deuteronomy',
    'deut': 'Deuteronomy',
    'dt': 'Deuteronomy',
    'joshua': 'Joshua',
    'josh': 'Joshua',
    'judges': 'Judges',
    'judg': 'Judges',
    'ruth': 'Ruth',
    '1 samuel': '1 Samuel',
    '1 sam': '1 Samuel',
    '1sam': '1 Samuel',
    '1st samuel': '1 Samuel',
    'i samuel': '1 Samuel',
    '2 samuel': '2 Samuel',
    '2 sam': '2 Samuel',
    '2sam': '2 Samuel',
    '2nd samuel': '2 Samuel',
    'ii samuel': '2 Samuel',
    '1 kings': '1 Kings',
    '1 kgs': '1 Kings',
    '1kgs': '1 Kings',
    '1st kings': '1 Kings',
    'i kings': '1 Kings',
    '2 kings': '2 Kings',
    '2 kgs': '2 Kings',
    '2kgs': '2 Kings',
    '2nd kings': '2 Kings',
    'ii kings': '2 Kings',
    '1 chronicles': '1 Chronicles',
    '1 chron': '1 Chronicles',
    '1chron': '1 Chronicles',
    '1st chronicles': '1 Chronicles',
    'i chronicles': '1 Chronicles',
    '2 chronicles': '2 Chronicles',
    '2 chron': '2 Chronicles',
    '2chron': '2 Chronicles',
    '2nd chronicles': '2 Chronicles',
    'ii chronicles': '2 Chronicles',
    'ezra': 'Ezra',
    'nehemiah': 'Nehemiah',
    'neh': 'Nehemiah',
    'esther': 'Esther',
    'est': 'Esther',
    'job': 'Job',
    'psalms': 'Psalms',
    'psalm': 'Psalms',
    'ps': 'Psalms',
    'proverbs': 'Proverbs',
    'prov': 'Proverbs',
    'ecclesiastes': 'Ecclesiastes',
    'eccl': 'Ecclesiastes',
    'ecc': 'Ecclesiastes',
    'song of solomon': 'Song of Solomon',
    'song': 'Song of Solomon',
    'isaiah': 'Isaiah',
    'isa': 'Isaiah',
    'jeremiah': 'Jeremiah',
    'jer': 'Jeremiah',
    'lamentations': 'Lamentations',
    'lam': 'Lamentations',
    'ezekiel': 'Ezekiel',
    'ezek': 'Ezekiel',
    'daniel': 'Daniel',
    'dan': 'Daniel',
    'hosea': 'Hosea',
    'hos': 'Hosea',
    'joel': 'Joel',
    'amos': 'Amos',
    'obadiah': 'Obadiah',
    'obad': 'Obadiah',
    'jonah': 'Jonah',
    'micah': 'Micah',
    'mic': 'Micah',
    'nahum': 'Nahum',
    'nah': 'Nahum',
    'habakkuk': 'Habakkuk',
    'hab': 'Habakkuk',
    'zephaniah': 'Zephaniah',
    'zeph': 'Zephaniah',
    'haggai': 'Haggai',
    'hag': 'Haggai',
    'zechariah': 'Zechariah',
    'zech': 'Zechariah',
    'malachi': 'Malachi',
    'mal': 'Malachi',
    
    # New Testament
    'matthew': 'Matthew',
    'matt': 'Matthew',
    'mt': 'Matthew',
    'mark': 'Mark',
    'mk': 'Mark',
    'luke': 'Luke',
    'lk': 'Luke',
    'john': 'John',
    'jn': 'John',
    'acts': 'Acts',
    'acts of the apostles': 'Acts',
    'romans': 'Romans',
    'rom': 'Romans',
    '1 corinthians': '1 Corinthians',
    '1 cor': '1 Corinthians',
    '1cor': '1 Corinthians',
    '1st corinthians': '1 Corinthians',
    'i corinthians': '1 Corinthians',
    '2 corinthians': '2 Corinthians',
    '2 cor': '2 Corinthians',
    '2cor': '2 Corinthians',
    '2nd corinthians': '2 Corinthians',
    'ii corinthians': '2 Corinthians',
    'galatians': 'Galatians',
    'gal': 'Galatians',
    'ephesians': 'Ephesians',
    'eph': 'Ephesians',
    'philippians': 'Philippians',
    'phil': 'Philippians',
    'colossians': 'Colossians',
    'col': 'Colossians',
    '1 thessalonians': '1 Thessalonians',
    '1 thess': '1 Thessalonians',
    '1thess': '1 Thessalonians',
    '1st thessalonians': '1 Thessalonians',
    'i thessalonians': '1 Thessalonians',
    '2 thessalonians': '2 Thessalonians',
    '2 thess': '2 Thessalonians',
    '2thess': '2 Thessalonians',
    '2nd thessalonians': '2 Thessalonians',
    'ii thessalonians': '2 Thessalonians',
    '1 timothy': '1 Timothy',
    '1 tim': '1 Timothy',
    '1tim': '1 Timothy',
    '1st timothy': '1 Timothy',
    'i timothy': '1 Timothy',
    '2 timothy': '2 Timothy',
    '2 tim': '2 Timothy',
    '2tim': '2 Timothy',
    '2nd timothy': '2 Timothy',
    'ii timothy': '2 Timothy',
    'titus': 'Titus',
    'philemon': 'Philemon',
    'phlm': 'Philemon',
    'hebrews': 'Hebrews',
    'heb': 'Hebrews',
    'james': 'James',
    'jas': 'James',
    '1 peter': '1 Peter',
    '1 pet': '1 Peter',
    '1pet': '1 Peter',
    '1st peter': '1 Peter',
    'i peter': '1 Peter',
    '2 peter': '2 Peter',
    '2 pet': '2 Peter',
    '2pet': '2 Peter',
    '2nd peter': '2 Peter',
    'ii peter': '2 Peter',
    '1 john': '1 John',
    '1 jn': '1 John',
    '1jn': '1 John',
    '1st john': '1 John',
    'i john': '1 John',
    '2 john': '2 John',
    '2 jn': '2 John',
    '2jn': '2 John',
    '2nd john': '2 John',
    'ii john': '2 John',
    '3 john': '3 John',
    '3 jn': '3 John',
    '3jn': '3 John',
    '3rd john': '3 John',
    'iii john': '3 John',
    'jude': 'Jude',
    'revelation': 'Revelation',
    'rev': 'Revelation',
}


def parse_plaintext(text: str) -> Dict[str, Any]:
    """
    Parse a plain text Bible into a structured dictionary.
    
    Args:
        text: Plain text content of a Bible
        
    Returns:
        Dictionary with structure {book: {chapter: {verse: text}}}
    """
    lines = text.splitlines()
    bible = {}
    
    current_book = None
    current_chapter = None
    
    for line in lines:
        line = line.strip()
        
        if not line:
            continue
            
        # Check for book header
        book_match = re.match(BOOK_PATTERN, line)
        if book_match and len(line) < 40:  # A reasonable length limit for book names
            potential_book = book_match.group(2).strip()
            
            # Normalize book name if possible
            lower_book = potential_book.lower()
            if lower_book in BOOK_ALIASES:
                current_book = BOOK_ALIASES[lower_book]
            else:
                current_book = potential_book
                
            if current_book not in bible:
                bible[current_book] = {}
                
            current_chapter = None
            continue
            
        # Check for chapter header
        chapter_match = re.match(CHAPTER_PATTERN, line)
        if chapter_match:
            if current_book is None:
                continue  # Skip if we haven't identified a book yet
                
            chapter_num = int(chapter_match.group(2))
            current_chapter = chapter_num
            
            if current_chapter not in bible[current_book]:
                bible[current_book][current_chapter] = {}
                
            continue
            
        # Check for verse line (format: "1 In the beginning...")
        verse_match = re.match(VERSE_PATTERN, line)
        if verse_match:
            if current_book is None or current_chapter is None:
                continue  # Skip if we haven't identified a book and chapter
                
            verse_num = int(verse_match.group(2))
            verse_text = verse_match.group(3).strip()
            
            bible[current_book][current_chapter][verse_num] = verse_text
            continue
            
        # Check for alternate verse format (1:1 In the beginning...)
        alt_verse_match = re.match(ALT_VERSE_PATTERN, line)
        if alt_verse_match:
            if current_book is None:
                continue  # Skip if we haven't identified a book
                
            chapter_num = int(alt_verse_match.group(2))
            verse_num = int(alt_verse_match.group(3))
            verse_text = alt_verse_match.group(4).strip()
            
            current_chapter = chapter_num
            
            if current_chapter not in bible[current_book]:
                bible[current_book][current_chapter] = {}
                
            bible[current_book][current_chapter][verse_num] = verse_text
            continue
            
        # If we reach here and have current book/chapter, this might be continuation of a verse
        if current_book and current_chapter and bible[current_book][current_chapter]:
            # Append to the last verse
            last_verse = max(bible[current_book][current_chapter].keys())
            bible[current_book][current_chapter][last_verse] += " " + line
    
    return bible


def parse_usfm(text: str) -> Dict[str, Any]:
    """
    Parse a USFM (Unified Standard Format Markers) formatted Bible text.
    
    Args:
        text: USFM formatted content
        
    Returns:
        Dictionary with structure {book: {chapter: {verse: text}}}
    """
    bible = {}
    
    current_book = None
    current_chapter = None
    current_verse = None
    
    # Regular expressions for USFM markers
    id_pattern = r'\\id\s+([A-Z1-3]{3})'  # Book identifier
    toc_pattern = r'\\toc[1-3]\s+(.+)'     # Table of contents (book name)
    chapter_pattern = r'\\c\s+(\d+)'       # Chapter marker
    verse_pattern = r'\\v\s+(\d+)\s+(.+?)(?=\\v|\Z)'  # Verse marker
    
    # Extract book ID
    id_match = re.search(id_pattern, text)
    book_id = id_match.group(1) if id_match else None
    
    # Extract book name from table of contents
    toc_match = re.search(toc_pattern, text)
    book_name = toc_match.group(1) if toc_match else book_id
    
    if not book_name:
        return bible  # Cannot proceed without book name
        
    current_book = book_name
    bible[current_book] = {}
    
    # Extract chapters and verses
    chapter_matches = re.finditer(chapter_pattern, text)
    
    for chapter_match in chapter_matches:
        chapter_num = int(chapter_match.group(1))
        current_chapter = chapter_num
        bible[current_book][current_chapter] = {}
        
        # Find chapter start position
        chapter_start = chapter_match.end()
        
        # Find chapter end position (next chapter or end of text)
        next_chapter_match = re.search(r'\\c\s+\d+', text[chapter_start:])
        if next_chapter_match:
            chapter_end = chapter_start + next_chapter_match.start()
        else:
            chapter_end = len(text)
            
        chapter_text = text[chapter_start:chapter_end]
        
        # Extract verses
        verse_matches = re.finditer(verse_pattern, chapter_text)
        
        for verse_match in verse_matches:
            verse_num = int(verse_match.group(1))
            verse_text = verse_match.group(2).strip()
            
            # Clean up USFM markers in verse text
            verse_text = re.sub(r'\\[a-z]+\s+', '', verse_text)  # Remove markers like \q, \m, etc.
            verse_text = re.sub(r'\\[a-z]+\*', '', verse_text)   # Remove end markers like \q*
            
            bible[current_book][current_chapter][verse_num] = verse_text
    
    return bible


def parse_osis(text: str) -> Dict[str, Any]:
    """
    Parse an OSIS (Open Scripture Information Standard) XML Bible text.
    
    Args:
        text: OSIS XML content
        
    Returns:
        Dictionary with structure {book: {chapter: {verse: text}}}
    """
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        # Try to clean up the XML if it's malformed
        text = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;)', '&amp;', text)
        try:
            root = ET.fromstring(text)
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse OSIS XML: {e}")
    
    bible = {}
    
    # Handle various OSIS formats
    # Extract the namespace if present
    ns_match = re.search(r'xmlns="([^"]+)"', text)
    ns = f"{{{ns_match.group(1)}}}" if ns_match else ""
    
    # Find all div elements that are books
    for book_div in root.findall(f".//div[@type='book']") or root.findall(f".//{ns}div[@type='book']"):
        # Get book name/ID
        book_id = book_div.get('osisID') or book_div.get(f'{ns}osisID')
        
        if not book_id:
            continue
            
        # Convert OSIS book ID to full name if possible
        book_name = book_id.split('.')[-1]
        lower_book = book_name.lower()
        
        if lower_book in BOOK_ALIASES:
            book_name = BOOK_ALIASES[lower_book]
            
        bible[book_name] = {}
        
        # Find all chapter divs
        for chapter_div in book_div.findall(f".//chapter") or book_div.findall(f".//{ns}chapter") or \
                           book_div.findall(f".//div[@type='chapter']") or book_div.findall(f".//{ns}div[@type='chapter']"):
            
            chapter_id = chapter_div.get('osisID') or chapter_div.get(f'{ns}osisID') or chapter_div.get('n')
            
            if not chapter_id:
                continue
                
            # Extract chapter number
            try:
                chapter_num = int(chapter_id.split('.')[-1])
            except (ValueError, AttributeError):
                continue
                
            bible[book_name][chapter_num] = {}
            
            # Find all verse elements
            for verse in chapter_div.findall(f".//verse") or chapter_div.findall(f".//{ns}verse") or \
                         chapter_div.findall(f".//div[@type='verse']") or chapter_div.findall(f".//{ns}div[@type='verse']"):
                
                verse_id = verse.get('osisID') or verse.get(f'{ns}osisID') or verse.get('n')
                
                if not verse_id:
                    continue
                    
                # Extract verse number
                try:
                    verse_num = int(verse_id.split('.')[-1])
                except (ValueError, AttributeError):
                    continue
                    
                # Extract text - combining all text inside the verse element
                verse_text = ''.join(verse.itertext()).strip()
                
                # Clean up text by removing line breaks and extra whitespace
                verse_text = re.sub(r'\s+', ' ', verse_text)
                
                bible[book_name][chapter_num][verse_num] = verse_text
    
    return bible


def parse_sword(text: str) -> Dict[str, Any]:
    """
    Parse a SWORD module XML format.
    
    Args:
        text: SWORD XML content
        
    Returns:
        Dictionary with structure {book: {chapter: {verse: text}}}
    """
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        # Try to clean up the XML if it's malformed
        text = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;)', '&amp;', text)
        try:
            root = ET.fromstring(text)
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse SWORD XML: {e}")
    
    bible = {}
    
    # SWORD XML structure varies, this is a basic implementation
    # that may need adjustment for specific SWORD modules
    
    # Find all testament elements
    for testament in root.findall(".//TESTAMENT"):
        # Find all book elements
        for book in testament.findall(".//BOOK"):
            book_name = book.get('bname')
            
            if not book_name:
                continue
                
            bible[book_name] = {}
            
            # Find all chapter elements
            for chapter in book.findall(".//CHAPTER"):
                try:
                    chapter_num = int(chapter.get('cnumber'))
                except (ValueError, TypeError):
                    continue
                    
                bible[book_name][chapter_num] = {}
                
                # Find all verse elements
                for verse in chapter.findall(".//VERSE"):
                    try:
                        verse_num = int(verse.get('vnumber'))
                    except (ValueError, TypeError):
                        continue
                        
                    verse_text = verse.text.strip() if verse.text else ''
                    
                    bible[book_name][chapter_num][verse_num] = verse_text
    
    return bible


def extract_metadata(bible: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metadata from a parsed Bible dictionary.
    
    Args:
        bible: Dictionary with parsed Bible content
        
    Returns:
        Dictionary containing metadata such as:
        - book_count: Number of books
        - chapter_count: Total number of chapters
        - verse_count: Total number of verses
        - books: List of book names
        - longest_book: Book with most chapters
        - shortest_book: Book with fewest chapters
        - average_chapters_per_book: Average number of chapters per book
    """
    metadata = {
        'book_count': len(bible),
        'books': sorted(list(bible.keys())),
        'chapter_count': 0,
        'verse_count': 0,
        'chapters_per_book': {},
        'verses_per_book': {},
        'verses_per_chapter': {},
    }
    
    for book, chapters in bible.items():
        chapter_count = len(chapters)
        metadata['chapter_count'] += chapter_count
        metadata['chapters_per_book'][book] = chapter_count
        
        verse_count = 0
        for chapter, verses in chapters.items():
            verse_count += len(verses)
            metadata['verses_per_chapter'][f"{book} {chapter}"] = len(verses)
            
        metadata['verse_count'] += verse_count
        metadata['verses_per_book'][book] = verse_count
    
    # Identify books with most/least chapters
    if metadata['chapters_per_book']:
        metadata['longest_book'] = max(metadata['chapters_per_book'].items(),
                                       key=lambda x: x[1])[0]
        metadata['shortest_book'] = min(metadata['chapters_per_book'].items(),
                                        key=lambda x: x[1])[0]
    
    # Calculate averages
    if metadata['book_count'] > 0:
        metadata['average_chapters_per_book'] = metadata['chapter_count'] / metadata['book_count']
        
    if metadata['chapter_count'] > 0:
        metadata['average_verses_per_chapter'] = metadata['verse_count'] / metadata['chapter_count']
    
    return metadata
