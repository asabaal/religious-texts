#!/usr/bin/env python3
"""
Filter corrupted Bible file to extract only the most reliable verses
"""
import re
import sys

def is_likely_valid_verse(verse_line):
    """Check if a verse line is likely to be correct"""
    if " - " not in verse_line:
        return False
    
    reference, text = verse_line.split(" - ", 1)
    
    # Check reference format
    match = re.match(r'(.+?)\s+(\d+):(\d+)', reference)
    if not match:
        return False
    
    book = match.group(1).strip()
    chapter = int(match.group(2))
    verse = int(match.group(3))
    
    # Basic sanity checks
    if chapter > 150 or verse > 200:  # Unreasonably high numbers
        return False
    
    if len(text.strip()) < 10:  # Too short to be a real verse
        return False
    
    # Check if text starts with capital letter (most verses do)
    if not text.strip()[0].isalpha() or not text.strip()[0].isupper():
        return False
    
    # Skip verses that contain obvious book names (indicates corruption)
    suspicious_words = ['Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy',
                       'Joshua', 'Judges', 'Ruth', 'Samuel', 'Kings', 'Chronicles',
                       'Ezra', 'Nehemiah', 'Esther', 'Job', 'Psalm', 'Proverb',
                       'Ecclesiastes', 'Song', 'Isaiah', 'Jeremiah', 'Lamentation',
                       'Ezekiel', 'Daniel', 'Hosea', 'Joel', 'Amos', 'Obadiah',
                       'Jonah', 'Micah', 'Nahum', 'Habakkuk', 'Zephaniah', 'Haggai',
                       'Zechariah', 'Malachi', 'Matthew', 'Mark', 'Luke', 'John',
                       'Acts', 'Romans', 'Corinthians', 'Galatians', 'Ephesians',
                       'Philippians', 'Colossians', 'Thessalonians', 'Timothy',
                       'Titus', 'Philemon', 'Hebrews', 'James', 'Peter', 'John',
                       'Jude', 'Revelation']
    
    for word in suspicious_words:
        if word.lower() in text.lower() and word != book.split()[0]:  # Allow the book's own name
            return False
    
    return True

def filter_bible(input_file, output_file):
    """Filter Bible file to keep only likely valid verses"""
    valid_verses = []
    total_verses = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            total_verses += 1
            line = line.strip()
            if line and is_likely_valid_verse(line):
                valid_verses.append(line)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for verse in valid_verses:
            f.write(verse + '\n')
    
    print(f"Filtered {total_verses:,} verses down to {len(valid_verses):,} likely valid verses")
    print(f"Saved to {output_file}")
    return len(valid_verses)

if __name__ == "__main__":
    input_file = "kjv_bible.txt"
    output_file = "kjv_bible_filtered.txt"
    
    count = filter_bible(input_file, output_file)
    print(f"Reduction: {((1 - count/90000) * 100):.1f}% of verses removed as likely corrupted")