#!/usr/bin/env python3
"""
Simple console-based verse notifier that always works
"""
import random
import time
import sys
import os

def load_verses(txt_path):
    """Loads verses from a text file into a list."""
    with open(txt_path, "r", encoding="utf-8") as f:
        verses = [v.strip() for v in f if v.strip()]
    print(f"Loaded {len(verses):,} verses.")
    return verses

def parse_verse_reference(verse_line):
    """Parse verse line to extract book, chapter, verse info."""
    if " - " in verse_line:
        reference, text = verse_line.split(" - ", 1)
        reference = reference.strip('[]')
        
        # Fix for known extraction error
        if reference == "Romans 7:23" and "wages of sin is death" in text:
            reference = "Romans 6:23"
        
        import re
        match = re.match(r'(.+?)\s+(\d+):(\d+)', reference)
        if match:
            book = match.group(1).strip()
            chapter = int(match.group(2))
            verse = int(match.group(3))
            return book, chapter, verse, text
    return None, None, None, verse_line

def show_verse_console(verses):
    """Show verse with console bells and whistles"""
    # Try to find a valid verse (avoid the messed up ones)
    attempts = 0
    book = chapter = verse_num = text = "Unknown"
    
    while attempts < 10:
        verse_line = random.choice(verses)
        book, chapter, verse_num, text = parse_verse_reference(verse_line)
        
        # Skip obviously invalid verses
        if book and chapter and verse_num:
            # Basic validation - skip if chapter > 50 or verse > 200
            if chapter <= 50 and verse_num <= 200:
                break
        attempts += 1
    
    # Clear screen and show verse with attention-grabbing formatting
    os.system('clear')
    print("\n" + "="*60)
    print("🔔 HOURLY BIBLE VERSE 🔔".center(60))
    print("="*60)
    print(f"\n📖 {book} {chapter}:{verse_num}")
    print("-" * 40)
    print(f"\n{text[:200]}{'...' if len(text) > 200 else ''}")
    print("\n" + "="*60)
    
    # Make noise (console bell)
    print('\a' * 3)  # Three bell characters
    
    # Also try to flash terminal (if supported)
    try:
        print('\033[5m')  # Slow blink
        time.sleep(0.5)
        print('\033[25m')  # Reset blink
    except:
        pass

def main():
    verses = load_verses("kjv_bible.txt")
    show_verse_console(verses)

if __name__ == "__main__":
    main()