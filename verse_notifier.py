import argparse
import random
import time
import os
import subprocess
import sys
import re
from plyer import notification
import pyttsx3

# High-quality TTS imports
GTTS_AVAILABLE = False
try:
    from gtts import gTTS
    import pygame
    GTTS_AVAILABLE = True
except ImportError:
    pass

# Coqui TTS imports (for future compatibility)
COQUI_AVAILABLE = False
try:
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except ImportError:
    pass

# Optional PDF extraction tools
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

def extract_bible_text(pdf_path, txt_path):
    """Extracts all text from a PDF and saves it as bible.txt with verse references."""
    if not pdfplumber:
        raise ImportError("pdfplumber not installed. Run: pip install pdfplumber")
    print(f"Extracting text from {pdf_path} ...")
    
    current_book = None
    current_chapter = None
    
    with pdfplumber.open(pdf_path) as pdf, open(txt_path, "w", encoding="utf-8") as out:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                lines = text.splitlines()
                for line in lines:
                    line = line.strip()
                    
                    # Skip empty lines, headings, page numbers, and metadata
                    if (len(line) < 10 or 
                        line.isupper() or 
                        line.isdigit() or 
                        line.startswith('Chapter') or
                        line.startswith('Psalm') or
                        'Copyright' in line or
                        'Bible' in line or
                        'Page' in line or
                        'Introduction' in line or
                        'Foreword' in line or
                        'Abbreviations' in line or
                        line.count(' ') < 2):
                        continue
                    
                    # Detect book names (look for common Bible book patterns)
                    if any(book in line for book in ['Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy',
                                                    'Joshua', 'Judges', 'Ruth', 'Samuel', 'Kings', 'Chronicles',
                                                    'Ezra', 'Nehemiah', 'Esther', 'Job', 'Psalms', 'Proverbs',
                                                    'Ecclesiastes', 'Song', 'Isaiah', 'Jeremiah', 'Lamentations',
                                                    'Ezekiel', 'Daniel', 'Hosea', 'Joel', 'Amos', 'Obadiah',
                                                    'Jonah', 'Micah', 'Nahum', 'Habakkuk', 'Zephaniah', 'Haggai',
                                                    'Zechariah', 'Malachi', 'Matthew', 'Mark', 'Luke', 'John',
                                                    'Acts', 'Romans', 'Corinthians', 'Galatians', 'Ephesians',
                                                    'Philippians', 'Colossians', 'Thessalonians', 'Timothy',
                                                    'Titus', 'Philemon', 'Hebrews', 'James', 'Peter',
                                                    'John', 'Jude', 'Revelation']):
                        # Extract book name
                        words = line.split()
                        for word in words:
                            if any(book in word for book in ['Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy',
                                                            'Joshua', 'Judges', 'Ruth', 'Samuel', 'Kings', 'Chronicles',
                                                            'Ezra', 'Nehemiah', 'Esther', 'Job', 'Psalms', 'Proverbs',
                                                            'Ecclesiastes', 'Song', 'Isaiah', 'Jeremiah', 'Lamentations',
                                                            'Ezekiel', 'Daniel', 'Hosea', 'Joel', 'Amos', 'Obadiah',
                                                            'Jonah', 'Micah', 'Nahum', 'Habakkuk', 'Zephaniah', 'Haggai',
                                                            'Zechariah', 'Malachi', 'Matthew', 'Mark', 'Luke', 'John',
                                                            'Acts', 'Romans', 'Corinthians', 'Galatians', 'Ephesians',
                                                            'Philippians', 'Colossians', 'Thessalonians', 'Timothy',
                                                            'Titus', 'Philemon', 'Hebrews', 'James', 'Peter',
                                                            'John', 'Jude', 'Revelation']):
                                current_book = word.rstrip(' .')
                                current_chapter = 1
                                break
                        continue
                    
                    # Detect chapter numbers (usually standalone numbers)
                    if line.isdigit() and current_book:
                        current_chapter = int(line)
                        continue
                    
                    # Detect verse numbers (usually at start of line)
                    verse_match = None
                    if line and line[0].isdigit():
                        import re
                        verse_match = re.match(r'^(\d+)\s+(.+)', line)
                    
                    if verse_match and current_book and current_chapter:
                        verse_num = verse_match.group(1)
                        verse_text = verse_match.group(2)
                        reference = f"{current_book} {current_chapter}:{verse_num}"
                        out.write(f"{reference} - {verse_text}\n")
                    elif current_book and current_chapter and len(line) > 20:
                        # Continuation verse without number (rare)
                        out.write(f"{current_book} {current_chapter}:? - {line}\n")
                        
            if i % 100 == 0:
                print(f"Processed {i} pages...")
    print(f"✅ Bible text saved to {txt_path}")

def initialize_high_quality_tts():
    """Initialize high-quality TTS (Google TTS or Coqui TTS)."""
    # Check if we're in a headless environment (no display)
    headless = os.environ.get('DISPLAY') is None and os.name == 'posix'
    
    # Try Google TTS first (more compatible)
    if GTTS_AVAILABLE:
        try:
            # Initialize pygame mixer for audio playback only if not headless
            if not headless:
                pygame.mixer.init()
                print("🎵 Using Google TTS for high-quality speech synthesis")
                return "gtts"
            else:
                print("🎵 Using Google TTS for high-quality speech synthesis (headless mode)")
                return "gtts_headless"
        except Exception as e:
            print(f"⚠️ Failed to initialize Google TTS: {e}")
    
    # Try Coqui TTS as fallback
    if COQUI_AVAILABLE:
        try:
            from TTS.api import TTS
            tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
            print("🎵 Using Coqui TTS for high-quality speech synthesis")
            return tts
        except Exception as e:
            print(f"⚠️ Failed to initialize Coqui TTS: {e}")
    
    print("🔊 Using pyttsx3 for speech synthesis")
    return None

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
        # Clean up reference - remove brackets if present
        reference = reference.strip('[]')
        
        # Fix for known extraction error: Romans 7:23 should be Romans 6:23
        if reference == "Romans 7:23" and "wages of sin is death" in text:
            reference = "Romans 6:23"
        
        # Parse book, chapter, verse - handle various formats
        match = re.match(r'(.+?)\s+(\d+):(\d+)', reference)
        if match:
            book = match.group(1).strip()
            chapter = int(match.group(2))
            verse = int(match.group(3))
            return book, chapter, verse, text
        else:
            # Handle cases with ? verse numbers
            match = re.match(r'(.+?)\s+(\d+):\?', reference)
            if match:
                book = match.group(1).strip()
                chapter = int(match.group(2))
                # For ? verses, let's try to find the next actual verse number
                # or use a sequential approach
                return book, chapter, 1, text  # Use 1 as default for ? verses
    return None, None, None, verse_line

def calculate_progress(book, chapter, verse, verses):
    """Calculate percentage progress through the book."""
    if not book or not chapter:
        return None
    
    # Count verses in this book
    book_verses = []
    for v in verses:
        b, c, vr, _ = parse_verse_reference(v)
        if b == book:
            book_verses.append((c, vr))
    
    if not book_verses:
        return None
    
    # Sort verses to find position
    book_verses.sort()
    total_verses = len(book_verses)
    
    # Find current position
    current_pos = 0
    for i, (c, vr) in enumerate(book_verses):
        if c == chapter and vr == verse:
            current_pos = i + 1
            break
    
    if current_pos > 0:
        percentage = (current_pos / total_verses) * 100
        return f"{percentage:.1f}%"
    return None

def prepare_audio_file(tts_message, high_quality_tts):
    """Prepare audio file for playback and return the file path."""
    try:
        if high_quality_tts in ["gtts", "gtts_headless"]:
            # Use Google TTS for high-quality speech
            temp_audio = "temp_verse.mp3"
            tts = gTTS(text=tts_message, lang='en', slow=False)
            tts.save(temp_audio)
            print("✅ Google TTS audio prepared successfully")
            return temp_audio
        elif high_quality_tts and high_quality_tts not in ["gtts", "gtts_headless"]:
            # Use Coqui TTS for high-quality speech
            temp_audio = "temp_verse.wav"
            high_quality_tts.tts_to_file(text=tts_message, file_path=temp_audio)
            print("✅ Coqui TTS audio prepared successfully")
            return temp_audio
        else:
            print("⚠️ No high-quality TTS available, will use fallback")
            return None
    except Exception as e:
        print(f"❌ Google TTS failed ({e}) - falling back to pyttsx3 (lower quality)")
        return None

def play_audio_file(audio_file, high_quality_tts):
    """Play the prepared audio file."""
    if not audio_file:
        return False
    
    try:
        if high_quality_tts == "gtts":
            # Play the audio file using pygame
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.music.unload()
        elif high_quality_tts == "gtts_headless":
            # In headless mode, use system audio player
            if os.name == 'posix':
                if sys.platform == 'darwin':  # macOS
                    subprocess.run(['afplay', audio_file], check=True)
                else:  # Linux
                    subprocess.run(['aplay', audio_file], check=True)
            elif os.name == 'nt':  # Windows
                os.startfile(audio_file)
        elif high_quality_tts and high_quality_tts not in ["gtts", "gtts_headless"]:
            # Use Coqui TTS for high-quality speech
            if os.name == 'nt':  # Windows
                os.startfile(audio_file)
            elif os.name == 'posix':  # macOS and Linux
                if sys.platform == 'darwin':  # macOS
                    subprocess.run(['afplay', audio_file], check=True)
                else:  # Linux
                    subprocess.run(['aplay', audio_file], check=True)
        return True
    except Exception as e:
        print(f"Audio playback failed: {e}")
        return False

def show_verse_zenity(verses, engine, high_quality_tts=None):
    """Show verse using zenity for better interaction with play button."""
    # Prefer verses with proper verse numbers over ones with ?
    proper_verses = [v for v in verses if re.search(r'\d+:\d+', v)]
    if proper_verses:
        verse = random.choice(proper_verses)
    else:
        verse = random.choice(verses)
    book, chapter, verse_num, text = parse_verse_reference(verse)
    
    if book and chapter:
        reference = f"{book} {chapter}:{verse_num}"
        progress = calculate_progress(book, chapter, verse_num, verses)
        if progress:
            title = f"🕊 Verse of the Hour - {reference} ({progress} through {book})"
            display_text = f"{reference} ({progress} through {book})\n\n{text}"
        else:
            title = f"🕊 Verse of the Hour - {reference}"
            display_text = f"{reference}\n\n{text}"
    else:
        title = "🕊 Verse of the Hour"
        display_text = verse
    
    print(f"🕊 Verse:\n{display_text}")
    
    # Also log the verse information for cron jobs
    print(f"📖 Reference: {reference}")
    print(f"📊 Progress: {progress if progress else 'N/A'}")
    print(f"📝 Text: {text}")
    
    # Prepare audio file in advance
    tts_message = f"{reference}. {text}" if book else text
    audio_file = prepare_audio_file(tts_message, high_quality_tts)
    
    # Play audio immediately when notification pops up
    if audio_file:
        # Play audio in background so notification appears immediately
        import threading
        def play_audio_background():
            play_audio_file(audio_file, high_quality_tts)
            # Clean up after playback
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            except:
                pass
        
        audio_thread = threading.Thread(target=play_audio_background, daemon=True)
        audio_thread.start()
    else:
        # Fallback to pyttsx3 for immediate playback
        try:
            engine.say(tts_message)
            engine.runAndWait()
        except Exception as e:
            print(f"Text-to-speech failed: {e}")
    
    # Try zenity first with play button (better for Linux)
    try:
        # Create a custom zenity dialog with play button
        zenity_text = f"{display_text}\n\n🔊 Audio playing automatically..."
        cmd = [
            'zenity', '--info', 
            '--title', title,
            '--text', zenity_text,
            '--no-wrap',
            '--width=500',
            '--height=250'
        ]
        subprocess.run(cmd, check=True)
        
        # Offer to replay audio after closing the main dialog
        if audio_file and os.path.exists(audio_file):
            try:
                replay_cmd = [
                    'zenity', '--question', 
                    '--title', 'Replay Audio?',
                    '--text', 'Would you like to hear the verse again?',
                    '--ok-label=Play Again',
                    '--cancel-label=No'
                ]
                result = subprocess.run(replay_cmd, capture_output=True)
                if result.returncode == 0:  # User clicked "Play Again"
                    play_audio_file(audio_file, high_quality_tts)
            except:
                pass  # Silently fail if user cancels
            
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to plyer
        try:
            notification.notify(
                title=title,
                message=display_text,
                timeout=10  # Show for 10 seconds
            )
        except Exception as e:
            print(f"Notification failed: {e}")
    
    # Final cleanup
    try:
        if audio_file and os.path.exists(audio_file):
            os.remove(audio_file)
    except:
        pass

def show_verse(verses, engine, high_quality_tts=None):
    """Displays a verse and reads it aloud."""
    # Check if zenity is available and we have a display
    try:
        subprocess.run(['which', 'zenity'], check=True, capture_output=True)
        if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
            show_verse_zenity(verses, engine, high_quality_tts)
            return
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
        # Fallback to original method
        # Prefer verses with proper verse numbers over ones with ?
        proper_verses = [v for v in verses if re.search(r'\d+:\d+', v)]
        if proper_verses:
            verse = random.choice(proper_verses)
        else:
            verse = random.choice(verses)
        book, chapter, verse_num, text = parse_verse_reference(verse)
        reference = f"{book} {chapter}:{verse_num}" if book and chapter else "Unknown Reference"
        
        if book and chapter:
            progress = calculate_progress(book, chapter, verse_num, verses)
            if progress:
                title = f"🕊 Verse of the Hour - {reference} ({progress} through {book})"
                display_message = f"{reference} ({progress} through {book})\n{text}"
            else:
                title = f"🕊 Verse of the Hour - {reference}"
                display_message = f"{reference}\n{text}"
            tts_message = f"{reference}. {text}"
        else:
            title = "🕊 Verse of the Hour"
            display_message = verse
            tts_message = verse
        
        print(f"🕊 Verse:\n{display_message}")
        
        # Also log the verse information for cron jobs
        print(f"📖 Reference: {reference}")
        if book and chapter:
            progress = calculate_progress(book, chapter, verse_num, verses)
            print(f"📊 Progress: {progress if progress else 'N/A'}")
        print(f"📝 Text: {text}")
        
        # Prepare and play audio immediately
        audio_file = prepare_audio_file(tts_message, high_quality_tts)
        
        # Play audio immediately when notification pops up
        if audio_file:
            # Play audio in background so notification appears immediately
            import threading
            def play_audio_background():
                play_audio_file(audio_file, high_quality_tts)
                # Clean up after playback
                try:
                    if os.path.exists(audio_file):
                        os.remove(audio_file)
                except:
                    pass
            
            audio_thread = threading.Thread(target=play_audio_background, daemon=True)
            audio_thread.start()
        else:
            # Fallback to pyttsx3 for immediate playback
            try:
                engine.say(tts_message)
                engine.runAndWait()
            except Exception as e:
                print(f"Text-to-speech failed: {e}")
        
        try:
            notification.notify(
                title=title,
                message=display_message,
                timeout=10  # Show for 10 seconds
            )
        except Exception as e:
            print(f"Notification failed: {e}")
        
        # Final cleanup
        try:
            if audio_file and os.path.exists(audio_file):
                os.remove(audio_file)
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description="Bible Verse Notifier")
    parser.add_argument("--extract", action="store_true", help="Extract text from PDF before starting")
    parser.add_argument("--pdf", default="new-revised-standard-version.pdf", help="Path to Bible PDF file")
    parser.add_argument("--txt", default="kjv_bible_filtered.txt", help="Output text file name")
    parser.add_argument("--test", action="store_true", help="Test with one verse then exit")
    parser.add_argument("--cron", action="store_true", help="Run once for cron job (no continuous loop)")
    args = parser.parse_args()

    # Extract phase
    if args.extract or not os.path.exists(args.txt):
        extract_bible_text(args.pdf, args.txt)

    # Load verses and show one verse
    try:
        verses = load_verses(args.txt)
        engine = pyttsx3.init()
        
        # Initialize high-quality TTS if available
        high_quality_tts = initialize_high_quality_tts()
        
        if args.test:
            print("🧪 Test mode - showing one verse:")
        
        show_verse(verses, engine, high_quality_tts)
        
        if not args.cron and not args.test:
            print(f"\n✨ To run continuously, use: while true; do python verse_notifier.py --cron; sleep 3600; done")
            print("💡 Or better yet, set up a cron job!")
            
    except KeyboardInterrupt:
        print("\n👋 Stopping verse notifications.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

