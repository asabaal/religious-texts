#!/usr/bin/env python3
"""
Main Bible Extraction CLI
Complete command-line interface for extracting entire Bibles
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pdf_processor import PDFProcessor
from ai_extractor import AIExtractor
from typing import List, Dict

class BibleExtractor:
    """Main orchestrator for Bible extraction"""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor("config/pdfs.yaml")
        self.ai_extractor = AIExtractor("config/models.yaml")
        
    def extract_full_bible(self, pdf_path: str, output_dir: str = "output", batch_size: int = 5):
        """
        Extract entire Bible PDF to structured verses
        
        Args:
            pdf_path: Path to Bible PDF
            output_dir: Directory to save results
            batch_size: Number of pages to process at once
        """
        print(f"🔥 Starting full Bible extraction from {pdf_path}")
        print(f"📁 Output directory: {output_dir}")
        print(f"📦 Batch size: {batch_size} pages")
        print("=" * 60)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Start timing
        start_time = time.time()
        
        try:
            # Phase 1: Book Discovery
            print("\n📚 PHASE 1: Discovering Books...")
            book_result = self.pdf_processor.discover_books(pdf_path)
            
            print(f"✅ Detected {len(book_result.detected_books)} books")
            print(f"📖 Books: {', '.join(book_result.detected_books[:10])}")
            if len(book_result.detected_books) > 10:
                print(f"   ... and {len(book_result.detected_books) - 10} more")
            print(f"🎯 Canon type: {book_result.canon_type}")
            print(f"📋 PDF format: {book_result.pdf_format}")
            
            # Save book discovery results
            discovery_file = output_path / "book_discovery.json"
            with open(discovery_file, 'w') as f:
                json.dump({
                    'detected_books': book_result.detected_books,
                    'canon_type': book_result.canon_type,
                    'confidence': book_result.confidence,
                    'pdf_format': book_result.pdf_format,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            print(f"💾 Saved book discovery to {discovery_file}")
            
            # Phase 2: Extract all pages
            print(f"\n📄 PHASE 2: Extracting Pages...")
            pages = self.pdf_processor.extract_pages(pdf_path)
            print(f"✅ Extracted {len(pages)} total pages")
            
            # Phase 3: Process pages in batches
            print(f"\n🤖 PHASE 3: AI Verse Extraction...")
            print(f"Processing {len(pages)} pages in batches of {batch_size}...")
            
            all_verses = []
            total_batches = (len(pages) + batch_size - 1) // batch_size
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(pages))
                batch_pages = pages[start_idx:end_idx]
                
                print(f"\n📦 Batch {batch_num + 1}/{total_batches} (pages {start_idx + 1}-{end_idx})")
                
                try:
                    # Extract verses from batch
                    from ai_extractor import ExtractionContext
                    context = ExtractionContext(
                        current_book=None,
                        current_chapter=None,
                        last_verse=None,
                        detected_books=book_result.detected_books,
                        pdf_format=book_result.pdf_format,
                        canon_type=book_result.canon_type
                    )
                    
                    verses, updated_context = self.ai_extractor.extract_verses_batch(
                        batch_pages, start_idx, book_result
                    )
                    
                    all_verses.extend(verses)
                    print(f"✅ Extracted {len(verses)} verses (total: {len(all_verses)})")
                    
                    # Save intermediate results
                    if (batch_num + 1) % 5 == 0:  # Every 5 batches
                        self._save_intermediate_results(all_verses, output_path, batch_num + 1)
                    
                except Exception as e:
                    print(f"❌ Error in batch {batch_num + 1}: {e}")
                    continue
            
            # Phase 4: Save final results
            print(f"\n💾 PHASE 4: Saving Results...")
            
            # Save as JSON
            json_file = output_path / "bible_verses.json"
            verses_data = []
            for verse in all_verses:
                verses_data.append({
                    'book': verse.book,
                    'chapter': verse.chapter,
                    'verse': verse.verse,
                    'text': verse.text,
                    'page': verse.page,
                    'confidence': verse.confidence,
                    'flags': verse.flags,
                    'context': verse.context
                })
            
            with open(json_file, 'w') as f:
                json.dump({
                    'metadata': {
                        'source_pdf': pdf_path,
                        'total_verses': len(all_verses),
                        'total_pages': len(pages),
                        'detected_books': book_result.detected_books,
                        'canon_type': book_result.canon_type,
                        'pdf_format': book_result.pdf_format,
                        'extraction_time': datetime.now().isoformat(),
                        'processing_time_seconds': time.time() - start_time
                    },
                    'verses': verses_data
                }, f, indent=2)
            
            print(f"✅ Saved {len(all_verses)} verses to {json_file}")
            
            # Save as plain text (for verse notifier)
            text_file = output_path / "bible_verses.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                for verse in all_verses:
                    f.write(f"{verse.book} {verse.chapter}:{verse.verse} - {verse.text}\n")
            
            print(f"✅ Saved plain text to {text_file}")
            
            # Phase 5: Summary
            elapsed_time = time.time() - start_time
            print(f"\n🎉 EXTRACTION COMPLETE!")
            print("=" * 60)
            print(f"📊 Results Summary:")
            print(f"   📖 Total verses: {len(all_verses):,}")
            print(f"   📄 Total pages: {len(pages)}")
            print(f"   📚 Books detected: {len(book_result.detected_books)}")
            print(f"   ⏱️  Processing time: {elapsed_time:.1f} seconds")
            print(f"   🚀 Verses/second: {len(all_verses)/elapsed_time:.1f}")
            print(f"   📁 Output files:")
            print(f"      - {json_file}")
            print(f"      - {text_file}")
            print(f"      - {discovery_file}")
            
            return all_verses
            
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            raise
    
    def _save_intermediate_results(self, verses: List, output_path: Path, batch_num: int):
        """Save intermediate results for recovery"""
        intermediate_file = output_path / f"intermediate_batch_{batch_num}.json"
        
        verses_data = []
        for verse in verses:
            verses_data.append({
                'book': verse.book,
                'chapter': verse.chapter,
                'verse': verse.verse,
                'text': verse.text,
                'page': verse.page,
                'confidence': verse.confidence
            })
        
        with open(intermediate_file, 'w') as f:
            json.dump({
                'batch_number': batch_num,
                'total_verses': len(verses),
                'timestamp': datetime.now().isoformat(),
                'verses': verses_data
            }, f, indent=2)
        
        print(f"💾 Saved intermediate results to {intermediate_file}")

def main():
    parser = argparse.ArgumentParser(description="AI-Powered Bible Extraction System")
    parser.add_argument("pdf_path", help="Path to Bible PDF file")
    parser.add_argument("-o", "--output", default="output", help="Output directory (default: output)")
    parser.add_argument("-b", "--batch-size", type=int, default=5, help="Batch size for processing (default: 5)")
    parser.add_argument("--test", action="store_true", help="Run in test mode (first 3 pages only)")
    
    args = parser.parse_args()
    
    # Validate PDF path
    if not Path(args.pdf_path).exists():
        print(f"❌ PDF file not found: {args.pdf_path}")
        sys.exit(1)
    
    # Create extractor
    extractor = BibleExtractor()
    
    try:
        if args.test:
            print("🧪 TEST MODE: Processing first 3 pages only")
            # Override for testing
            pages = extractor.pdf_processor.extract_pages(args.pdf_path, start_page=0, end_page=3)
            print(f"📄 Extracted {len(pages)} pages for testing")
            
            # Run book discovery
            book_result = extractor.pdf_processor.discover_books(args.pdf_path)
            print(f"📚 Books: {book_result.detected_books}")
            
        else:
            # Full extraction
            verses = extractor.extract_full_bible(
                pdf_path=args.pdf_path,
                output_dir=args.output,
                batch_size=args.batch_size
            )
            
            print(f"\n🎯 Ready to use! Try the verses with:")
            print(f"   python ../verse_notifier.py --txt {args.output}/bible_verses.txt --test")
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Extraction interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()