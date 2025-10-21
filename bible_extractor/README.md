# Bible Extraction System - Usage Guide

## Quick Start

### 1. Test the System (Recommended)
```bash
cd bible_extractor
python extract_bible.py ../bible_king_james_version.pdf --test
```

### 2. Extract Full Bible
```bash
cd bible_extractor
python extract_bible.py ../bible_king_james_version.pdf -o kjv_output
```

### 3. Use with Different PDFs
```bash
# NRSV
python extract_bible.py ../new-revised-standard-version.pdf -o nrsv_output

# ESV  
python extract_bible.py ../english-standard-version-bible.pdf -o esv_output
```

## Command Line Options

- `pdf_path`: Path to Bible PDF file (required)
- `-o, --output`: Output directory (default: "output")
- `-b, --batch-size`: Number of pages to process at once (default: 5)
- `--test`: Test mode - process first 3 pages only

## Output Files

The system generates several output files:

### 1. `bible_verses.json`
Complete structured data with metadata:
```json
{
  "metadata": {
    "source_pdf": "bible_king_james_version.pdf",
    "total_verses": 31102,
    "total_pages": 2652,
    "detected_books": ["Genesis", "Exodus", ...],
    "canon_type": "protestant",
    "pdf_format": "verse_start_lines"
  },
  "verses": [
    {
      "book": "Genesis",
      "chapter": 1,
      "verse": 1,
      "text": "In the beginning God created...",
      "page": 6,
      "confidence": 0.95,
      "flags": ["starts_chapter"],
      "context": {...}
    }
  ]
}
```

### 2. `bible_verses.txt`
Plain text format compatible with verse_notifier:
```
Genesis 1:1 - In the beginning God created the heaven and the earth.
Genesis 1:2 - And the earth was without form...
```

### 3. `book_discovery.json`
Book detection results:
```json
{
  "detected_books": ["Genesis", "Exodus", ...],
  "canon_type": "protestant",
  "confidence": 0.95,
  "pdf_format": "verse_start_lines"
}
```

### 4. `intermediate_batch_*.json`
Recovery files saved every 5 batches

## Performance Tips

### Batch Size
- **Small batches (3-5)**: More accurate, slower
- **Large batches (10-20)**: Faster, may use more memory
- **Recommended**: 5 pages for most systems

### Resource Requirements
- **RAM**: 4GB+ recommended
- **Storage**: 100MB+ for output files
- **Time**: ~30-60 minutes for full Bible (depends on hardware)

## Using Extracted Verses

### With Verse Notifier
```bash
# Update verse notifier to use new extraction
python ../verse_notifier.py --txt kjv_output/bible_verses.txt --test
```

### For Analysis
```python
import json

# Load verses
with open('kjv_output/bible_verses.json') as f:
    data = json.load(f)

verses = data['verses']
print(f"Total verses: {len(verses)}")

# Find specific verses
for verse in verses:
    if verse['book'] == 'John' and verse['chapter'] == 3 and verse['verse'] == 16:
        print(verse['text'])
```

## Troubleshooting

### Ollama Not Running
```bash
# Start Ollama
ollama serve

# Check available models
ollama list
```

### Memory Issues
- Reduce batch size: `-b 3`
- Close other applications
- Use test mode first

### PDF Not Found
- Check file path
- Ensure PDF is in parent directory
- Use absolute path if needed

## Expected Results

### KJV Bible
- **Verses**: ~31,102
- **Books**: 66 (Protestant canon)
- **Pages**: 2,652
- **Format**: verse_start_lines

### NRSV Bible  
- **Verses**: ~31,102
- **Books**: 73 (Catholic canon)
- **Pages**: 3,062
- **Format**: inline_verses

### ESV Bible
- **Verses**: ~31,102
- **Books**: 66 (Protestant canon)
- **Pages**: 5,973
- **Format**: compact_inline