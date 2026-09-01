# AI-Powered Bible Extraction System Architecture

## Overview
An intelligent system using Ollama + Qwen3 30B to extract Bible verses from PDFs with reliable chapter/verse references through AI-powered parsing and comprehensive validation.

## System Architecture
```
PDF Input → Page Processor → AI Engine → Validator → Structured Output
```

## Core Components

### 1. PDF Page Processor (`pdf_processor.py`)
**Purpose**: Extract raw text from PDF pages with metadata
**Input**: PDF file path
**Output**: Array of page objects
```json
{
  "page_num": 51,
  "raw_text": "51 Genesis 24\n24\nA Wife Obtained for Isaac\n1 And Abraham was old...",
  "metadata": {
    "pdf_format": "kjv",
    "has_headers": true,
    "line_count": 25,
    "detected_books": ["Genesis", "Exodus", "Leviticus"]
  }
}
```

**Key Features**:
- Handle different PDF layouts (KJV, NRSV, ESV)
- Preserve page structure and formatting clues
- **Dynamic book detection** - Scan all pages to build book list
- Extract images/footnotes as metadata

### 2. AI Engine (`ai_extractor.py`)
**Purpose**: Intelligent verse extraction and reference parsing
**Input**: Page text + context from previous pages
**Output**: Structured verse data
```json
{
  "book": "Genesis",
  "chapter": 24,
  "verse": 1,
  "text": "And Abraham was old, and well stricken in age: and the Lord had blessed Abraham in all things.",
  "page": 51,
  "confidence": 0.95,
  "flags": ["starts_verse", "complete_sentence"],
  "context": {
    "previous_verse": "23:20",
    "next_verse": "24:2",
    "section_header": "A Wife Obtained for Isaac"
  }
}
```

**Key Features**:
- Context-aware chapter/verse detection
- Handle edge cases (spanning verses, footnotes, section headers)
- Maintain state across pages
- **Dynamic book validation** using detected book list

### 3. Validation Layer (`validator.py`)
**Purpose**: Quality assurance and error correction
**Input**: AI-generated verse data
**Output**: Validated/corrected verse data

**Key Features**:
- Verse number sequence validation
- Book/chapter consistency checks
- Reference format standardization
- **Dynamic book list validation** against detected books

### 4. Output Manager (`output_manager.py`)
**Purpose**: File I/O and formatting
**Formats**: JSON, CSV, plain text
**Features**: Progress tracking, resume capability

### 5. Session Manager (`session_manager.py`)
**Purpose**: Progress tracking and state management
```json
{
  "session_id": "extraction_20251021_143000",
  "pdf_source": "bible_king_james_version.pdf",
  "total_pages": 2652,
  "processed_pages": 100,
  "verses_extracted": 850,
  "detected_books": ["Genesis", "Exodus", "Leviticus", ...],
  "quality_metrics": {
    "avg_confidence": 0.92,
    "validation_errors": 3,
    "reference_accuracy": 0.98
  }
}
```

## Data Flow

### Phase 1: Discovery & Processing
```
PDF → Book Discovery Scan → Page Processing → Context Builder
```

### Phase 2: AI Extraction
```
Context Builder → Prompt Generator → Ollama API → Response Parser
```

### Phase 3: Validation & Output
```
Response Parser → Validator → Output Manager → Quality Metrics
```

## Dynamic Book Detection Strategy

### Discovery Phase (First Pass)
1. **Scan all pages** for book name patterns
2. **Build comprehensive book list** from detected headers
3. **Validate against known canon families**:
   - Protestant canon (66 books)
   - Catholic canon (73 books) 
   - Orthodox canon (78+ books)
   - Custom/Apocryphal content

### Book Detection Algorithm
```python
def detect_books(pdf_pages):
    # 1. Header pattern detection
    header_books = extract_from_headers(pdf_pages)
    
    # 2. Content pattern detection  
    content_books = extract_from_content(pdf_pages)
    
    # 3. Cross-reference with canon databases
    canon_type = identify_canon_type(header_books + content_books)
    
    # 4. Build final book list with validation
    return validate_book_list(books, canon_type)
```

### Canon Type Identification
- **Protestant**: 39 OT + 27 NT books
- **Catholic**: Protestant + Deuterocanonical books
- **Orthodox**: Catholic + Additional books
- **Custom**: Unusual combinations or apocrypha

## AI Prompt Strategy

### Context Building
- **Sliding window**: Previous 2 pages + current page + next 1 page
- **State tracking**: Current book, chapter, last verse number
- **Dynamic book list**: Include detected books for validation
- **Format hints**: PDF-specific patterns

### Enhanced Prompt Structure
```
You are a Bible text extraction expert. Extract verses from this page with perfect accuracy.

DETECTED BOOKS IN THIS BIBLE:
{dynamic_book_list}

CANON TYPE: {canon_type} (Protestant/Catholic/Orthodox/Custom)

CONTEXT:
- Current book: {current_book}
- Current chapter: {current_chapter}
- Last verse: {last_verse}
- PDF format: {pdf_format}

PAGE TEXT:
{page_text}

TASK:
1. Identify book/chapter headers (validate against detected books)
2. Extract all verses with correct references
3. Handle footnotes, section headers, page numbers
4. Maintain verse sequence continuity
5. Flag any books not in the detected list for review

OUTPUT JSON FORMAT:
{verse_object_schema}
```

## Error Handling & Validation

### Dynamic Book Validation
- **Book name verification**: Check against detected book list
- **Canon consistency**: Ensure books match identified canon type
- **Anomaly detection**: Flag unexpected books for manual review

### Correction Strategies
- **Sequence repair**: Fix gaps in verse numbering
- **Reference correction**: Use context to fix mislabeled verses
- **Book name standardization**: Handle variations (e.g., "Song of Solomon" vs "Song of Songs")

### Fallback Mechanisms
- **Retry with different prompts** if confidence < 0.8
- **Manual review queue** for ambiguous cases
- **Regex fallback** for simple verse patterns

## Performance Optimization

### Batch Processing
- **Discovery batch**: Quick scan for book detection
- **Processing batches**: 5-10 pages at a time for context
- **Parallel processing**: Multiple Ollama instances if available
- **Checkpointing**: Save progress every 50 pages

### Memory Management
- **Streaming**: Process large PDFs without loading everything
- **Context windows**: Manage AI context limits efficiently
- **Resource monitoring**: Track Ollama memory usage

## Configuration Files

### `config/pdfs.yaml`
```yaml
pdfs:
  kjv:
    format: "verse_start_lines"
    header_pattern: "^\\d+\\s+([A-Za-z\\s]+?)\\s+(\\d+)$"
    verse_pattern: "^(\\d+)\\s+(.+)"
  nrsv:
    format: "inline_verses"
    header_pattern: "^[A-Z\\s]+$"
    verse_pattern: "\\s(\\d+)\\s"
  esv:
    format: "compact_inline"
    header_pattern: "^[A-Z][a-z]+\\s+\\d+$"
    verse_pattern: "(\\d+)[A-Za-z]"
```

### `config/models.yaml`
```yaml
ollama:
  model: "qwen2.5:30b"
  temperature: 0.1
  max_tokens: 8000
  context_window: 32000

extraction:
  confidence_threshold: 0.8
  batch_size: 5
  max_retries: 3
```

### `config/validation.yaml`
```yaml
quality:
  min_confidence: 0.8
  max_verse_gap: 5
  require_sequence_continuity: true

canon_types:
  protestant:
    old_testament: 39
    new_testament: 27
    total: 66
  catholic:
    old_testament: 46
    new_testament: 27
    total: 73
  orthodox:
    old_testament: 51
    new_testament: 27
    total: 78
```

## Implementation Phases

### Phase 1: Core Infrastructure
- [ ] PDF processor with dynamic book detection
- [ ] Basic AI extraction engine
- [ ] Simple validation layer
- [ ] JSON output format

### Phase 2: Enhanced Intelligence
- [ ] Advanced prompting with dynamic book lists
- [ ] Comprehensive validation and correction
- [ ] Error handling and retry mechanisms
- [ ] Progress tracking and resume

### Phase 3: Production Features
- [ ] Performance optimization
- [ ] Multiple output formats
- [ ] Quality dashboard
- [ ] Full test suite

## File Structure
```
bible_extractor/
├── src/
│   ├── pdf_processor.py
│   ├── ai_extractor.py
│   ├── validator.py
│   ├── output_manager.py
│   └── session_manager.py
├── config/
│   ├── pdfs.yaml
│   ├── models.yaml
│   └── validation.yaml
├── data/
│   ├── canon_databases/
│   └── output/
├── tests/
└── docs/
```

This architecture provides a robust, flexible system that can handle any Bible PDF format while maintaining high accuracy through AI intelligence and dynamic book detection.