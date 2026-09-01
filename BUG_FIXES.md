# Religious Texts Project - Bug Fixes & Improvements

## Verse Notifier Fixes

### 2025-10-21 - GUI Notification Environment Variables
**Issue**: Verse notifier running but no GUI popups appearing
**Root Cause**: Systemd service missing display environment variables for Wayland/GNOME
**Fix**: Added proper environment variables to verse-notifier.service:
- `DISPLAY=:0`
- `WAYLAND_DISPLAY=wayland-0` 
- `XDG_RUNTIME_DIR=/run/user/1000`
- `DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus`
- `XDG_SESSION_TYPE=wayland`
- `GNOME_SHELL_SESSION_MODE=ubuntu`
**Status**: ✅ FIXED - GUI notifications now working

### 2025-10-21 - Verse Text Truncation
**Issue**: Verse notifications showing truncated text (cut off mid-sentence)
**Root Cause**: Corrupted Bible extraction file (`kjv_bible.txt`) with scrambled verse references
- Example: "Ruth 4:6" had 200+ entries containing verses from many different books
- Famous verse Romans 6:23 was mislabeled as Romans 7:23
**Fix**: 
1. **Code Fix**: Added specific correction in `verse_notifier.py:167-168`:
   ```python
   # Fix for known extraction error: Romans 7:23 should be Romans 6:23
   if reference == "Romans 7:23" and "wages of sin is death" in text:
       reference = "Romans 6:23"
   ```
2. **Data Fix**: Created `kjv_clean_verses.txt` with 20 verified clean verses
3. **Default Change**: Updated verse notifier to use clean file by default
**Status**: ✅ FIXED - Full verse text now displaying correctly

### 2025-10-21 - Romans 6:23 Reference Correction
**Issue**: Famous verse "For the wages of sin is death" incorrectly labeled as Romans 7:23
**Root Cause**: PDF extraction error in original Bible processing
**Fix**: Implemented targeted reference correction in verse parsing logic
**Impact**: Ensures correct attribution of this well-known verse
**Status**: ✅ FIXED - Now correctly shows as Romans 6:23

## Bible Extraction System

### 2025-10-21 - AI-Powered Extraction System
**Improvement**: Complete AI extraction system using Ollama + Qwen3:30b
**Components**:
- `bible_extractor/src/pdf_processor.py` - Dynamic book detection
- `bible_extractor/src/ai_extractor.py` - Qwen3-powered intelligent extraction
- `bible_extractor/config/` - Format configurations and validation rules
- `bible_extractor/extract_bible.py` - CLI interface
**Status**: 🔄 IN PROGRESS - Currently running (3+ hours, ~8/884 batches complete)
**Expected Result**: ~31,100 clean, properly referenced verses

## Future Improvements

- [ ] Complete AI Bible extraction and replace corrupted `kjv_bible.txt`
- [ ] Add verse validation layer to prevent impossible references
- [ ] Implement comprehensive testing for verse reference accuracy
- [ ] Add audio system fallback for headless environments