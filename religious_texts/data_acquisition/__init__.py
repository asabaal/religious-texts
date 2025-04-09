"""
Data Acquisition Module

This module provides functionality for loading and parsing biblical texts
from various sources and formats.
"""

from religious_texts.data_acquisition.loaders import (
    load_text, 
    load_xml, 
    load_json, 
    load_csv,
    download_text
)

from religious_texts.data_acquisition.parsers import (
    parse_plaintext,
    parse_usfm,
    parse_osis,
    parse_sword,
    extract_metadata
)

__all__ = [
    'load_text',
    'load_xml',
    'load_json',
    'load_csv',
    'download_text',
    'parse_plaintext',
    'parse_usfm',
    'parse_osis',
    'parse_sword',
    'extract_metadata'
]
