Developer Guide
===============

This guide is for developers who want to contribute to or extend the Biblical Content Agent system.

Development Setup
-----------------

1. **Clone the repository**:
   .. code-block:: bash
   
      git clone <repository-url>
      cd religious-texts

2. **Create virtual environment**:
   .. code-block:: bash
   
      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install development dependencies**:
   .. code-block:: bash
   
      pip install -r requirements.txt
      pip install -r requirements-dev.txt  # If available

4. **Install in development mode**:
   .. code-block:: bash
   
      pip install -e .

5. **Set up pre-commit hooks** (if configured):
   .. code-block:: bash
   
      pre-commit install

Project Structure
-----------------

.. code-block:: text

   religious-texts/
   ├── content_gen_cli.py          # Main CLI interface
   ├── news_gathering.py           # News sentiment analysis
   ├── weekly_content.py           # Weekly automation
   ├── content-gen                 # Executable CLI script
   ├── religious_texts/            # Core analysis modules
   │   ├── data_acquisition/       # Data loading and parsing
   │   ├── text_analysis/          # Text processing and analysis
   │   ├── theological_analysis/   # Biblical analysis tools
   │   ├── debate_response/        # Debate and apologetics
   │   ├── specialized/            # Specialized analysis
   │   └── visualization/          # Data visualization
   ├── tests/                      # Test suite
   ├── docs/                       # Documentation
   └── generated_content/          # Output directory

Core Components
---------------

Content Generation CLI
~~~~~~~~~~~~~~~~~~~~~~

The main interface is ``content_gen_cli.py``:

- ``SimpleContentGenerator``: Main class for content generation
- ``gather_news_sentiment()``: News analysis functionality
- ``generate_content()``: Core content generation
- Configuration management and caching

News Gathering
~~~~~~~~~~~~~~

``news_gathering.py`` handles news sentiment analysis:

- ``NewsGatherer``: RSS feed processing and sentiment analysis
- ``get_simulated_news_data()``: Demo data generation
- Christian news source configuration
- Sentiment calculation and trend analysis

Religious Texts Analysis
~~~~~~~~~~~~~~~~~~~~~~~

The ``religious_texts/`` package contains specialized analysis modules:

**Data Acquisition**:
- ``loaders.py``: Load biblical texts and data
- ``parsers.py``: Parse different text formats

**Text Analysis**:
- ``frequency.py``: Word frequency analysis
- ``concordance.py``: Biblical concordance tools
- ``sentiment.py``: Sentiment analysis for religious texts
- ``entities.py``: Entity extraction (people, places, etc.)

**Theological Analysis**:
- ``worship.py``: Worship terminology analysis
- ``authority.py``: Biblical authority analysis
- ``divine_names.py``: Names and titles of God
- ``parallel.py``: Parallel passage analysis

**Debate Response**:
- ``context.py``: Contextual analysis for debates
- ``comparisons.py``: Comparative religious analysis
- ``consensus.py``: Consensus building tools

Code Style and Standards
------------------------

Python Standards
~~~~~~~~~~~~~~~~

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write comprehensive docstrings
- Maximum line length: 88 characters (Black default)

Docstring Format
~~~~~~~~~~~~~~~~

Use Google-style docstrings:

.. code-block:: python

   def calculate_sentiment(self, content: str) -> float:
       """Calculate sentiment score for content.
       
       Analyzes the emotional tone of the given content using
       positive and negative word lists.
       
       Args:
           content (str): Text content to analyze
           
       Returns:
           float: Sentiment score from -1 (negative) to 1 (positive)
           
       Example:
           >>> analyzer = SentimentAnalyzer()
           >>> score = analyzer.calculate_sentiment("God is love")
           >>> print(f"Sentiment: {score:.2f}")
       """
       pass

Type Hints
~~~~~~~~~~

Use type hints for all function signatures:

.. code-block:: python

   from typing import Dict, List, Optional, Union, Tuple

   def generate_content(
       self,
       count: int = 3,
       concepts: Optional[List[str]] = None,
       use_news: bool = True
   ) -> Dict[str, Any]:
       """Generate content with specified parameters."""
       pass

Testing
-------

Test Structure
~~~~~~~~~~~~~~

Tests are organized in the ``tests/`` directory:

.. code-block:: text

   tests/
   ├── test_content_cli.py      # CLI functionality tests
   ├── test_news_gathering.py   # News gathering tests
   ├── test_worship.py          # Worship analysis tests
   └── test_*.py                # Other module tests

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   python -m pytest

   # Run with coverage
   python -m pytest --cov=religious_texts --cov-report=html

   # Run specific test file
   python -m pytest tests/test_content_cli.py

   # Run with verbose output
   python -m pytest -v

Writing Tests
~~~~~~~~~~~~~

Use pytest with descriptive test names:

.. code-block:: python

   class TestSimpleContentGenerator:
       """Test cases for SimpleContentGenerator class"""
       
       def test_generate_content_with_news_integration(self):
           """Test content generation with news sentiment integration."""
           generator = SimpleContentGenerator()
           
           result = generator.generate_content(
               count=1,
               use_news=True
           )
           
           assert 'content' in result
           assert len(result['content']) == 1
           assert 'news_sentiment' in result

Fixtures
~~~~~~~~

Use pytest fixtures for common setup:

.. code-block:: python

   @pytest.fixture
   def mock_news_data(self):
       """Create mock news data for testing."""
       return {
           "sentiment_analysis": {
               "overall_sentiment": "positive",
               "trending_topics": [{"topic": "faith", "sentiment": 0.7}]
           }
       }

Mocking
~~~~~~~

Use unittest.mock for external dependencies:

.. code-block:: python

   from unittest.mock import Mock, patch

   @patch('feedparser.parse')
   def test_rss_gathering_success(self, mock_parse):
       """Test successful RSS gathering."""
       mock_parse.return_value = mock_feed_data
       
       gatherer = NewsGatherer()
       articles = gatherer.gather_from_rss("test", config, 7)
       
       assert len(articles) > 0

Adding New Features
------------------

New Content Sources
~~~~~~~~~~~~~~~~~~~

To add a new news source:

1. Update ``news_gathering.py``:
   
.. code-block:: python

   self.sources["new_source"] = {
       "rss": "https://example.com/feed",
       "weight": 0.8,
       "focus": ["relevant", "topics"]
   }

2. Add tests for the new source
3. Update documentation

New Analysis Modules
~~~~~~~~~~~~~~~~~~~~

To add a new analysis module:

1. Create module in appropriate ``religious_texts/`` subdirectory
2. Follow existing patterns for imports and structure
3. Add comprehensive docstrings
4. Create tests in ``tests/`` directory
5. Update API documentation

Example new module:

.. code-block:: python

   """
   New analysis module for biblical content.
   
   This module provides specialized analysis capabilities
   for biblical texts and themes.
   """

   from typing import Dict, List, Optional
   import re

   class NewAnalyzer:
       """Analyzer for new biblical content analysis."""
       
       def __init__(self):
           """Initialize the analyzer with default settings."""
           self.patterns = {
               "example": r"pattern_here"
           }
       
       def analyze(self, text: str) -> Dict:
           """
           Analyze text using new methodology.
           
           Args:
               text (str): Text to analyze
               
           Returns:
               Dict: Analysis results
           """
           results = {}
           # Analysis logic here
           return results

New CLI Commands
~~~~~~~~~~~~~~~~

To add a new CLI command:

1. Add command to ``content_gen_cli.py`` argument parser
2. Implement command function
3. Add tests
4. Update documentation

.. code-block:: python

   def add_analyze_command(self, subparsers):
       """Add new analyze command to CLI."""
       parser = subparsers.add_parser('analyze', help='Analyze content')
       parser.add_argument('--type', choices=['sentiment', 'themes'])
       parser.add_argument('file', help='File to analyze')
       parser.set_defaults(func=self.handle_analyze)

   def handle_analyze(self, args):
       """Handle analyze command."""
       # Implementation here
       pass

Documentation
-------------

Building Documentation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd docs/
   make html

   # Or using sphinx-build directly
   sphinx-build -b html . _build/html

API Documentation
~~~~~~~~~~~~~~~~~

Update ``docs/api.rst`` when adding new modules:

.. code-block:: rst

   .. automodule:: religious_texts.new_module
      :members:
      :undoc-members:
      :show-inheritance:

Examples
~~~~~~~~

Add new examples to ``docs/examples.rst`` following the existing format.

Performance Considerations
--------------------------

Caching
~~~~~~~

The system uses caching for:

- News sentiment data (24-hour cache)
- Configuration data
- Analysis results

Implement caching for new expensive operations:

.. code-block:: python

   import json
   from pathlib import Path
   from datetime import datetime, timedelta

   class CachedAnalyzer:
       def __init__(self, cache_dir: Path):
           self.cache_dir = cache_dir
           self.cache_dir.mkdir(exist_ok=True)
       
       def get_cached_result(self, key: str, max_age_hours: int = 24):
           """Get cached result if fresh enough."""
           cache_file = self.cache_dir / f"{key}.json"
           
           if not cache_file.exists():
               return None
           
           try:
               with open(cache_file) as f:
                   cached = json.load(f)
               
               cache_time = datetime.fromisoformat(cached['timestamp'])
               if (datetime.now() - cache_time) < timedelta(hours=max_age_hours):
                   return cached['data']
           except Exception:
               pass
           
           return None

Async Operations
~~~~~~~~~~~~~~~~

For I/O-bound operations, consider using async/await:

.. code-block:: python

   import asyncio
   import aiohttp

   async def fetch_multiple_feeds(urls: List[str]) -> List[Dict]:
       """Fetch multiple RSS feeds concurrently."""
       async with aiohttp.ClientSession() as session:
           tasks = [fetch_feed(session, url) for url in urls]
           results = await asyncio.gather(*tasks, return_exceptions=True)
           return [r for r in results if not isinstance(r, Exception)]

Deployment
----------

Production Setup
~~~~~~~~~~~~~~~~

For production deployment:

1. Use environment variables for configuration
2. Set up proper logging
3. Use a process manager (systemd, supervisor)
4. Monitor system health

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   export OLLAMA_URL="http://localhost:11434"
   export CONTENT_GEN_MODEL="llama3.1:8b"
   export CONTENT_GEN_CACHE_DIR="/var/cache/content-gen"
   export CONTENT_GEN_LOG_LEVEL="INFO"

Docker Deployment
~~~~~~~~~~~~~~~~~

Example Dockerfile:

.. code-block:: dockerfile

   FROM python:3.11-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY . .

   RUN chmod +x content-gen

   EXPOSE 8000

   CMD ["python", "weekly_content.py"]

Monitoring
~~~~~~~~~~~

Set up health checks and monitoring:

.. code-block:: python

   import logging
   import psutil

   def health_check():
       """System health check for monitoring."""
       checks = {
           'memory_usage': psutil.virtual_memory().percent,
           'disk_usage': psutil.disk_usage('/').percent,
           'cpu_usage': psutil.cpu_percent()
       }
       
       for check, value in checks.items():
           if value > 80:  # Threshold
               logging.warning(f"High {check}: {value}%")
       
       return checks

Contributing
------------

Pull Request Process
~~~~~~~~~~~~~~~~~~~~

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Update documentation
6. Run the test suite
7. Submit a pull request

Code Review
~~~~~~~~~~~

All changes require code review. Reviewers should check:

- Code follows style guidelines
- Tests are comprehensive
- Documentation is updated
- No breaking changes without justification
- Security considerations addressed

Release Process
~~~~~~~~~~~~~~~

1. Update version number
2. Update CHANGELOG.md
3. Create git tag
4. Build and test release
5. Deploy to production

This developer guide provides the foundation for contributing to the Biblical Content Agent system. For specific questions, refer to the existing codebase and documentation.