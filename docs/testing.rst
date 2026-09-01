Testing Framework
=================

This document describes the testing framework and guidelines for the Biblical Content Agent system.

Test Structure
--------------

The test suite is organized in the ``tests/`` directory with the following structure:

.. code-block:: text

   tests/
   ├── test_content_cli.py      # CLI functionality tests
   ├── test_news_gathering.py   # News sentiment analysis tests
   ├── test_worship.py          # Worship analysis tests
   ├── test_debate_worship_analysis.py  # Debate analysis tests
   ├── test_specialized_worship.py      # Specialized analysis tests
   └── conftest.py              # Pytest configuration and fixtures

Running Tests
-------------

Basic Test Commands
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   python -m pytest

   # Run with verbose output
   python -m pytest -v

   # Run with coverage report
   python -m pytest --cov=religious_texts --cov-report=html

   # Run specific test file
   python -m pytest tests/test_content_cli.py

   # Run specific test class
   python -m pytest tests/test_content_cli.py::TestSimpleContentGenerator

   # Run specific test method
   python -m pytest tests/test_content_cli.py::TestSimpleContentGenerator::test_init

Coverage Reports
~~~~~~~~~~~~~~~~

Generate detailed coverage reports:

.. code-block:: bash

   # HTML coverage report
   python -m pytest --cov=religious_texts --cov-report=html

   # Terminal coverage report
   python -m pytest --cov=religious_texts --cov-report=term-missing

   # Coverage with specific modules
   python -m pytest --cov=news_gathering --cov=content_gen_cli

Test Configuration
------------------

Pytest Configuration
~~~~~~~~~~~~~~~~~~~~

The project uses ``pytest.ini`` for configuration:

.. code-block:: ini

   [tool:pytest]
   testpaths = tests
   python_files = test_*.py
   python_classes = Test*
   python_functions = test_*
   addopts = 
       --strict-markers
       --strict-config
       --verbose
       --tb=short
   markers =
       slow: marks tests as slow (deselect with -m "not slow")
       integration: marks tests as integration tests
       unit: marks tests as unit tests

Fixtures
~~~~~~~~

Common fixtures are defined in ``conftest.py`` and individual test files:

.. code-block:: python

   import pytest
   import tempfile
   from pathlib import Path

   @pytest.fixture
   def temp_dir():
       """Create a temporary directory for testing."""
       with tempfile.TemporaryDirectory() as tmp_dir:
           yield Path(tmp_dir)

   @pytest.fixture
   def mock_config():
       """Create mock configuration data."""
       return {
           "model": "test-model",
           "default_count": 2,
           "use_simulation": True
       }

Test Categories
---------------

Unit Tests
~~~~~~~~~~

Unit tests test individual functions and classes in isolation:

.. code-block:: python

   class TestNewsGatherer:
       """Unit tests for NewsGatherer class"""
       
       def test_calculate_sentiment_positive(self):
           """Test sentiment calculation for positive content."""
           gatherer = NewsGatherer()
           
           content = "hope love joy peace blessing grace"
           sentiment = gatherer.calculate_sentiment(content)
           
           assert sentiment > 0.5
       
       def test_calculate_sentiment_negative(self):
           """Test sentiment calculation for negative content."""
           gatherer = NewsGatherer()
           
           content = "suffering pain fear anxiety crisis"
           sentiment = gatherer.calculate_sentiment(content)
           
           assert sentiment < -0.5

Integration Tests
~~~~~~~~~~~~~~~~~

Integration tests test multiple components working together:

.. code-block:: python

   class TestContentGenerationIntegration:
       """Integration tests for content generation"""
       
       @patch('feedparser.parse')
       def test_generate_content_with_news_integration(self, mock_parse):
           """Test content generation with real news integration."""
           # Mock RSS feed
           mock_parse.return_value = create_mock_feed()
           
           generator = SimpleContentGenerator()
           result = generator.generate_content(count=1, use_news=True)
           
           assert 'content' in result
           assert 'news_sentiment' in result
           assert len(result['content']) == 1

CLI Tests
~~~~~~~~~

CLI tests test the command-line interface:

.. code-block:: python

   class TestCLIIntegration:
       """Test CLI integration and commands"""
       
       def test_cli_help_command(self):
           """Test CLI help command."""
           result = subprocess.run(
               [sys.executable, "content_gen_cli.py", "--help"],
               capture_output=True,
               text=True
           )
           
           assert result.returncode == 0
           assert "Biblical Content Generation CLI" in result.stdout
       
       def test_cli_generate_command(self):
           """Test CLI generate command."""
           result = subprocess.run(
               [sys.executable, "content_gen_cli.py", "generate", "--count", "1"],
               capture_output=True,
               text=True
           )
           
           assert result.returncode == 0

Mocking and Patching
--------------------

Using unittest.mock
~~~~~~~~~~~~~~~~~~~

Mock external dependencies and expensive operations:

.. code-block:: python

   from unittest.mock import Mock, patch, MagicMock

   class TestWithMocks:
       """Test class demonstrating mocking techniques"""
       
       @patch('feedparser.parse')
       def test_rss_gathering_with_mock(self, mock_parse):
           """Test RSS gathering with mocked feedparser."""
           # Setup mock
           mock_feed = Mock()
           mock_feed.entries = [create_mock_entry()]
           mock_parse.return_value = mock_feed
           
           # Test
           gatherer = NewsGatherer()
           articles = gatherer.gather_from_rss("test", config, 7)
           
           # Assertions
           assert len(articles) == 1
           mock_parse.assert_called_once()
       
       @patch('subprocess.run')
       def test_content_generation_with_mock_subprocess(self, mock_run):
           """Test content generation with mocked subprocess."""
           # Setup mock
           mock_run.return_value = Mock(
               returncode=0,
               stdout="Content generated successfully",
               stderr=""
           )
           
           # Test
           generator = SimpleContentGenerator()
           result = generator.generate_content_with_existing_system(["test"])
           
           # Assertions
           assert result['success'] is True
           mock_run.assert_called_once()

Mock Objects
~~~~~~~~~~~~

Create detailed mock objects for complex data structures:

.. code-block:: python

   def create_mock_news_data():
       """Create comprehensive mock news data."""
       return {
           "date": "2024-10-07T12:00:00",
           "period_days": 7,
           "sources_count": 4,
           "articles_analyzed": 25,
           "sentiment_analysis": {
               "overall_sentiment": "positive",
               "trending_topics": [
                   {
                       "topic": "faith",
                       "frequency": 8,
                       "sentiment": 0.7,
                       "relevance": 0.9
                   }
               ],
               "recommended_themes": ["Hope in Crisis", "Faith and Trust"],
               "audience_concerns": ["Anxiety", "Isolation"]
           }
       }

   def create_mock_entry():
       """Create mock RSS feed entry."""
       entry = Mock()
       entry.title = "Test Article Title"
       entry.summary = "Test article summary with faith content"
       entry.link = "https://example.com/article"
       entry.published = "2024-10-07"
       entry.published_parsed = (2024, 10, 7, 12, 0, 0, 0, 281, 0)
       return entry

Test Data Management
--------------------

Test Data Files
~~~~~~~~~~~~~~~

Use test data files for complex scenarios:

.. code-block:: python

   @pytest.fixture
   def sample_news_data():
       """Load sample news data from file."""
       test_data_dir = Path(__file__).parent / "test_data"
       with open(test_data_dir / "sample_news.json") as f:
           return json.load(f)

   @pytest.fixture
   def sample_bible_text():
       """Load sample Bible text for analysis."""
       test_data_dir = Path(__file__).parent / "test_data"
       with open(test_data_dir / "sample_text.txt") as f:
           return f.read()

Temporary Files
~~~~~~~~~~~~~~~

Use temporary files for file-based tests:

.. code-block:: python

   def test_config_file_operations(self, tmp_path):
       """Test configuration file operations."""
       config_file = tmp_path / "test_config.json"
       
       # Write test config
       config_data = {"model": "test", "count": 5}
       with open(config_file, 'w') as f:
           json.dump(config_data, f)
       
       # Test loading
       generator = SimpleContentGenerator()
       generator.config_file = config_file
       loaded_config = generator.load_config()
       
       assert loaded_config["model"] == "test"
       assert loaded_config["count"] == 5

Performance Testing
-------------------

Benchmark Tests
~~~~~~~~~~~~~~~

Test performance of critical operations:

.. code-block:: python

   import time
   import pytest

   class TestPerformance:
       """Performance tests for critical operations"""
       
       @pytest.mark.slow
       def test_news_gathering_performance(self):
           """Test news gathering performance."""
           gatherer = NewsGatherer()
           
           start_time = time.time()
           result = gatherer.gather_all_news(days_back=7)
           end_time = time.time()
           
           # Should complete within 30 seconds
           assert end_time - start_time < 30
           assert result['articles_analyzed'] > 0
       
       def test_sentiment_calculation_performance(self):
           """Test sentiment calculation performance."""
           gatherer = NewsGatherer()
           large_text = "faith hope love " * 1000  # Large text
           
           start_time = time.time()
           sentiment = gatherer.calculate_sentiment(large_text)
           end_time = time.time()
           
           # Should complete quickly even for large text
           assert end_time - start_time < 1.0
           assert -1 <= sentiment <= 1

Memory Testing
~~~~~~~~~~~~~~

Test memory usage for large operations:

.. code-block:: python

   import psutil
   import os

   def test_memory_usage_during_analysis(self):
       """Test memory usage during large text analysis."""
       process = psutil.Process(os.getpid())
       initial_memory = process.memory_info().rss
       
       # Perform memory-intensive operation
       analyzer = TextAnalyzer()
       large_corpus = ["text"] * 10000
       results = analyzer.analyze_batch(large_corpus)
       
       final_memory = process.memory_info().rss
       memory_increase = final_memory - initial_memory
       
       # Memory increase should be reasonable (less than 100MB)
       assert memory_increase < 100 * 1024 * 1024

Test Best Practices
------------------

Test Organization
~~~~~~~~~~~~~~~~~

- **Descriptive names**: Use ``test_what_happens_when_condition`` format
- **One assertion per test**: Focus on testing one behavior
- **Arrange-Act-Assert**: Structure tests clearly
- **Independent tests**: Tests should not depend on each other

.. code-block:: python

   def test_calculate_sentiment_returns_positive_score_for_positive_content(self):
       """Test that calculate_sentiment returns positive score for positive content."""
       # Arrange
       gatherer = NewsGatherer()
       positive_content = "hope love joy peace blessing grace faith"
       
       # Act
       sentiment_score = gatherer.calculate_sentiment(positive_content)
       
       # Assert
       assert sentiment_score > 0.5
       assert sentiment_score <= 1.0

Test Data Management
~~~~~~~~~~~~~~~~~~~~

- **Use fixtures**: For reusable test data
- **Avoid hardcoding**: Use parameterized tests for variations
- **Clean up**: Ensure tests clean up temporary files
- **Isolate data**: Each test should use its own data

.. code-block:: python

   @pytest.mark.parametrize("content,expected_range", [
       ("hope love joy", (0.5, 1.0)),
       ("pain suffering fear", (-1.0, -0.5)),
       ("neutral text", (-0.2, 0.2)),
   ])
   def test_sentiment_calculation_various_content(self, content, expected_range):
       """Test sentiment calculation with various content types."""
       gatherer = NewsGatherer()
       sentiment = gatherer.calculate_sentiment(content)
       
       min_val, max_val = expected_range
       assert min_val <= sentiment <= max_val

Error Handling
~~~~~~~~~~~~~~

Test both success and failure scenarios:

.. code-block:: python

   def test_news_gathering_handles_network_errors(self):
       """Test that news gathering handles network errors gracefully."""
       with patch('feedparser.parse', side_effect=Exception("Network error")):
           gatherer = NewsGatherer()
           articles = gatherer.gather_from_rss("test", config, 7)
           
           # Should return empty list, not crash
           assert articles == []

   def test_config_loading_handles_corrupted_file(self):
       """Test config loading with corrupted JSON file."""
       with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
           f.write('{"invalid": json}')  # Invalid JSON
           temp_file = f.name
       
       try:
           generator = SimpleContentGenerator()
           generator.config_file = Path(temp_file)
           config = generator.load_config()
           
           # Should fall back to default config
           assert 'model' in config
           assert config['model'] == 'llama3.1:8b'
       finally:
           os.unlink(temp_file)

Continuous Integration
----------------------

GitHub Actions
~~~~~~~~~~~~~~~

Example CI workflow:

.. code-block:: yaml

   name: Tests

   on: [push, pull_request]

   jobs:
     test:
       runs-on: ubuntu-latest
       strategy:
         matrix:
           python-version: [3.9, 3.10, 3.11]
       
       steps:
       - uses: actions/checkout@v3
       
       - name: Set up Python ${{ matrix.python-version }}
         uses: actions/setup-python@v3
         with:
           python-version: ${{ matrix.python-version }}
       
       - name: Install dependencies
         run: |
           python -m pip install --upgrade pip
           pip install -r requirements.txt
           pip install pytest pytest-cov
       
       - name: Run tests
         run: |
           python -m pytest --cov=religious_texts --cov-report=xml
       
       - name: Upload coverage
         uses: codecov/codecov-action@v3
         with:
           file: ./coverage.xml

Pre-commit Hooks
~~~~~~~~~~~~~~~~

Set up pre-commit hooks for code quality:

.. code-block:: yaml

   # .pre-commit-config.yaml
   repos:
   - repo: https://github.com/psf/black
     rev: 22.3.0
     hooks:
       - id: black
   
   - repo: https://github.com/pycqa/flake8
     rev: 4.0.1
     hooks:
       - id: flake8
   
   - repo: https://github.com/pycqa/isort
     rev: 5.10.1
     hooks:
       - id: isort

Test Documentation
------------------

Docstring Tests
~~~~~~~~~~~~~~~

Include doctests in docstrings:

.. code-block:: python

   def calculate_relevance(self, content: str, focus_topics: List[str]) -> float:
       """
       Calculate relevance score for content based on focus topics.
       
       Args:
           content (str): Content to analyze
           focus_topics (List[str]): List of focus topics
           
       Returns:
           float: Relevance score from 0 to 1
           
       Example:
           >>> gatherer = NewsGatherer()
           >>> score = gatherer.calculate_relevance("faith hope love", ["faith", "hope"])
           >>> print(f"Score: {score:.2f}")
           Score: 0.67
       """
       pass

Run doctests:

.. code-block:: bash

   python -m pytest --doctest-modules religious_texts/

This testing framework ensures comprehensive coverage and reliability of the Biblical Content Agent system. Regular testing helps maintain code quality and prevents regressions.