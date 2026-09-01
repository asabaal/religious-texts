# Contributing to Religious Texts

Thank you for your interest in contributing to the Religious Texts project! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- A GitHub account

### Development Setup

1. **Fork the repository**
   ```bash
   # Fork the repository on GitHub, then clone your fork
   git clone https://github.com/your-username/religious-texts.git
   cd religious-texts
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Download NLTK data**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
   ```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Write clean, readable code following the project's style guidelines
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests and Quality Checks

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=religious_texts --cov-report=html

# Run code quality checks
pre-commit run --all-files

# Or run individual tools
black .
isort .
flake8 .
mypy religious_texts/
bandit -r religious_texts/
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat: add new feature description"
```

Follow [Conventional Commits](https://www.conventionalcommits.org/) specification:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `style:` for code style changes
- `refactor:` for code refactoring
- `test:` for adding or updating tests
- `chore:` for maintenance tasks

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear title and description
- Reference any relevant issues
- Include screenshots if applicable

## Code Style

We use several tools to maintain code quality:

### Formatting
- **Black**: Code formatting (line length: 88)
- **isort**: Import sorting

### Linting
- **flake8**: Code linting
- **mypy**: Type checking
- **bandit**: Security analysis

### Documentation
- All public functions and classes must have docstrings
- Use Google-style docstrings
- Include type hints where possible

## Testing

### Test Structure
```
tests/
├── test_data_acquisition/
├── test_text_analysis/
├── test_theological_analysis/
├── test_visualization/
└── test_integration/
```

### Writing Tests

1. **Unit tests**: Test individual functions and classes
2. **Integration tests**: Test component interactions
3. **Fixtures**: Use pytest fixtures for common setup

```python
import pytest
from religious_texts.text_analysis.frequency import word_frequency

class TestWordFrequency:
    def test_basic_functionality(self):
        text = "God is love. God is great."
        result = word_frequency(text)
        assert "god" in result
        assert result["god"] == 2
    
    @pytest.mark.parametrize("text,expected", [
        ("hello world", {"hello": 1, "world": 1}),
        ("hello hello world", {"hello": 2, "world": 1}),
    ])
    def test_parametrized_cases(self, text, expected):
        result = word_frequency(text)
        assert result == expected
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_text_analysis/test_frequency.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=religious_texts --cov-report=html

# Run only fast tests
pytest -m "not slow"
```

## Documentation

### Building Documentation

```bash
cd docs
make html
```

Documentation will be available in `docs/_build/html/`.

### Writing Documentation

- Use reStructuredText format
- Include code examples
- Add cross-references between sections
- Update API docs for new functions

## Project Structure

```
religious-texts/
├── religious_texts/           # Main package
│   ├── data_acquisition/      # Data loading and parsing
│   ├── text_analysis/         # NLP and text analysis
│   ├── theological_analysis/  # Religious text analysis
│   ├── visualization/         # Data visualization
│   └── specialized/           # Specialized analysis tools
├── tests/                     # Test suite
├── docs/                      # Documentation
├── notebooks/                 # Jupyter notebooks
└── examples/                  # Example scripts
```

## Release Process

Releases are automated through GitHub Actions:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create a release tag
4. GitHub Actions will build and publish to PyPI

## Getting Help

- **Issues**: Report bugs or request features on GitHub
- **Discussions**: Ask questions or share ideas
- **Documentation**: Check the docs for detailed information

## Code of Conduct

Please be respectful and inclusive in all interactions. Follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.