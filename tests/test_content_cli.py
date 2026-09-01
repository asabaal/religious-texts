"""
Tests for content generation CLI
"""

import pytest
import json
import subprocess
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from content_gen_cli import SimpleContentGenerator


class TestSimpleContentGenerator:
    """Test cases for SimpleContentGenerator class"""
    
    @pytest.fixture
    def generator(self, tmp_path):
        """Create a SimpleContentGenerator instance with temporary directories"""
        # Mock the config directory to use temp path
        with patch('pathlib.Path.home', return_value=tmp_path):
            generator = SimpleContentGenerator()
        return generator
    
    def test_init(self, generator):
        """Test generator initialization"""
        assert generator.config_dir.exists()
        assert generator.output_dir.exists()
        assert generator.config_file.exists()
        
        # Check default config values
        assert generator.config['model'] == 'llama3.1:8b'
        assert generator.config['default_count'] == 3
        assert generator.config['angles_per_concept'] == 3
        assert generator.config['auto_news'] is True
        assert generator.config['use_simulation'] is True
    
    def test_load_config_default(self, tmp_path):
        """Test loading default configuration"""
        with patch('pathlib.Path.home', return_value=tmp_path):
            generator = SimpleContentGenerator()
        
        assert 'model' in generator.config
        assert 'default_count' in generator.config
        assert generator.config['default_count'] == 3
    
    def test_save_and_load_config(self, generator):
        """Test saving and loading configuration"""
        # Modify config
        generator.config['default_count'] = 5
        generator.config['test_key'] = 'test_value'
        
        # Save config
        generator.save_config(generator.config)
        
        # Create new generator instance
        with patch('pathlib.Path.home', return_value=generator.config_dir.parent):
            new_generator = SimpleContentGenerator()
        
        # Check config was loaded
        assert new_generator.config['default_count'] == 5
        assert new_generator.config['test_key'] == 'test_value'
    
    def test_gather_news_sentiment_cached(self, generator):
        """Test news sentiment gathering with cache"""
        # Create cached news data
        from datetime import datetime
        cache_data = {
            "date": datetime.now().isoformat(),
            "data": {
                "sentiment_analysis": {
                    "overall_sentiment": "positive",
                    "trending_topics": [{"topic": "faith", "sentiment": 0.7, "relevance": 0.9}],
                    "recommended_themes": ["Hope and Faith"],
                    "audience_concerns": ["Anxiety"]
                }
            }
        }
        
        with open(generator.news_cache, 'w') as f:
            json.dump(cache_data, f)
        
        # Should use cached data
        news_data = generator.gather_news_sentiment()
        
        assert news_data['sentiment_analysis']['overall_sentiment'] in ['positive', 'neutral_positive']
        assert len(news_data['sentiment_analysis']['trending_topics']) == 1
        assert news_data['sentiment_analysis']['trending_topics'][0]['topic'] == 'faith'
    
    def test_gather_news_sentiment_simulation(self, generator):
        """Test news sentiment gathering with simulation"""
        # Ensure no cache exists
        if generator.news_cache.exists():
            generator.news_cache.unlink()
        
        # Should use simulated data
        news_data = generator.gather_news_sentiment()
        
        assert 'sentiment_analysis' in news_data
        assert 'overall_sentiment' in news_data['sentiment_analysis']
        assert 'trending_topics' in news_data['sentiment_analysis']
        assert 'recommended_themes' in news_data['sentiment_analysis']
        assert 'audience_concerns' in news_data['sentiment_analysis']
        
        # Check simulated data structure
        assert len(news_data['sentiment_analysis']['trending_topics']) == 5
        assert len(news_data['sentiment_analysis']['recommended_themes']) == 5
        assert len(news_data['sentiment_analysis']['audience_concerns']) == 4
    
    @patch('subprocess.run')
    def test_generate_content_with_existing_system_success(self, mock_run, generator):
        """Test content generation using existing system"""
        # Mock successful subprocess run
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Content generated successfully",
            stderr=""
        )
        
        # Mock the script path existence
        script_path = Path(__file__).parent.parent / "generate_content_with_angles.py"
        
        with patch('pathlib.Path.exists', return_value=True):
            result = generator.generate_content_with_existing_system(["Test Concept"])
        
        assert result['success'] is True
        assert 'Content generated successfully' in result['output']
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_generate_content_with_existing_system_failure(self, mock_run, generator):
        """Test content generation failure handling"""
        # Mock failed subprocess run
        mock_run.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="Generation failed"
        )
        
        # Mock the script path existence
        with patch('pathlib.Path.exists', return_value=True):
            result = generator.generate_content_with_existing_system(["Test Concept"])
        
        assert result['success'] is False
        assert 'Generation failed' in result['error']
    
    def test_generate_content_with_existing_system_no_script(self, generator):
        """Test content generation when script doesn't exist"""
        # Mock script path doesn't exist
        with patch('pathlib.Path.exists', return_value=False):
            result = generator.generate_content_with_existing_system(["Test Concept"])
        
        assert result['success'] is False
        assert 'Script not found' in result['error']
    
    @patch.object(SimpleContentGenerator, 'generate_content_with_existing_system')
    @patch.object(SimpleContentGenerator, 'gather_news_sentiment')
    def test_generate_content_with_news(self, mock_news, mock_generate, generator):
        """Test content generation with news sentiment"""
        # Mock news data
        mock_news.return_value = {
            "sentiment_analysis": {
                "recommended_themes": ["Hope in Crisis", "Faith and Trust"]
            }
        }
        
        # Mock successful generation
        mock_generate.return_value = {"success": True, "output": "Generated"}
        
        result = generator.generate_content(count=2, use_news=True)
        
        assert result['config']['count'] == 2
        assert result['config']['use_news'] is True
        assert 'news_sentiment' in result
        assert result['summary']['success'] is True
        
        # Should use news-recommended themes
        mock_generate.assert_called_once()
        call_args = mock_generate.call_args[0][0]  # First positional argument
        assert "Hope in Crisis" in call_args
        assert "Faith and Trust" in call_args
    
    @patch.object(SimpleContentGenerator, 'generate_content_with_existing_system')
    def test_generate_content_custom_concepts(self, mock_generate, generator):
        """Test content generation with custom concepts"""
        # Mock successful generation
        mock_generate.return_value = {"success": True, "output": "Generated"}
        
        custom_concepts = ["Custom Theme 1", "Custom Theme 2"]
        result = generator.generate_content(concepts=custom_concepts, use_news=False)
        
        assert result['config']['concepts'] == custom_concepts
        assert result['config']['use_news'] is False
        
        # Should use custom concepts
        mock_generate.assert_called_once()
        call_args = mock_generate.call_args[0][0]
        assert call_args == custom_concepts
    
    @patch.object(SimpleContentGenerator, 'generate_content_with_existing_system')
    def test_generate_content_default_concepts(self, mock_generate, generator):
        """Test content generation with default concepts"""
        # Mock successful generation
        mock_generate.return_value = {"success": True, "output": "Generated"}
        
        result = generator.generate_content(use_news=False)
        
        # Should use default concepts
        mock_generate.assert_called_once()
        call_args = mock_generate.call_args[0][0]
        assert len(call_args) == 3  # Default count
        assert "God's Unconditional Love" in call_args
    
    def test_show_status(self, generator):
        """Test status display"""
        # This test mainly ensures the method runs without error
        # In a real scenario, you might capture stdout and verify content
        try:
            generator.show_status()
            status_success = True
        except Exception as e:
            status_success = False
            print(f"Status display failed: {e}")
        
        assert status_success is True


class TestCLIIntegration:
    """Test CLI integration and command-line interface"""
    
    def test_cli_help(self):
        """Test CLI help command"""
        result = subprocess.run(
            [sys.executable, "content_gen_cli.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        assert result.returncode == 0
        assert "Biblical Content Generation CLI" in result.stdout
        assert "generate" in result.stdout
        assert "news" in result.stdout
        assert "status" in result.stdout
    
    def test_cli_status_command(self):
        """Test CLI status command"""
        result = subprocess.run(
            [sys.executable, "content_gen_cli.py", "status"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        assert result.returncode == 0
        assert "Content Generation System Status" in result.stdout
        assert "Model:" in result.stdout
        assert "System Components Check:" in result.stdout
    
    def test_cli_news_command(self):
        """Test CLI news command"""
        result = subprocess.run(
            [sys.executable, "content_gen_cli.py", "news"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        assert result.returncode == 0
        assert "News Sentiment Analysis" in result.stdout
        assert "Overall Sentiment:" in result.stdout
        assert "Trending Topics:" in result.stdout
        assert "Recommended Themes:" in result.stdout
    
    def test_cli_config_show(self):
        """Test CLI config show command"""
        result = subprocess.run(
            [sys.executable, "content_gen_cli.py", "config", "--show"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        assert result.returncode == 0
        # Should output JSON configuration
        config_data = json.loads(result.stdout)
        assert "model" in config_data
        assert "default_count" in config_data
    
    def test_cli_config_set(self):
        """Test CLI config set command"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set custom home directory for testing
            env = os.environ.copy()
            env['HOME'] = temp_dir
            
            result = subprocess.run(
                [sys.executable, "content_gen_cli.py", "config", "--set", "default_count", "5"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
                env=env
            )
            
            assert result.returncode == 0
            assert "Set default_count = 5" in result.stdout
            
            # Verify the setting was saved
            result = subprocess.run(
                [sys.executable, "content_gen_cli.py", "config", "--show"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
                env=env
            )
            
            config_data = json.loads(result.stdout)
            assert config_data['default_count'] == 5


if __name__ == "__main__":
    pytest.main([__file__])