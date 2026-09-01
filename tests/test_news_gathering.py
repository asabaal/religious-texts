"""
Tests for news gathering module
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from news_gathering import NewsGatherer, get_simulated_news_data


class TestNewsGatherer:
    """Test cases for NewsGatherer class"""
    
    @pytest.fixture
    def news_gatherer(self, tmp_path):
        """Create a NewsGatherer instance with temporary cache directory"""
        return NewsGatherer(cache_dir=tmp_path)
    
    @pytest.fixture
    def mock_feed_data(self):
        """Mock RSS feed data"""
        mock_feed = Mock()
        
        # Create mock entry objects with attributes
        entry1 = Mock()
        entry1.title = 'Finding Hope and Faith in Difficult Times'
        entry1.summary = 'A message about Christian faith during uncertainty'
        entry1.link = 'https://example.com/hope'
        # Use current date to ensure it's within days_back
        from datetime import datetime, timedelta
        now = datetime.now()
        entry1.published_parsed = (now.year, now.month, now.day, 12, 0, 0, 0, 281, 0)
        entry1.published = now.strftime('%Y-%m-%d')
        
        entry2 = Mock()
        entry2.title = 'Christian Faith and Community Fellowship'
        entry2.summary = 'The importance of faith in Christian community'
        entry2.link = 'https://example.com/community'
        entry2.published_parsed = (now.year, now.month, now.day-1, 14, 30, 0, 0, 280, 0)
        entry2.published = (now - timedelta(days=1)).strftime('%Y-%m-%d')
        
        mock_feed.entries = [entry1, entry2]
        return mock_feed
    
    def test_init(self, news_gatherer):
        """Test NewsGatherer initialization"""
        assert news_gatherer.cache_dir.exists()
        assert len(news_gatherer.sources) == 4
        assert 'christianitytoday' in news_gatherer.sources
        assert len(news_gatherer.positive_words) > 0
        assert len(news_gatherer.negative_words) > 0
        assert len(news_gatherer.religious_keywords) > 0
    
    def test_calculate_relevance(self, news_gatherer):
        """Test relevance calculation"""
        # High relevance content
        high_relevance = news_gatherer.calculate_relevance(
            "faith hope love god jesus prayer worship", 
            ["faith", "hope"]
        )
        assert high_relevance > 0.5
        
        # Low relevance content
        low_relevance = news_gatherer.calculate_relevance(
            "politics economics technology science", 
            ["faith", "hope"]
        )
        assert low_relevance < 0.3
    
    def test_calculate_sentiment(self, news_gatherer):
        """Test sentiment calculation"""
        # Positive sentiment
        positive = news_gatherer.calculate_sentiment(
            "hope love joy peace blessing grace faith"
        )
        assert positive > 0.5
        
        # Negative sentiment
        negative = news_gatherer.calculate_sentiment(
            "suffering pain fear anxiety crisis struggle hardship"
        )
        assert negative < -0.5
        
        # Neutral sentiment
        neutral = news_gatherer.calculate_sentiment(
            "article text content information news report"
        )
        assert abs(neutral) < 0.2
    
    def test_extract_keywords(self, news_gatherer):
        """Test keyword extraction"""
        content = "faith hope love god jesus christian church prayer worship bible"
        keywords = news_gatherer.extract_keywords(content)
        
        # Should extract religious keywords
        assert 'faith' in keywords
        assert 'god' in keywords
        assert 'jesus' in keywords
        assert len(keywords) > 0
    
    @patch('feedparser.parse')
    def test_gather_from_rss_success(self, mock_parse, news_gatherer, mock_feed_data):
        """Test successful RSS gathering"""
        mock_parse.return_value = mock_feed_data
        
        articles = news_gatherer.gather_from_rss(
            "test_source", 
            {"rss": "https://example.com/feed", "weight": 1.0, "focus": ["faith"]},
            days_back=7
        )
        
        assert len(articles) == 2
        assert articles[0]['title'] == 'Finding Hope and Faith in Difficult Times'
        assert articles[0]['source'] == 'test_source'
        assert articles[0]['relevance'] > 0
        assert 'sentiment' in articles[0]
        assert 'keywords' in articles[0]
    
    @patch('feedparser.parse')
    def test_gather_from_rss_failure(self, mock_parse, news_gatherer):
        """Test RSS gathering failure handling"""
        mock_parse.side_effect = Exception("Network error")
        
        articles = news_gatherer.gather_from_rss(
            "test_source", 
            {"rss": "https://example.com/feed", "weight": 1.0, "focus": ["faith"]},
            days_back=7
        )
        
        assert len(articles) == 0
    
    def test_analyze_trends(self, news_gatherer):
        """Test trend analysis"""
        articles = [
            {"keywords": ["faith", "hope"], "sentiment": 0.7},
            {"keywords": ["faith", "love"], "sentiment": 0.8},
            {"keywords": ["hope", "prayer"], "sentiment": 0.6},
            {"keywords": ["anxiety", "fear"], "sentiment": -0.3},
        ]
        
        trends = news_gatherer.analyze_trends(articles)
        
        assert len(trends) <= 5  # Should return top 5
        assert all('topic' in trend for trend in trends)
        assert all('frequency' in trend for trend in trends)
        assert all('sentiment' in trend for trend in trends)
        assert all('relevance' in trend for trend in trends)
        
        # Should be sorted by relevance
        if len(trends) > 1:
            assert trends[0]['relevance'] >= trends[1]['relevance']
    
    def test_calculate_overall_sentiment(self, news_gatherer):
        """Test overall sentiment calculation"""
        # Positive articles
        positive_articles = [
            {"sentiment": 0.7},
            {"sentiment": 0.8},
            {"sentiment": 0.6}
        ]
        assert news_gatherer.calculate_overall_sentiment(positive_articles) == "positive"
        
        # Negative articles
        negative_articles = [
            {"sentiment": -0.7},
            {"sentiment": -0.8},
            {"sentiment": -0.6}
        ]
        assert news_gatherer.calculate_overall_sentiment(negative_articles) == "negative"
        
        # Mixed articles
        mixed_articles = [
            {"sentiment": 0.7},
            {"sentiment": -0.3},
            {"sentiment": 0.2}
        ]
        assert news_gatherer.calculate_overall_sentiment(mixed_articles) in ["neutral_positive", "neutral"]
        
        # Empty articles
        assert news_gatherer.calculate_overall_sentiment([]) == "neutral"
    
    def test_generate_recommendations(self, news_gatherer):
        """Test recommendation generation"""
        trends = [
            {"topic": "faith", "sentiment": 0.7, "relevance": 0.9},
            {"topic": "hope", "sentiment": 0.8, "relevance": 0.8},
        ]
        
        articles = [
            {"title": "dealing with anxiety", "summary": "struggling with fear", "sentiment": -0.5},
            {"title": "finding peace", "summary": "overcoming worry", "sentiment": -0.3},
        ]
        
        recommendations = news_gatherer.generate_recommendations(trends, articles)
        
        assert 'themes' in recommendations
        assert 'concerns' in recommendations
        assert len(recommendations['themes']) == 5
        assert len(recommendations['concerns']) <= 4
        assert all(isinstance(theme, str) for theme in recommendations['themes'])
        assert all(isinstance(concern, str) for concern in recommendations['concerns'])
    
    def test_get_cached_news(self, news_gatherer):
        """Test news caching"""
        # Test no cache
        cached = news_gatherer.get_cached_news()
        assert cached is None
        
        # Test with cache
        from datetime import datetime
        cache_data = {
            "date": datetime.now().isoformat(),
            "data": {"test": "data"}
        }
        
        with open(news_gatherer.news_cache, 'w') as f:
            json.dump(cache_data, f)
        
        # Should return cached data if recent
        cached = news_gatherer.get_cached_news(max_age_hours=24)
        assert cached is not None
        assert cached["test"] == "data"
        
        # Should return None if cache is old
        old_cache = {
            "date": "2020-01-01T12:00:00",  # Very old
            "data": {"test": "data"}
        }
        
        with open(news_gatherer.news_cache, 'w') as f:
            json.dump(old_cache, f)
        
        cached = news_gatherer.get_cached_news(max_age_hours=24)
        assert cached is None
    
    @patch('feedparser.parse')
    def test_gather_all_news(self, mock_parse, news_gatherer, mock_feed_data):
        """Test complete news gathering process"""
        mock_parse.return_value = mock_feed_data
        
        result = news_gatherer.gather_all_news(days_back=7)
        
        assert 'date' in result
        assert 'period_days' in result
        assert 'sources_count' in result
        assert 'articles_analyzed' in result
        assert 'sentiment_analysis' in result
        assert 'articles' in result
        
        assert result['period_days'] == 7
        assert result['sources_count'] == 4
        assert result['articles_analyzed'] > 0
        
        sentiment = result['sentiment_analysis']
        assert 'overall_sentiment' in sentiment
        assert 'trending_topics' in sentiment
        assert 'recommended_themes' in sentiment
        assert 'audience_concerns' in sentiment
        
        # Check cache file was created
        assert news_gatherer.news_cache.exists()


class TestSimulatedData:
    """Test simulated news data function"""
    
    def test_get_simulated_news_data(self):
        """Test simulated news data generation"""
        data = get_simulated_news_data()
        
        assert 'date' in data
        assert 'period_days' in data
        assert 'sources_count' in data
        assert 'articles_analyzed' in data
        assert 'sentiment_analysis' in data
        
        assert data['period_days'] == 7
        assert data['sources_count'] == 4
        assert data['articles_analyzed'] == 25
        
        sentiment = data['sentiment_analysis']
        assert 'overall_sentiment' in sentiment
        assert 'trending_topics' in sentiment
        assert 'recommended_themes' in sentiment
        assert 'audience_concerns' in sentiment
        
        assert len(sentiment['trending_topics']) == 5
        assert len(sentiment['recommended_themes']) == 5
        assert len(sentiment['audience_concerns']) == 4
        
        # Check structure of trending topics
        for topic in sentiment['trending_topics']:
            assert 'topic' in topic
            assert 'frequency' in topic
            assert 'sentiment' in topic
            assert 'relevance' in topic
            assert isinstance(topic['sentiment'], (int, float))
            assert isinstance(topic['frequency'], int)
            assert isinstance(topic['relevance'], (int, float))


if __name__ == "__main__":
    pytest.main([__file__])