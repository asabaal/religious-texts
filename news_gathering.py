"""
News Gathering and Sentiment Analysis Module

This module provides functionality to gather news from Christian sources,
analyze sentiment, and identify trending topics relevant to biblical content creation.
"""

import json
import feedparser
import requests
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time

class NewsGatherer:
    """
    Gather and analyze news from Christian sources.
    
    This class provides comprehensive news gathering and sentiment analysis
    functionality specifically tailored for Christian content creation. It
    collects articles from multiple Christian RSS feeds, analyzes sentiment
    and relevance, identifies trending topics, and generates content recommendations.
    
    Attributes:
        cache_dir (Path): Directory for cache files
        cache_file (Path): Path to news cache file
        news_cache (Path): Alias for cache_file compatibility
        sources (Dict): Configuration of RSS news sources
        positive_words (set): Words indicating positive sentiment
        negative_words (set): Words indicating negative sentiment
        religious_keywords (set): Keywords relevant to religious content
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".content-gen"
        self.cache_file = self.cache_dir / "news_cache.json"
        self.news_cache = self.cache_file  # Add alias for compatibility
        self.cache_dir.mkdir(exist_ok=True)
        
        # Christian news sources
        self.sources = {
            "christianitytoday": {
                "rss": "https://www.christianitytoday.com/rss",
                "weight": 1.0,
                "focus": ["evangelical", "culture", "church"]
            },
            "relevantmagazine": {
                "rss": "https://relevantmagazine.com/feed/",
                "weight": 0.9,
                "focus": ["young_adults", "culture", "faith"]
            },
            "churchleaders": {
                "rss": "https://www.churchleaders.com/feed/",
                "weight": 0.8,
                "focus": ["leadership", "ministry", "church"]
            },
            "biblestudytools": {
                "rss": "https://www.biblestudytools.com/rss",
                "weight": 0.7,
                "focus": ["bible", "study", "devotion"]
            }
        }
        
        # Sentiment keywords
        self.positive_words = {
            'hope', 'faith', 'love', 'joy', 'peace', 'blessing', 'grace', 
            'mercy', 'strength', 'courage', 'healing', 'restoration', 
            'unity', 'community', 'fellowship', 'worship', 'prayer'
        }
        
        self.negative_words = {
            'suffering', 'pain', 'crisis', 'fear', 'anxiety', 'depression',
            'division', 'conflict', 'struggle', 'hardship', 'difficulty',
            'loneliness', 'isolation', 'doubt', 'discouragement'
        }
        
        self.religious_keywords = {
            'god', 'jesus', 'christ', 'christian', 'church', 'bible', 
            'prayer', 'worship', 'faith', 'spiritual', 'scripture',
            'gospel', 'ministry', 'theology', 'doctrine'
        }
    
    def gather_from_rss(self, source_name: str, source_config: Dict, days_back: int = 7) -> List[Dict]:
        """
        Gather articles from RSS feed.
        
        Args:
            source_name (str): Name identifier for the RSS source
            source_config (Dict): Configuration containing RSS URL, weight, and focus topics
            days_back (int): Number of days to look back for articles
            
        Returns:
            List[Dict]: List of processed articles with metadata including:
                - title: Article title
                - summary: Article summary
                - link: Article URL
                - date: Publication date
                - source: Source name
                - relevance: Relevance score (0-1)
                - sentiment: Sentiment score (-1 to 1)
                - keywords: Extracted keywords
        """
        articles = []
        
        try:
            feed = feedparser.parse(source_config["rss"])
            
            for entry in feed.entries:
                # Check article date
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    article_date = datetime(*entry.published_parsed[:6])
                    if (datetime.now() - article_date).days > days_back:
                        continue
                
                # Extract content
                title = entry.title.lower()
                summary = getattr(entry, 'summary', '').lower()
                content = f"{title} {summary}"
                
                # Calculate relevance score
                relevance = self.calculate_relevance(content, source_config["focus"])
                
                if relevance > 0.3:  # Only include relevant articles
                    articles.append({
                        "title": entry.title,
                        "summary": getattr(entry, 'summary', ''),
                        "link": getattr(entry, 'link', ''),
                        "date": getattr(entry, 'published', ''),
                        "source": source_name,
                        "relevance": relevance,
                        "sentiment": self.calculate_sentiment(content),
                        "keywords": self.extract_keywords(content)
                    })
                
        except Exception as e:
            print(f"⚠️  Error fetching from {source_name}: {e}")
        
        return articles
    
    def calculate_relevance(self, content: str, focus_areas: List[str]) -> float:
        """Calculate relevance score based on content and focus areas"""
        content_words = set(content.split())
        religious_overlap = len(content_words & self.religious_keywords)
        
        # Check focus area relevance
        focus_score = 0
        for focus in focus_areas:
            if focus.replace('_', ' ') in content:
                focus_score += 0.2
        
        # Base relevance from religious content
        base_relevance = min(religious_overlap / 10, 0.8)
        
        return min(base_relevance + focus_score, 1.0)
    
    def calculate_sentiment(self, content: str) -> float:
        """Calculate sentiment score (-1 to 1)"""
        content_words = set(re.findall(r'\b\w+\b', content.lower()))
        
        positive_count = len(content_words & self.positive_words)
        negative_count = len(content_words & self.negative_words)
        total_relevant = positive_count + negative_count
        
        if total_relevant == 0:
            return 0.0
        
        # Normalize to -1 to 1 range
        sentiment = (positive_count - negative_count) / total_relevant
        return max(-1.0, min(1.0, sentiment))
    
    def extract_keywords(self, content: str) -> List[str]:
        """Extract relevant keywords from content"""
        content_words = set(re.findall(r'\b\w+\b', content.lower()))
        
        # Religious keywords found
        religious_found = list(content_words & self.religious_keywords)
        
        # Additional significant words (longer than 4 chars)
        significant = [word for word in content_words 
                      if len(word) > 4 and word not in religious_found]
        
        return religious_found + significant[:5]  # Limit to prevent too many keywords
    
    def gather_all_news(self, days_back: int = 7) -> Dict:
        """Gather news from all sources"""
        print("📰 Gathering news from Christian sources...")
        
        all_articles = []
        
        for source_name, source_config in self.sources.items():
            print(f"  📡 Fetching from {source_name}...")
            articles = self.gather_from_rss(source_name, source_config, days_back)
            all_articles.extend(articles)
            time.sleep(1)  # Be respectful to servers
        
        # Analyze trends
        trends = self.analyze_trends(all_articles)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(trends, all_articles)
        
        result = {
            "date": datetime.now().isoformat(),
            "period_days": days_back,
            "sources_count": len(self.sources),
            "articles_analyzed": len(all_articles),
            "sentiment_analysis": {
                "overall_sentiment": self.calculate_overall_sentiment(all_articles),
                "trending_topics": trends,
                "recommended_themes": recommendations["themes"],
                "audience_concerns": recommendations["concerns"]
            },
            "articles": all_articles[:20]  # Keep top 20 articles
        }
        
        # Cache results
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not cache news data: {e}")
        
        print(f"✅ Analyzed {len(all_articles)} articles from {len(self.sources)} sources")
        return result
    
    def analyze_trends(self, articles: List[Dict]) -> List[Dict]:
        """Analyze trending topics from articles"""
        # Count keyword frequencies
        keyword_counts = {}
        topic_sentiments = {}
        
        for article in articles:
            for keyword in article["keywords"]:
                if keyword not in keyword_counts:
                    keyword_counts[keyword] = 0
                    topic_sentiments[keyword] = []
                
                keyword_counts[keyword] += 1
                topic_sentiments[keyword].append(article["sentiment"])
        
        # Create trending topics
        trends = []
        for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            avg_sentiment = sum(topic_sentiments[keyword]) / len(topic_sentiments[keyword])
            relevance = min(count / len(articles), 1.0)
            
            trends.append({
                "topic": keyword,
                "frequency": count,
                "sentiment": avg_sentiment,
                "relevance": relevance
            })
        
        return sorted(trends, key=lambda x: x["relevance"], reverse=True)[:5]
    
    def calculate_overall_sentiment(self, articles: List[Dict]) -> str:
        """Calculate overall sentiment category"""
        if not articles:
            return "neutral"
        
        avg_sentiment = sum(article["sentiment"] for article in articles) / len(articles)
        
        if avg_sentiment > 0.2:
            return "positive"
        elif avg_sentiment < -0.2:
            return "negative"
        else:
            return "neutral_positive" if avg_sentiment > 0 else "neutral"
    
    def generate_recommendations(self, trends: List[Dict], articles: List[Dict]) -> Dict:
        """Generate content recommendations based on trends"""
        # Extract audience concerns from negative sentiment articles
        negative_articles = [a for a in articles if a["sentiment"] < -0.1]
        concerns = []
        
        concern_patterns = {
            "anxiety": ["anxiety", "fear", "worry", "stress"],
            "isolation": ["lonely", "alone", "isolated", "disconnected"],
            "doubt": ["doubt", "questioning", "uncertainty", "struggle"],
            "division": ["division", "conflict", "disagreement", "polarization"]
        }
        
        for article in negative_articles:
            content = f"{article['title']} {article['summary']}".lower()
            for concern, keywords in concern_patterns.items():
                if any(keyword in content for keyword in keywords):
                    if concern not in concerns:
                        concerns.append(f"Emotional {concern}")
        
        # Generate themes from positive trends
        positive_trends = [t for t in trends if t["sentiment"] > 0.1]
        themes = []
        
        theme_templates = {
            "hope": ["Finding Hope in {context}", "Hope Beyond {situation}"],
            "faith": ["Faith in the Midst of {challenge}", "Unshakeable Faith"],
            "community": ["Power of Christian Community", "Finding Your Tribe"],
            "strength": ["God's Strength in Weakness", "Finding Inner Strength"],
            "peace": ["Peace Beyond Understanding", "Finding Peace in Chaos"]
        }
        
        for trend in positive_trends[:3]:
            topic = trend["topic"]
            for theme_type, templates in theme_templates.items():
                if theme_type in topic or topic in theme_type:
                    # Use only the placeholders that exist in the template
                    if "{context}" in templates[0]:
                        themes.append(templates[0].format(context=topic.title()))
                    elif "{situation}" in templates[0]:
                        themes.append(templates[0].format(situation=topic.title()))
                    elif "{challenge}" in templates[0]:
                        themes.append(templates[0].format(challenge=topic.title()))
                    else:
                        themes.append(templates[0])
                    break
        
        # Ensure we have enough themes
        default_themes = [
            "God's Unconditional Love",
            "Faith in Difficult Times",
            "Power of Christian Community",
            "Finding Purpose in Suffering",
            "Strength Through Humility"
        ]
        
        while len(themes) < 5:
            themes.append(default_themes[len(themes)])
        
        return {
            "themes": themes[:5],
            "concerns": concerns[:4] if concerns else ["Life challenges", "Spiritual growth", "Daily faith struggles"]
        }
    
    def get_cached_news(self, max_age_hours: int = 24) -> Optional[Dict]:
        """Get cached news if recent enough"""
        if not self.cache_file.exists():
            return None
        
        try:
            with open(self.cache_file, 'r') as f:
                cached = json.load(f)
            
            cache_date = datetime.fromisoformat(cached.get('date', '1970-01-01'))
            age_hours = (datetime.now() - cache_date).total_seconds() / 3600
            
            if age_hours < max_age_hours:
                return cached.get('data', cached)
        except Exception:
            pass
        
        return None

# Simulated news data for when RSS is not available
def get_simulated_news_data() -> Dict:
    """Get simulated news data for testing/demo purposes"""
    return {
        "date": datetime.now().isoformat(),
        "period_days": 7,
        "sources_count": 4,
        "articles_analyzed": 25,
        "sentiment_analysis": {
            "overall_sentiment": "neutral_positive",
            "trending_topics": [
                {"topic": "faith", "frequency": 8, "sentiment": 0.7, "relevance": 0.9},
                {"topic": "community", "frequency": 6, "sentiment": 0.8, "relevance": 0.8},
                {"topic": "anxiety", "frequency": 5, "sentiment": -0.3, "relevance": 0.7},
                {"topic": "leadership", "frequency": 4, "sentiment": 0.2, "relevance": 0.6},
                {"topic": "prayer", "frequency": 7, "sentiment": 0.5, "relevance": 0.8}
            ],
            "recommended_themes": [
                "Finding Hope in Uncertainty",
                "Power of Christian Community", 
                "Overcoming Spiritual Anxiety",
                "Leadership Lessons from Scripture",
                "Daily Prayer and Devotion"
            ],
            "audience_concerns": [
                "Economic uncertainty affecting faith",
                "Social isolation and need for connection",
                "Political division and Christian response",
                "Mental health and spiritual wellness"
            ]
        }
    }