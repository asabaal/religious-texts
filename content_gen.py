#!/usr/bin/env python3
"""
Content Generation CLI - Unified Biblical Content Agent
Usage: content-gen [command] [options]

Commands:
    generate    Generate content with optional sentiment analysis
    schedule    Set up automated weekly content generation
    config      Configure system settings
    news        Gather and analyze current news/sentiment
    status      Show system status and recent activity
"""

import argparse
import json
import sys
import os
import subprocess
try:
    import schedule
except ImportError:
    print("⚠️  Warning: schedule module not found. Install with: pip install schedule")
    schedule = None
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import with error handling for missing dependencies
try:
    from rag import ensure_indexes, search_passages_for_concept, get_passage_texts_by_ref
except ImportError:
    print("⚠️  Warning: RAG module not found. Using fallback functions.")
    def ensure_indexes():
        pass
    def search_passages_for_concept(concept, top_k=3):
        return ["John 3:16", "Romans 8:28", "Matthew 5:3"][:top_k]
    def get_passage_texts_by_ref(ref):
        return {"ESV": "Sample biblical text for demonstration."}

try:
    from content_agent_optimized import generate_content_options, chat
except ImportError:
    print("⚠️  Warning: Content agent module not found. Using fallback functions.")
    def chat(prompt, system=None, temp=0.7):
        return '{"concept": "Sample", "script": "Sample script", "evaluation": {"scores": {"Engagement": 7}}}'
    def generate_content_options(k=3):
        return []

class ContentGenerator:
    """Unified content generation system with news sentiment analysis"""
    
    def __init__(self):
        self.config_dir = Path.home() / ".content-gen"
        self.config_file = self.config_dir / "config.json"
        self.news_cache = self.config_dir / "news_cache.json"
        self.output_dir = Path("generated_content")
        
        # Ensure directories exist
        self.config_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load or create config
        self.config = self.load_config()
        
    def load_config(self) -> Dict:
        """Load configuration from file or create default"""
        default_config = {
            "model": "llama3.1:8b",
            "ollama_url": "http://localhost:11434",
            "primary_translation": "ESV",
            "default_count": 3,
            "angles_per_concept": 3,
            "temperature": 0.7,
            "news_sources": [
                "christianitytoday.com",
                "relevantmagazine.com", 
                "churchleaders.com",
                "biblestudytools.com"
            ],
            "sentiment_keywords": [
                "faith", "church", "christian", "bible", "prayer",
                "god", "jesus", "worship", "spiritual", "religion"
            ],
            "schedule_day": "monday",
            "schedule_time": "09:00",
            "last_run": None,
            "auto_news": True
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults for any new keys
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                print(f"⚠️  Error loading config: {e}")
                return default_config
        else:
            self.save_config(default_config)
            return default_config
    
    def save_config(self, config: Dict):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"⚠️  Error saving config: {e}")
    
    def gather_news_sentiment(self, days_back: int = 7) -> Dict:
        """Gather and analyze news sentiment for relevant topics"""
        print("📰 Gathering news sentiment data...")
        
        # Check cache first
        if self.news_cache.exists():
            try:
                with open(self.news_cache, 'r') as f:
                    cached = json.load(f)
                    cache_date = datetime.fromisoformat(cached.get('date', '1970-01-01'))
                    if (datetime.now() - cache_date).days < 1:
                        print("📋 Using cached news data (less than 24h old)")
                        return cached['data']
            except Exception:
                pass
        
        # Simulate news gathering (in real implementation, use RSS/API)
        news_data = {
            "date": datetime.now().isoformat(),
            "sentiment_analysis": {
                "overall_sentiment": "neutral_positive",
                "trending_topics": [
                    {"topic": "faith in difficult times", "sentiment": 0.7, "relevance": 0.9},
                    {"topic": "community and fellowship", "sentiment": 0.8, "relevance": 0.8},
                    {"topic": "spiritual anxiety", "sentiment": -0.3, "relevance": 0.7},
                    {"topic": "church leadership", "sentiment": 0.2, "relevance": 0.6},
                    {"topic": "biblical literacy", "sentiment": 0.5, "relevance": 0.8}
                ],
                "recommended_themes": [
                    "Finding Hope in Uncertainty",
                    "Power of Christian Community", 
                    "Overcoming Spiritual Anxiety",
                    "Leadership Lessons from Scripture",
                    "Daily Bible Reading Habits"
                ],
                "audience_concerns": [
                    "Economic uncertainty affecting faith",
                    "Social isolation and need for connection",
                    "Political division and Christian response",
                    "Mental health and spiritual wellness"
                ]
            }
        }
        
        # Cache the results
        try:
            with open(self.news_cache, 'w') as f:
                json.dump({"date": news_data["date"], "data": news_data}, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not cache news data: {e}")
        
        print(f"✅ Analyzed {len(news_data['sentiment_analysis']['trending_topics'])} trending topics")
        return news_data
    
    def generate_content(self, 
                        count: Optional[int] = None,
                        concepts: Optional[List[str]] = None,
                        use_news: bool = True,
                        custom_prompt: Optional[str] = None) -> Dict:
        """Generate content with optional news sentiment integration"""
        
        count = count or self.config["default_count"]
        
        print(f"🎬 Generating {count} content options with {self.config['angles_per_concept']} angles each")
        
        # Gather news sentiment if enabled
        news_data = None
        if use_news and self.config["auto_news"]:
            news_data = self.gather_news_sentiment()
        
        # Determine concepts
        if concepts:
            target_concepts = concepts
        elif news_data and use_news:
            # Use news-recommended themes
            target_concepts = news_data["sentiment_analysis"]["recommended_themes"][:count]
            print("📰 Using news-recommended themes")
        else:
            # Generate concepts via AI
            target_concepts = self.generate_ai_concepts(count)
        
        # Initialize RAG system
        ensure_indexes()
        
        results = []
        for i, concept in enumerate(target_concepts[:count]):
            print(f"📝 Processing {i+1}/{count}: {concept}")
            
            # Find relevant passages
            passages = search_passages_for_concept(concept, top_k=3)
            if not passages:
                print(f"⚠️  No passages found for '{concept}'")
                continue
            
            # Generate content with news context if available
            content = self.generate_single_content(concept, passages[0], news_data)
            if content:
                results.append(content)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"content_{timestamp}.json"
        
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "count": count,
                "use_news": use_news,
                "model": self.config["model"]
            },
            "news_sentiment": news_data["sentiment_analysis"] if news_data else None,
            "content": results,
            "summary": {
                "total_concepts": len(results),
                "total_angles": sum(r.get("total_angles", 0) for r in results),
                "avg_quality": sum(r.get("avg_score", 0) for r in results) / len(results) if results else 0
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Saved to: {output_file}")
        print(f"📊 Generated {len(results)} concepts with {output_data['summary']['total_angles']} total angles")
        
        # Update last run
        self.config["last_run"] = datetime.now().isoformat()
        self.save_config(self.config)
        
        return output_data
    
    def generate_ai_concepts(self, count: int) -> List[str]:
        """Generate concepts using AI"""
        system = "You are a biblical content creator specializing in short-form video content."
        
        # Add news context if available
        context_prompt = ""
        if self.news_cache.exists():
            try:
                with open(self.news_cache, 'r') as f:
                    cached = json.load(f)
                    if "data" in cached and "audience_concerns" in cached["data"]["sentiment_analysis"]:
                        concerns = cached["data"]["sentiment_analysis"]["audience_concerns"]
                        context_prompt = f" Consider these current audience concerns: {', '.join(concerns[:3])}."
            except Exception:
                pass
        
        user_prompt = (
            f"Generate {count} compelling biblical concepts for short videos (3-5 words each)."
            f"{context_prompt} Focus on themes that resonate with modern audiences. "
            "Return only a JSON array of strings."
        )
        
        response = chat(user_prompt, system, self.config["temperature"])
        
        try:
            concepts = json.loads(response)
            if isinstance(concepts, list):
                return [str(c).strip() for c in concepts if str(c).strip()][:count]
        except json.JSONDecodeError:
            pass
        
        # Fallback concepts
        return [
            "God's Unconditional Love",
            "Faith in Difficult Times", 
            "Power of Forgiveness",
            "Finding Purpose in Suffering",
            "Strength Through Humility"
        ][:count]
    
    def generate_single_content(self, concept: str, passage: str, news_data: Optional[Dict]) -> Optional[Dict]:
        """Generate single content item with news context"""
        # Get passage text
        texts = get_passage_texts_by_ref(passage)
        primary_text = texts.get(self.config["primary_translation"])
        if not primary_text:
            return None
        
        # Build news-aware prompt
        news_context = ""
        if news_data:
            top_concerns = news_data["sentiment_analysis"]["audience_concerns"][:2]
            news_context = f" Current audience concerns include: {', '.join(top_concerns)}."
        
        system_prompt = "You are an expert biblical storyteller and content creator."
        user_prompt = f"""
        For this biblical passage, generate the TOP {self.config['angles_per_concept']} narrative angles for short-form video content.
        
        Concept: {concept}
        Passage: {passage}
        Text: {primary_text}{news_context}
        
        For each angle, provide:
        1. A compelling angle title (5-8 words)
        2. Target audience (specific demographic)
        3. Justification (why this angle works right now)
        4. A short script (hook, context, application, CTA)
        
        Return JSON format:
        {{
            "concept": "{concept}",
            "passage": "{passage}",
            "text": "{primary_text}",
            "narrative_angles": [
                {{
                    "angle_info": {{
                        "angle": "Angle title",
                        "target_audience": "Specific audience",
                        "justification": "Why this works"
                    }},
                    "script": "Complete script here",
                    "evaluation": {{"scores": {{"Engagement": 7, "Clarity": 8, "Authenticity": 7}}, "why": "Good fit"}}
                }}
            ],
            "total_angles": {self.config['angles_per_concept']}
        }}
        """
        
        response = chat(user_prompt, system_prompt, self.config["temperature"])
        
        try:
            result = json.loads(response)
            if "narrative_angles" in result:
                # Calculate average score
                total_score = 0
                count = 0
                for angle in result["narrative_angles"]:
                    if "evaluation" in angle and "scores" in angle["evaluation"]:
                        scores = angle["evaluation"]["scores"]
                        total_score += sum(scores.values())
                        count += len(scores)
                
                result["avg_score"] = total_score / count if count > 0 else 0
                return result
        except json.JSONDecodeError:
            print(f"⚠️  Could not parse AI response for '{concept}'")
        
        return None
    
    def show_status(self):
        """Display system status"""
        print("📊 Content Generation System Status")
        print("=" * 50)
        
        # Config info
        print(f"🤖 Model: {self.config['model']}")
        print(f"🔗 Ollama URL: {self.config['ollama_url']}")
        print(f"📖 Translation: {self.config['primary_translation']}")
        print(f"📝 Default Count: {self.config['default_count']} concepts")
        print(f"🎯 Angles per Concept: {self.config['angles_per_concept']}")
        
        # Schedule info
        print(f"⏰ Schedule: {self.config['schedule_day']}s at {self.config['schedule_time']}")
        
        # Last run
        if self.config.get("last_run"):
            last_run = datetime.fromisoformat(self.config["last_run"])
            print(f"🕐 Last Run: {last_run.strftime('%Y-%m-%d %H:%M:%S')}")
            time_since = datetime.now() - last_run
            print(f"⏱️  Time Since: {time_since.days} days, {time_since.seconds // 3600} hours")
        else:
            print("🕐 Last Run: Never")
        
        # Recent content
        if self.output_dir.exists():
            content_files = list(self.output_dir.glob("content_*.json"))
            if content_files:
                latest = max(content_files, key=lambda f: f.stat().st_mtime)
                mtime = datetime.fromtimestamp(latest.stat().st_mtime)
                print(f"📁 Latest Content: {latest.name} ({mtime.strftime('%Y-%m-%d %H:%M')})")
                
                # Show summary of latest
                try:
                    with open(latest, 'r') as f:
                        data = json.load(f)
                    summary = data.get("summary", {})
                    print(f"📊 Summary: {summary.get('total_concepts', 0)} concepts, {summary.get('total_angles', 0)} angles")
                    if summary.get('avg_quality'):
                        print(f"⭐ Avg Quality: {summary['avg_quality']:.1f}/10")
                except Exception:
                    pass
            else:
                print("📁 No content files found")
        
        # News cache
        if self.news_cache.exists():
            try:
                with open(self.news_cache, 'r') as f:
                    cached = json.load(f)
                cache_date = datetime.fromisoformat(cached.get('date', '1970-01-01'))
                age_hours = (datetime.now() - cache_date).total_seconds() / 3600
                print(f"📰 News Cache: {age_hours:.1f} hours old")
                
                if "data" in cached and "sentiment_analysis" in cached["data"]:
                    sentiment = cached["data"]["sentiment_analysis"]
                    print(f"😊 Sentiment: {sentiment.get('overall_sentiment', 'unknown')}")
                    print(f"🔥 Trending Topics: {len(sentiment.get('trending_topics', []))}")
            except Exception:
                print("📰 News Cache: Error reading")
        else:
            print("📰 News Cache: Not found")
    
    def run_scheduler(self):
        """Run the scheduled content generation"""
        if not schedule:
            print("❌ Schedule module not available. Install with: pip install schedule")
            return
            
        print(f"⏰ Starting content generation scheduler...")
        print(f"📅 Scheduled: {self.config['schedule_day']}s at {self.config['schedule_time']}")
        
        # Schedule the job
        schedule.every().__getattribute__(self.config['schedule_day'].lower()).at(self.config['schedule_time']).do(
            lambda: self.generate_content(use_news=True)
        )
        
        print("✅ Scheduler started. Press Ctrl+C to stop.")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            print("\n⏹️  Scheduler stopped.")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Biblical Content Generation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    content-gen generate                           # Generate 3 concepts with news sentiment
    content-gen generate --count 5 --no-news      # Generate 5 concepts without news
    content-gen generate --concepts "Love, Faith" # Use specific concepts
    content-gen schedule                           # Start weekly scheduler
    content-gen news                               # Gather news sentiment only
    content-gen status                             # Show system status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate content')
    gen_parser.add_argument('--count', '-c', type=int, help='Number of concepts to generate')
    gen_parser.add_argument('--concepts', nargs='+', help='Specific concepts to use')
    gen_parser.add_argument('--no-news', action='store_true', help='Skip news sentiment analysis')
    gen_parser.add_argument('--custom-prompt', help='Custom prompt for concept generation')
    
    # Schedule command
    sched_parser = subparsers.add_parser('schedule', help='Start automated scheduler')
    sched_parser.add_argument('--day', default='monday', help='Day of week (default: monday)')
    sched_parser.add_argument('--time', default='09:00', help='Time (default: 09:00)')
    
    # News command
    news_parser = subparsers.add_parser('news', help='Gather news sentiment')
    news_parser.add_argument('--days', type=int, default=7, help='Days back to analyze')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Manage configuration')
    config_parser.add_argument('--show', action='store_true', help='Show current config')
    config_parser.add_argument('--set', nargs=2, metavar=('KEY', 'VALUE'), help='Set config value')
    
    # Status command
    subparsers.add_parser('status', help='Show system status')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize generator
    generator = ContentGenerator()
    
    # Execute command
    if args.command == 'generate':
        result = generator.generate_content(
            count=args.count,
            concepts=args.concepts,
            use_news=not args.no_news,
            custom_prompt=args.custom_prompt
        )
        
        # Print summary
        if result and result.get('summary'):
            summary = result['summary']
            print(f"\n🎉 Generation Complete!")
            print(f"📊 {summary['total_concepts']} concepts, {summary['total_angles']} angles")
            if summary.get('avg_quality'):
                print(f"⭐ Average quality: {summary['avg_quality']:.1f}/10")
    
    elif args.command == 'schedule':
        # Update schedule config
        generator.config['schedule_day'] = args.day
        generator.config['schedule_time'] = args.time
        generator.save_config(generator.config)
        
        generator.run_scheduler()
    
    elif args.command == 'news':
        news_data = generator.gather_news_sentiment(args.days)
        
        print("\n📰 News Sentiment Analysis")
        print("=" * 30)
        print(f"📅 Period: Last {args.days} days")
        print(f"😊 Overall Sentiment: {news_data['sentiment_analysis']['overall_sentiment']}")
        
        print("\n🔥 Trending Topics:")
        for topic in news_data['sentiment_analysis']['trending_topics']:
            print(f"  • {topic['topic']} (sentiment: {topic['sentiment']:+.2f}, relevance: {topic['relevance']:.2f})")
        
        print("\n💡 Recommended Themes:")
        for theme in news_data['sentiment_analysis']['recommended_themes']:
            print(f"  • {theme}")
        
        print("\n🎯 Audience Concerns:")
        for concern in news_data['sentiment_analysis']['audience_concerns']:
            print(f"  • {concern}")
    
    elif args.command == 'config':
        if args.show:
            print(json.dumps(generator.config, indent=2))
        elif args.set:
            key, value = args.set
            # Convert value to appropriate type
            if value.isdigit():
                value = int(value)
            elif value.replace('.', '').isdigit():
                value = float(value)
            elif value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            
            generator.config[key] = value
            generator.save_config(generator.config)
            print(f"✅ Set {key} = {value}")
        else:
            print("Use --show to display config or --set KEY VALUE to update")
    
    elif args.command == 'status':
        generator.show_status()

if __name__ == "__main__":
    main()