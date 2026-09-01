#!/usr/bin/env python3
"""
Content Generation CLI - Simplified Biblical Content Agent
Usage: python content_gen_cli.py [command] [options]

Commands:
    generate    Generate content with optional sentiment analysis
    news        Gather and analyze current news/sentiment
    status      Show system status and recent activity
"""

import argparse
import json
import sys
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import news gathering
from news_gathering import NewsGatherer, get_simulated_news_data

class SimpleContentGenerator:
    """
    Simplified content generation system with news sentiment analysis.
    
    This class provides a unified interface for generating biblical content
    with optional integration of current news sentiment analysis. It supports
    multiple content generation strategies and can automatically adapt content
    themes based on current trending topics and audience concerns.
    
    Attributes:
        config_dir (Path): Directory for configuration files
        config_file (Path): Path to main configuration file
        news_cache (Path): Path to news sentiment cache file
        output_dir (Path): Directory for generated content
        config (Dict): Current configuration settings
        news_gatherer (NewsGatherer): News sentiment analysis instance
    """
    
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
        
        # Initialize news gatherer
        self.news_gatherer = NewsGatherer(self.config_dir)
        
    def load_config(self) -> Dict:
        """
        Load configuration from file or create default configuration.
        
        Returns:
            Dict: Configuration dictionary containing settings for:
                - model: AI model to use for content generation
                - ollama_url: URL for Ollama API endpoint
                - default_count: Default number of content options to generate
                - angles_per_concept: Number of angles to generate per concept
                - auto_news: Whether to automatically use news sentiment
                - use_simulation: Whether to use simulated news data
        """
        default_config = {
            "model": "llama3.1:8b",
            "ollama_url": "http://localhost:11434",
            "primary_translation": "ESV",
            "default_count": 3,
            "angles_per_concept": 3,
            "temperature": 0.7,
            "schedule_day": "monday",
            "schedule_time": "09:00",
            "last_run": None,
            "auto_news": True,
            "use_simulation": True  # Use simulated data for now
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
        """
        Save configuration to file.
        
        Args:
            config (Dict): Configuration dictionary to save
        """
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"⚠️  Error saving config: {e}")
    
    def gather_news_sentiment(self, days_back: int = 7) -> Dict:
        """
        Gather and analyze news sentiment for relevant topics.
        
        This method collects news from Christian RSS sources, analyzes sentiment
        and trending topics, and generates content recommendations based on
        current audience concerns and interests.
        
        Args:
            days_back (int): Number of days to look back for news articles
            
        Returns:
            Dict: News sentiment analysis containing:
                - date: Analysis timestamp
                - period_days: Analysis period
                - sources_count: Number of sources analyzed
                - articles_analyzed: Number of articles processed
                - sentiment_analysis: Detailed sentiment data including:
                    - overall_sentiment: General sentiment direction
                    - trending_topics: Top trending topics with sentiment scores
                    - recommended_themes: Suggested content themes
                    - audience_concerns: Identified audience concerns
        """
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
        
        # Use simulated or real news data
        if self.config.get("use_simulation", True):
            print("🎭 Using simulated news data (demo mode)")
            news_data = get_simulated_news_data()
        else:
            print("🌐 Gathering real news from RSS feeds")
            news_data = self.news_gatherer.gather_all_news(days_back)
        
        # Cache the results
        try:
            with open(self.news_cache, 'w') as f:
                json.dump({"date": news_data["date"], "data": news_data}, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not cache news data: {e}")
        
        print(f"✅ Analyzed {news_data['sentiment_analysis']['trending_topics'].__len__()} trending topics")
        return news_data
    
    def generate_content_with_existing_system(self, concepts: List[str]) -> Dict:
        """Generate content using existing system"""
        print("🔄 Using existing content generation system...")
        
        # Try to use existing generate_content_with_angles.py
        script_path = Path(__file__).parent / "generate_content_with_angles.py"
        if script_path.exists():
            try:
                # Create a temporary version with custom concepts
                temp_script = Path(__file__).parent / "temp_generate.py"
                
                # Read the original script
                with open(script_path, 'r') as f:
                    content = f.read()
                
                # Modify concepts if provided
                if concepts:
                    concepts_str = json.dumps(concepts)
                    content = content.replace(
                        'concepts = [',
                        f'concepts = {concepts_str}  # Overridden'
                    )
                
                # Write temporary script
                with open(temp_script, 'w') as f:
                    f.write(content)
                
                # Run the script
                result = subprocess.run([sys.executable, str(temp_script)], 
                                      capture_output=True, text=True, cwd=Path(__file__).parent)
                
                # Clean up
                temp_script.unlink(missing_ok=True)
                
                if result.returncode == 0:
                    print("✅ Content generation completed")
                    return {"success": True, "output": result.stdout}
                else:
                    print(f"❌ Error: {result.stderr}")
                    return {"success": False, "error": result.stderr}
                    
            except Exception as e:
                print(f"❌ Error running existing system: {e}")
                return {"success": False, "error": str(e)}
        else:
            print("❌ Existing content generation script not found")
            return {"success": False, "error": "Script not found"}
    
    def generate_content(self, 
                        count: Optional[int] = None,
                        concepts: Optional[List[str]] = None,
                        use_news: bool = True) -> Dict:
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
            print(f"📝 Using provided concepts: {', '.join(concepts)}")
        elif news_data and use_news:
            # Use news-recommended themes
            target_concepts = news_data["sentiment_analysis"]["recommended_themes"][:count]
            print("📰 Using news-recommended themes")
        else:
            # Default concepts
            target_concepts = [
                "God's Unconditional Love",
                "Faith in Difficult Times", 
                "Power of Forgiveness",
                "Finding Purpose in Suffering",
                "Strength Through Humility"
            ][:count]
            print("📝 Using default concepts")
        
        # Generate content using existing system
        result = self.generate_content_with_existing_system(target_concepts)
        
        # Create summary
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "count": count,
                "concepts": target_concepts,
                "use_news": use_news,
                "model": self.config["model"]
            },
            "news_sentiment": news_data["sentiment_analysis"] if news_data else None,
            "generation_result": result,
            "summary": {
                "requested_concepts": len(target_concepts),
                "concepts_used": target_concepts,
                "success": result.get("success", False)
            }
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"content_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Saved to: {output_file}")
        
        # Update last run
        self.config["last_run"] = datetime.now().isoformat()
        self.save_config(self.config)
        
        return output_data
    
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
        print(f"🎭 Simulation Mode: {self.config.get('use_simulation', True)}")
        
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
        
        # System check
        print("\n🔧 System Components Check:")
        
        # Check existing scripts
        scripts = [
            "generate_content_with_angles.py",
            "content_agent_optimized.py",
            "rag.py"
        ]
        
        for script in scripts:
            script_path = Path(__file__).parent / script
            status = "✅" if script_path.exists() else "❌"
            print(f"  {status} {script}")
        
        # Check Ollama
        try:
            import requests
            response = requests.get(f"{self.config['ollama_url']}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                if self.config["model"] in model_names:
                    print(f"  ✅ Ollama: {self.config['model']} available")
                else:
                    print(f"  ⚠️  Ollama: {self.config['model']} not found (available: {', '.join(model_names[:3])})")
            else:
                print("  ❌ Ollama: Not responding")
        except Exception:
            print("  ❌ Ollama: Connection failed")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Biblical Content Generation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python content_gen_cli.py generate                           # Generate 3 concepts with news sentiment
    python content_gen_cli.py generate --count 5 --no-news      # Generate 5 concepts without news
    python content_gen_cli.py generate --concepts "Love" "Faith" # Use specific concepts
    python content_gen_cli.py news                               # Gather news sentiment only
    python content_gen_cli.py status                             # Show system status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate content')
    gen_parser.add_argument('--count', '-c', type=int, help='Number of concepts to generate')
    gen_parser.add_argument('--concepts', nargs='+', help='Specific concepts to use')
    gen_parser.add_argument('--no-news', action='store_true', help='Skip news sentiment analysis')
    
    # News command
    news_parser = subparsers.add_parser('news', help='Gather news sentiment')
    news_parser.add_argument('--days', type=int, default=7, help='Days back to analyze')
    news_parser.add_argument('--real', action='store_true', help='Use real RSS feeds instead of simulation')
    
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
    generator = SimpleContentGenerator()
    
    # Execute command
    if args.command == 'generate':
        result = generator.generate_content(
            count=args.count,
            concepts=args.concepts,
            use_news=not args.no_news
        )
        
        # Print summary
        if result and result.get('summary'):
            summary = result['summary']
            print(f"\n🎉 Generation Complete!")
            print(f"📊 Requested: {summary['requested_concepts']} concepts")
            print(f"📝 Concepts: {', '.join(summary['concepts_used'])}")
            print(f"✅ Success: {summary['success']}")
    
    elif args.command == 'news':
        # Toggle simulation mode if real flag is used
        if args.real:
            generator.config['use_simulation'] = False
            generator.save_config(generator.config)
        
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