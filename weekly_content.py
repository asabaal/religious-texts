#!/usr/bin/env python3
"""
Weekly Content Generation Automation
This script handles the weekly cadence for generating 3-5 content options
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from content_gen_cli import SimpleContentGenerator

def run_weekly_generation():
    """Run the weekly content generation process"""
    print("🗓️  Starting Weekly Content Generation")
    print("=" * 50)
    
    # Initialize generator
    generator = SimpleContentGenerator()
    
    # Generate 3-5 content options with news sentiment
    count = 5  # Generate 5 concepts for weekly planning
    print(f"📊 Generating {count} content concepts with news sentiment analysis...")
    
    result = generator.generate_content(
        count=count,
        use_news=True
    )
    
    if result and result.get('summary', {}).get('success'):
        print("\n✅ Weekly content generation completed successfully!")
        
        # Create weekly summary
        summary = {
            "week_of": datetime.now().strftime("%Y-%m-%d"),
            "generation_timestamp": result["timestamp"],
            "concepts_generated": result["summary"]["concepts_used"],
            "news_sentiment": result.get("news_sentiment"),
            "files_generated": [str(f) for f in Path("generated_content").glob("content_*.json")],
            "next_steps": [
                "Review generated content for quality and relevance",
                "Select best angles for your target platforms",
                "Schedule content for the upcoming week",
                "Prepare visuals and supplementary materials"
            ]
        }
        
        # Save weekly summary
        summary_file = Path("weekly_summary") / f"week_{datetime.now().strftime('%Y%m%d')}.json"
        summary_file.parent.mkdir(exist_ok=True)
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Weekly summary saved to: {summary_file}")
        
        # Display recommendations
        if result.get("news_sentiment"):
            sentiment = result["news_sentiment"]
            print(f"\n📰 This Week's Insights:")
            print(f"😊 Overall Sentiment: {sentiment.get('overall_sentiment', 'unknown')}")
            
            print(f"\n🎯 Top Audience Concerns:")
            for concern in sentiment.get('audience_concerns', [])[:3]:
                print(f"  • {concern}")
            
            print(f"\n💡 Content Recommendations:")
            for theme in sentiment.get('recommended_themes', [])[:3]:
                print(f"  • Focus on: {theme}")
        
        print(f"\n📅 Content Planning Checklist:")
        print("  □ Review all generated angles and scripts")
        print("  □ Select 3-5 best options for this week")
        print("  □ Customize for your specific audience")
        print("  □ Schedule posting times across platforms")
        print("  □ Prepare accompanying visuals")
        print("  □ Monitor engagement and adjust strategy")
        
    else:
        print("❌ Weekly content generation failed!")
        print("Check system status with: ./content-gen status")
        return False
    
    return True

if __name__ == "__main__":
    success = run_weekly_generation()
    sys.exit(0 if success else 1)