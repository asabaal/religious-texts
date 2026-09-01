# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: religous_texts_env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Biblical Content Agent - Content Generation Demo
#
# This notebook demonstrates the core functionality of the Biblical Content Agent system, including:
# - Basic content generation
# - News sentiment integration
# - Multi-angle content creation
# - Configuration management

# %% [markdown]
# ## Setup and Imports

# %%
# Import necessary libraries
import sys
import os
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add parent directory to path for imports
sys.path.append('..')

# Import content generation modules
from content_gen_cli import SimpleContentGenerator
from news_gathering import NewsGatherer, get_simulated_news_data

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("✅ All imports successful!")
print(f"📅 Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% [markdown]
# ## 1. Basic Content Generation

# %%
# Initialize the content generator
generator = SimpleContentGenerator()

# Display current configuration
print("🔧 Current Configuration:")
for key, value in generator.config.items():
    print(f"  {key}: {value}")

print(f"\n📁 Output directory: {generator.output_dir}")
print(f"📁 Cache directory: {generator.config_dir}")

# %%
# Generate basic content (3 concepts, no news integration)
print("🚀 Generating basic content...")
result = generator.generate_content(
    count=3,
    use_news=False
)

# Display results summary
print(f"\n📊 Generation Summary:")
print(f"  Timestamp: {result['timestamp']}")
print(f"  Concepts generated: {len(result['content'])}")
print(f"  Total angles: {sum(len(c['angles']) for c in result['content'])}")

# Display concepts
print(f"\n🎯 Generated Concepts:")
for i, concept in enumerate(result['content'], 1):
    print(f"  {i}. {concept['concept']}")
    for j, angle in enumerate(concept['angles'], 1):
        print(f"     - Angle {j}: {angle['angle']}")
        print(f"       Words: {angle.get('word_count', 'N/A')}")
        if 'scripture_references' in angle and angle['scripture_references']:
            print(f"       Scriptures: {', '.join(angle['scripture_references'][:2])}")
    print()

# %% [markdown]
# ## 2. Content with Custom Themes

# %%
# Define custom themes
custom_themes = [
    "Faith in Modern Challenges",
    "Christian Leadership in Crisis",
    "Hope Through Difficult Times"
]

print(f"🎨 Generating content for custom themes:")
for theme in custom_themes:
    print(f"  - {theme}")

# Generate content for custom themes
custom_result = generator.generate_content(
    concepts=custom_themes,
    use_news=False
)

# Display detailed results for first concept
first_concept = custom_result['content'][0]
print(f"\n📝 Detailed Example - {first_concept['concept']}:")
print("=" * 50)

for i, angle in enumerate(first_concept['angles'], 1):
    print(f"\n🔸 Angle {i}: {angle['angle']}")
    print(f"Content preview: {angle['content'][:200]}...")
    print(f"Word count: {angle.get('word_count', 'N/A')}")
    if 'scripture_references' in angle:
        print(f"Scripture references: {angle['scripture_references']}")

# %% [markdown]
# ## 3. News Sentiment Analysis

# %%
# Get current news sentiment analysis
print("📰 Analyzing news sentiment...")
news_data = generator.gather_news_sentiment()

# Display sentiment analysis
sentiment = news_data['sentiment_analysis']
print(f"\n📊 News Sentiment Analysis:")
print(f"  Overall sentiment: {sentiment['overall_sentiment']}")
print(f"  Analysis period: {news_data['period_days']} days")
print(f"  Sources analyzed: {news_data['sources_count']}")
print(f"  Articles processed: {news_data['articles_analyzed']}")

# Display trending topics
print(f"\n🔥 Trending Topics:")
for i, topic in enumerate(sentiment['trending_topics'], 1):
    print(f"  {i}. {topic['topic']} (Frequency: {topic['frequency']}, Sentiment: {topic['sentiment']:.2f})")

# Display recommended themes
print(f"\n💡 Recommended Themes:")
for i, theme in enumerate(sentiment['recommended_themes'], 1):
    print(f"  {i}. {theme}")

# Display audience concerns
print(f"\n⚠️  Audience Concerns:")
for i, concern in enumerate(sentiment['audience_concerns'], 1):
    print(f"  {i}. {concern}")

# %%
# Visualize sentiment data
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('News Sentiment Analysis Dashboard', fontsize=16, fontweight='bold')

# 1. Trending Topics Bar Chart
topics = [t['topic'] for t in sentiment['trending_topics']]
frequencies = [t['frequency'] for t in sentiment['trending_topics']]
sentiments = [t['sentiment'] for t in sentiment['trending_topics']]

colors = ['green' if s > 0 else 'red' if s < 0 else 'gray' for s in sentiments]
axes[0, 0].barh(topics, frequencies, color=colors, alpha=0.7)
axes[0, 0].set_title('Trending Topics by Frequency')
axes[0, 0].set_xlabel('Frequency')

# 2. Sentiment Distribution
sentiment_scores = [t['sentiment'] for t in sentiment['trending_topics']]
axes[0, 1].hist(sentiment_scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 1].set_title('Sentiment Distribution')
axes[0, 1].set_xlabel('Sentiment Score')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)

# 3. Source Analysis
source_data = {
    'Sources': news_data['sources_count'],
    'Articles': news_data['articles_analyzed'],
    'Period (days)': news_data['period_days']
}

axes[1, 0].bar(source_data.keys(), source_data.values(), color=['blue', 'green', 'orange'])
axes[1, 0].set_title('Analysis Overview')
axes[1, 0].set_ylabel('Count')

# 4. Overall Sentiment Gauge
overall_sentiment = sentiment['overall_sentiment']
sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0, 'neutral_positive': 0.5}
sentiment_value = sentiment_map.get(overall_sentiment, 0)

axes[1, 1].bar(['Overall Sentiment'], [sentiment_value], 
              color='green' if sentiment_value > 0 else 'red' if sentiment_value < 0 else 'gray')
axes[1, 1].set_title(f'Overall Sentiment: {overall_sentiment.title()}')
axes[1, 1].set_ylabel('Sentiment Value')
axes[1, 1].set_ylim(-1, 1)

plt.tight_layout()
plt.show()

print("📈 Sentiment analysis visualization complete!")

# %% [markdown]
# ## 4. Content Generation with News Integration

# %%
# Generate content using news sentiment
print("🚀 Generating content with news sentiment integration...")
news_result = generator.generate_content(
    count=3,
    use_news=True
)

# Compare with and without news
print(f"\n📊 Comparison:")
print(f"  Without news: {len(result['content'])} concepts")
print(f"  With news: {len(news_result['content'])} concepts")

# Show news-influenced content
print(f"\n📰 News-Influenced Content:")
for i, concept in enumerate(news_result['content'], 1):
    print(f"  {i}. {concept['concept']}")
    
    # Show how news influenced this concept
    if 'news_sentiment' in news_result:
        news_themes = news_result['news_sentiment']['sentiment_analysis']['recommended_themes']
        matching_themes = [theme for theme in news_themes if any(word in concept['concept'].lower() for word in theme.lower().split())]
        if matching_themes:
            print(f"     📡 Influenced by news themes: {', '.join(matching_themes)}")
    
    # Show first angle preview
    if concept['angles']:
        angle = concept['angles'][0]
        print(f"     📝 Preview: {angle['content'][:150]}...")
    print()


# %% [markdown]
# ## 5. Content Analysis and Statistics

# %%
# Analyze generated content
def analyze_content(content_data, title="Content Analysis"):
    """Analyze content generation results."""
    concepts = content_data['content']
    
    # Basic statistics
    total_concepts = len(concepts)
    total_angles = sum(len(c['angles']) for c in concepts)
    total_words = sum(
        sum(angle.get('word_count', 0) for angle in c['angles'])
        for c in concepts
    )
    
    # Scripture references
    all_scriptures = []
    for concept in concepts:
        for angle in concept['angles']:
            scriptures = angle.get('scripture_references', [])
            all_scriptures.extend(scriptures)
    
    # Print analysis
    print(f"📊 {title}:")
    print(f"  Total concepts: {total_concepts}")
    print(f"  Total angles: {total_angles}")
    print(f"  Total words: {total_words:,}")
    print(f"  Average words per angle: {total_words // total_angles if total_angles > 0 else 0}")
    print(f"  Scripture references: {len(all_scriptures)}")
    
    if all_scriptures:
        # Most common scriptures
        from collections import Counter
        scripture_counts = Counter(all_scriptures)
        print(f"\n📖 Top Scripture References:")
        for scripture, count in scripture_counts.most_common(5):
            print(f"    {scripture}: {count} times")
    
    return {
        'concepts': total_concepts,
        'angles': total_angles,
        'words': total_words,
        'scriptures': len(all_scriptures)
    }

# Analyze both results
basic_stats = analyze_content(result, "Basic Content Generation")
print()
news_stats = analyze_content(news_result, "News-Integrated Content Generation")

# Comparison visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Content Generation Comparison', fontsize=16, fontweight='bold')

metrics = ['concepts', 'angles', 'words']
basic_values = [basic_stats[m] for m in metrics]
news_values = [news_stats[m] for m in metrics]

x = range(len(metrics))
width = 0.35

axes[0].bar([i - width/2 for i in x], basic_values, width, label='Basic', alpha=0.7)
axes[0].bar([i + width/2 for i in x], news_values, width, label='News-Integrated', alpha=0.7)
axes[0].set_xlabel('Metrics')
axes[0].set_ylabel('Count')
axes[0].set_title('Generation Comparison')
axes[0].set_xticks(x)
axes[0].set_xticklabels(['Concepts', 'Angles', 'Words'])
axes[0].legend()

# Scripture usage comparison
scripture_data = [basic_stats['scriptures'], news_stats['scriptures']]
axes[1].bar(['Basic', 'News-Integrated'], scripture_data, color=['blue', 'green'], alpha=0.7)
axes[1].set_title('Scripture References')
axes[1].set_ylabel('Count')

# Efficiency (words per concept)
efficiency_basic = basic_stats['words'] / basic_stats['concepts'] if basic_stats['concepts'] > 0 else 0
efficiency_news = news_stats['words'] / news_stats['concepts'] if news_stats['concepts'] > 0 else 0

axes[2].bar(['Basic', 'News-Integrated'], [efficiency_basic, efficiency_news], 
           color=['orange', 'purple'], alpha=0.7)
axes[2].set_title('Words per Concept')
axes[2].set_ylabel('Average Words')

plt.tight_layout()
plt.show()

print("📈 Content analysis complete!")

# %% [markdown]
# ## 6. Configuration Management

# %%
# Demonstrate configuration management
print("⚙️  Configuration Management Demo")
print("=" * 40)

# Show current config
print("\n📋 Current Configuration:")
for key, value in generator.config.items():
    print(f"  {key}: {value}")

# Modify configuration
print("\n🔧 Modifying configuration...")
original_count = generator.config['default_count']
generator.config['default_count'] = 5
generator.config['angles_per_concept'] = 2
generator.save_config(generator.config)

print(f"  Changed default_count: {original_count} → 5")
print(f"  Changed angles_per_concept: {generator.config.get('angles_per_concept', 'N/A')} → 2")

# Test new configuration
print("\n🧪 Testing new configuration...")
config_result = generator.generate_content(count=2, use_news=False)

print(f"  Generated concepts: {len(config_result['content'])}")
for concept in config_result['content']:
    print(f"    - {concept['concept']}: {len(concept['angles'])} angles")

# Restore original configuration
print("\n🔄 Restoring original configuration...")
generator.config['default_count'] = original_count
generator.save_config(generator.config)
print(f"  Restored default_count: {generator.config['default_count']}")

print("\n✅ Configuration management demo complete!")

# %% [markdown]
# ## 7. Save and Export Results

# %%
# Save results to files
output_dir = Path('../generated_content')
output_dir.mkdir(exist_ok=True)

# Save with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save basic content
basic_file = output_dir / f'demo_basic_{timestamp}.json'
with open(basic_file, 'w') as f:
    json.dump(result, f, indent=2)
print(f"💾 Saved basic content: {basic_file}")

# Save news-integrated content
news_file = output_dir / f'demo_news_{timestamp}.json'
with open(news_file, 'w') as f:
    json.dump(news_result, f, indent=2)
print(f"💾 Saved news content: {news_file}")

# Save news analysis
news_analysis_file = output_dir / f'demo_news_analysis_{timestamp}.json'
with open(news_analysis_file, 'w') as f:
    json.dump(news_data, f, indent=2)
print(f"💾 Saved news analysis: {news_analysis_file}")

# Create summary CSV
summary_data = []
for content_item in [result, news_result]:
    for concept in content_item['content']:
        for angle in concept['angles']:
            summary_data.append({
                'timestamp': content_item['timestamp'],
                'concept': concept['concept'],
                'angle': angle['angle'],
                'word_count': angle.get('word_count', 0),
                'scriptures': len(angle.get('scripture_references', [])),
                'content_preview': angle['content'][:100] + '...'
            })

summary_df = pd.DataFrame(summary_data)
summary_file = output_dir / f'demo_summary_{timestamp}.csv'
summary_df.to_csv(summary_file, index=False)
print(f"📊 Saved summary CSV: {summary_file}")

print(f"\n📁 All files saved to: {output_dir}")
print(f"📈 Generated {len(summary_data)} total content pieces")

# %% [markdown]
# ## 8. Summary and Next Steps

# %%
# Final summary
print("🎉 Biblical Content Agent Demo Complete!")
print("=" * 50)

print("\n✅ What we demonstrated:")
print("  1. Basic content generation with multiple angles")
print("  2. Custom theme generation")
print("  3. News sentiment analysis and visualization")
print("  4. News-integrated content generation")
print("  5. Content analysis and statistics")
print("  6. Configuration management")
print("  7. File export and summary generation")

print("\n📊 Generation Statistics:")
total_generated = len(summary_data)
total_words_generated = summary_df['word_count'].sum()
total_scriptures = summary_df['scriptures'].sum()

print(f"  Total content pieces: {total_generated}")
print(f"  Total words: {total_words_generated:,}")
print(f"  Total scripture references: {total_scriptures}")
print(f"  Average words per piece: {total_words_generated // total_generated if total_generated > 0 else 0}")

print("\n🚀 Next Steps:")
print("  1. Try the CLI interface: ./content-gen generate --help")
print("  2. Set up weekly automation: python weekly_content.py")
print("  3. Explore religious_texts analysis modules")
print("  4. Customize content themes and angles")
print("  5. Integrate with your content management system")

print(f"\n📁 All outputs saved in: {output_dir}")
print(f"⏰ Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n🙏 Thank you for using the Biblical Content Agent!")
