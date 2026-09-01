Examples and Demos
==================

This section contains practical examples and runnable demos for the Biblical Content Agent.

Basic Examples
--------------

Example 1: Simple Content Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate 3 content options with default settings:

.. code-block:: bash

   ./content-gen generate

Expected output:
- Creates 3 content concepts
- Each concept has 3 different angles
- Total: 9 content pieces
- Saved to ``generated_content/content_YYYYMMDD_HHMMSS.json``

Example 2: Custom Number of Concepts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate 5 content options:

.. code-block:: bash

   ./content-gen generate --count 5

Expected output:
- 5 content concepts
- 15 total content pieces (5 × 3 angles)
- Larger JSON file with more content

Example 3: Custom Themes
~~~~~~~~~~~~~~~~~~~~~~~~

Generate content for specific themes:

.. code-block:: bash

   ./content-gen generate --concepts "Faith in Crisis" "Christian Community" "Biblical Wisdom"

Expected output:
- 3 specific concepts (instead of defaults)
- Each with 3 angles
- Focused on your chosen themes

News Integration Examples
-------------------------

Example 4: Content with News Sentiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate content based on current trends:

.. code-block:: bash

   ./content-gen generate --use-news

Process:
1. Analyzes Christian news from last 7 days
2. Identifies trending topics and sentiment
3. Generates relevant content themes
4. Creates content tailored to current audience needs

Example 5: News Analysis Only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get current news sentiment without generating content:

.. code-block:: bash

   ./content-gen news

Sample output:

.. code-block:: json

   {
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
       "recommended_themes": [
         "Hope in Difficult Times",
         "Faith and Trust"
       ],
       "audience_concerns": [
         "Anxiety",
         "Isolation"
       ]
     }
   }

Configuration Examples
----------------------

Example 6: View Current Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   ./content-gen config --show

Example 7: Set Custom Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Set production settings
   ./content-gen config --set use_simulation false
   ./content-gen config --set default_count 5
   ./content-gen config --set model llama3.1:8b

   # Set development settings
   ./content-gen config --set use_simulation true
   ./content-gen config --set default_count 2

Advanced Examples
-----------------

Example 8: Python API Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the system programmatically:

.. code-block:: python

   from content_gen_cli import SimpleContentGenerator
   import json

   # Initialize generator
   generator = SimpleContentGenerator()

   # Generate content with news integration
   result = generator.generate_content(
       count=3,
       use_news=True
   )

   # Access generated content
   for concept in result['content']:
       print(f"Concept: {concept['concept']}")
       for angle in concept['angles']:
           print(f"  Angle: {angle['angle']}")
           print(f"  Content: {angle['content'][:100]}...")

   # Get news analysis
   news_data = generator.gather_news_sentiment()
   print(f"Overall sentiment: {news_data['sentiment_analysis']['overall_sentiment']}")

Example 9: Batch Content Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate content for multiple themes:

.. code-block:: python

   themes = [
       "Faith in Modern Life",
       "Christian Leadership",
       "Biblical Wisdom for Today",
       "Hope in Difficult Times",
       "Power of Prayer"
   ]

   generator = SimpleContentGenerator()

   for theme in themes:
       result = generator.generate_content(
           concepts=[theme],
           use_news=True
       )
       
       # Save to separate file
       timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
       filename = f"generated_content/{theme.replace(' ', '_')}_{timestamp}.json"
       
       with open(filename, 'w') as f:
           json.dump(result, f, indent=2)

Example 10: Weekly Automation Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set up automated weekly content generation:

.. code-block:: python

   # weekly_content.py
   import schedule
   import time
   from content_gen_cli import SimpleContentGenerator
   from datetime import datetime

   def generate_weekly_content():
       """Generate weekly content with news integration"""
       generator = SimpleContentGenerator()
       
       # Generate 5 concepts with news sentiment
       result = generator.generate_content(
           count=5,
           use_news=True
       )
       
       # Save with timestamp
       timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
       filename = f"generated_content/weekly_{timestamp}.json"
       
       with open(filename, 'w') as f:
           json.dump(result, f, indent=2)
       
       print(f"✅ Weekly content generated: {filename}")

   # Schedule every Sunday at 9:00 AM
   schedule.every().sunday.at("09:00").do(generate_weekly_content)

   print("📅 Weekly content scheduler started...")
   print("Content will be generated every Sunday at 9:00 AM")

   while True:
       schedule.run_pending()
       time.sleep(60)  # Check every minute

Content Analysis Examples
-------------------------

Example 11: Analyze Generated Content
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze content quality and themes:

.. code-block:: python

   import json
   from pathlib import Path
   from collections import Counter

   def analyze_content(file_path):
       """Analyze generated content for insights"""
       with open(file_path) as f:
           data = json.load(f)
       
       # Basic stats
       concepts = len(data['content'])
       total_angles = sum(len(c['angles']) for c in data['content'])
       total_words = sum(
           sum(a['word_count'] for a in c['angles'])
           for c in data['content']
       )
       
       print(f"📊 Content Analysis:")
       print(f"  Concepts: {concepts}")
       print(f"  Total Angles: {total_angles}")
       print(f"  Total Words: {total_words:,}")
       print(f"  Avg Words per Angle: {total_words // total_angles}")
       
       # Theme analysis
       themes = [c['concept'] for c in data['content']]
       print(f"\n🎯 Themes: {', '.join(themes)}")
       
       # Scripture references
       all_scriptures = []
       for concept in data['content']:
           for angle in concept['angles']:
               all_scriptures.extend(angle.get('scripture_references', []))
       
       scripture_counts = Counter(all_scriptures)
       print(f"\n📖 Top Scripture References:")
       for scripture, count in scripture_counts.most_common(5):
           print(f"  {scripture}: {count} times")

   # Analyze latest content
   content_dir = Path("generated_content")
   latest_file = max(content_dir.glob("content_*.json"))
   analyze_content(latest_file)

Example 12: Content Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare content across different time periods:

.. code-block:: python

   def compare_content_files(file1, file2):
       """Compare two content files"""
       with open(file1) as f:
           data1 = json.load(f)
       with open(file2) as f:
           data2 = json.load(f)
       
       # Compare themes
       themes1 = set(c['concept'] for c in data1['content'])
       themes2 = set(c['concept'] for c in data2['content'])
       
       print(f"📅 Content Comparison:")
       print(f"File 1 themes: {', '.join(themes1)}")
       print(f"File 2 themes: {', '.join(themes2)}")
       print(f"Common themes: {', '.join(themes1 & themes2)}")
       print(f"Unique to file 1: {', '.join(themes1 - themes2)}")
       print(f"Unique to file 2: {', '.join(themes2 - themes1)}")
       
       # Compare news integration
       if 'news_sentiment' in data1 and 'news_sentiment' in data2:
           sentiment1 = data1['news_sentiment']['sentiment_analysis']['overall_sentiment']
           sentiment2 = data2['news_sentiment']['sentiment_analysis']['overall_sentiment']
           print(f"\n📰 News Sentiment:")
           print(f"File 1: {sentiment1}")
           print(f"File 2: {sentiment2}")

Integration Examples
--------------------

Example 13: Web Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Flask web interface for content generation:

.. code-block:: python

   from flask import Flask, request, jsonify
   from content_gen_cli import SimpleContentGenerator

   app = Flask(__name__)
   generator = SimpleContentGenerator()

   @app.route('/api/generate', methods=['POST'])
   def generate_content():
       """API endpoint for content generation"""
       data = request.json
       
       count = data.get('count', 3)
       concepts = data.get('concepts', [])
       use_news = data.get('use_news', True)
       
       result = generator.generate_content(
           count=count,
           concepts=concepts,
           use_news=use_news
       )
       
       return jsonify(result)

   @app.route('/api/news', methods=['GET'])
   def get_news_sentiment():
       """API endpoint for news sentiment"""
       news_data = generator.gather_news_sentiment()
       return jsonify(news_data)

   if __name__ == '__main__':
       app.run(debug=True)

Example 14: Database Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Save generated content to database:

.. code-block:: python

   import sqlite3
   from content_gen_cli import SimpleContentGenerator
   import json

   def setup_database():
       """Create database tables"""
       conn = sqlite3.connect('content.db')
       cursor = conn.cursor()
       
       cursor.execute('''
           CREATE TABLE IF NOT EXISTS content (
               id INTEGER PRIMARY KEY,
               timestamp TEXT,
               concept TEXT,
               angle TEXT,
               content TEXT,
               scriptures TEXT,
               word_count INTEGER
           )
       ''')
       
       conn.commit()
       conn.close()

   def save_content_to_db(result):
       """Save generated content to database"""
       conn = sqlite3.connect('content.db')
       cursor = conn.cursor()
       
       timestamp = result['timestamp']
       
       for concept in result['content']:
           concept_name = concept['concept']
           
           for angle in concept['angles']:
               cursor.execute('''
                   INSERT INTO content 
                   (timestamp, concept, angle, content, scriptures, word_count)
                   VALUES (?, ?, ?, ?, ?, ?)
               ''', (
                   timestamp,
                   concept_name,
                   angle['angle'],
                   angle['content'],
                   json.dumps(angle.get('scripture_references', [])),
                   angle.get('word_count', 0)
               ))
       
       conn.commit()
       conn.close()

   # Usage
   setup_database()
   generator = SimpleContentGenerator()
   result = generator.generate_content(use_news=True)
   save_content_to_db(result)

Troubleshooting Examples
------------------------

Example 15: Debug Content Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Debug mode for troubleshooting:

.. code-block:: python

   import logging
   from content_gen_cli import SimpleContentGenerator

   # Enable debug logging
   logging.basicConfig(level=logging.DEBUG)

   def debug_content_generation():
       """Debug content generation process"""
       generator = SimpleContentGenerator()
       
       print("🔍 Debug: Checking configuration...")
       print(f"Model: {generator.config['model']}")
       print(f"Use simulation: {generator.config['use_simulation']}")
       
       print("\n🔍 Debug: Testing news gathering...")
       try:
           news_data = generator.gather_news_sentiment()
           print(f"✅ News data retrieved: {len(news_data)} keys")
       except Exception as e:
           print(f"❌ News gathering failed: {e}")
       
       print("\n🔍 Debug: Testing content generation...")
       try:
           result = generator.generate_content(count=1, use_news=False)
           print(f"✅ Content generated: {len(result['content'])} concepts")
       except Exception as e:
           print(f"❌ Content generation failed: {e}")

   debug_content_generation()

These examples demonstrate the flexibility and power of the Biblical Content Agent system. Adapt them to your specific use case and requirements.