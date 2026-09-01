User Guide
==========

This comprehensive guide covers all features of the Biblical Content Agent system.

Overview
--------

The Biblical Content Agent is an AI-powered system that generates Christian content
with optional integration of current news sentiment analysis. It can create multiple
content options from different angles, making it ideal for churches, ministries,
and Christian content creators.

Features
--------

- **Multi-angle Content Generation**: Generate 3 different angles for each concept
- **News Sentiment Integration**: Analyze current trends and audience concerns
- **Automated Scheduling**: Set up weekly content generation
- **Configurable Output**: Customize number of options, angles, and themes
- **Christian News Sources**: Monitor relevant Christian news outlets
- **Sentiment Analysis**: Understand audience emotional state and needs

Command Line Interface
----------------------

The main interface is the ``content-gen`` command-line tool.

Basic Commands
~~~~~~~~~~~~~~

Generate Content
^^^^^^^^^^^^^^^

.. code-block:: bash

   ./content-gen generate [OPTIONS]

Options:
- ``--count, -n``: Number of content concepts to generate (default: 3)
- ``--concepts``: Custom concepts/themes (comma-separated)
- ``--use-news``: Integrate news sentiment analysis
- ``--no-news``: Skip news sentiment analysis

Examples:

.. code-block:: bash

   # Generate 3 default concepts
   ./content-gen generate

   # Generate 5 concepts with news integration
   ./content-gen generate --count 5 --use-news

   # Generate custom concepts
   ./content-gen generate --concepts "Faith in Crisis" "Christian Hope"

News Analysis
^^^^^^^^^^^^^

.. code-block:: bash

   ./content-gen news [OPTIONS]

Options:
- ``--days``: Number of days to analyze (default: 7)

Example:

.. code-block:: bash

   # Analyze last 7 days
   ./content-gen news

   # Analyze last 30 days
   ./content-gen news --days 30

System Status
^^^^^^^^^^^^^

.. code-block:: bash

   ./content-gen status

Shows:
- Current configuration
- System component status
- Recent activity
- Cache information

Configuration Management
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   ./content-gen config [OPTIONS]

Options:
- ``--show``: Display current configuration
- ``--set KEY VALUE``: Set configuration value
- ``--reset``: Reset to defaults

Configuration Options:
- ``model``: AI model to use (default: llama3.1:8b)
- ``ollama_url``: Ollama API URL (default: http://localhost:11434)
- ``default_count``: Default number of concepts (default: 3)
- ``angles_per_concept``: Angles per concept (default: 3)
- ``auto_news``: Automatically use news (default: true)
- ``use_simulation``: Use simulated data (default: true)

Content Generation Process
--------------------------

1. **Theme Selection**: Choose from default themes or provide custom ones
2. **News Analysis** (optional): Analyze current Christian news and sentiment
3. **Content Generation**: Create content for each theme from multiple angles
4. **Output**: Save structured JSON with all generated content

Content Structure
~~~~~~~~~~~~~~~~~

Each generated content piece includes:

.. code-block:: json

   {
     "concept": "Theme Name",
     "angles": [
       {
         "angle": "Angle Description",
         "content": "Generated biblical content...",
         "scripture_references": ["John 3:16", "Romans 8:28"],
         "word_count": 500,
         "tone": "encouraging",
         "target_audience": "general"
       }
     ]
   }

News Sentiment Integration
--------------------------

When ``--use-news`` is enabled, the system:

1. **Gathers News**: Collects articles from Christian RSS feeds
2. **Analyzes Sentiment**: Calculates positive/negative sentiment
3. **Identifies Trends**: Finds trending topics and concerns
4. **Generates Themes**: Creates relevant content themes
5. **Adapts Content**: Tailors content to current audience needs

News Sources
~~~~~~~~~~~~

- Christianity Today
- Relevant Magazine
- ChurchLeaders
- Bible Study Tools

Sentiment Metrics
~~~~~~~~~~~~~~~~~

- **Overall Sentiment**: positive, negative, neutral, mixed
- **Trending Topics**: Top 5 topics with frequency and sentiment
- **Audience Concerns**: Identified emotional and spiritual needs
- **Recommended Themes**: AI-generated content themes

Automation
----------

Weekly Content Generation
~~~~~~~~~~~~~~~~~~~~~~~~~

Set up automated weekly content generation:

.. code-block:: bash

   python weekly_content.py

Features:
- Runs every week at configurable time
- Generates 3-5 content concepts
- Integrates news sentiment analysis
- Saves with timestamp in output directory
- Sends notifications (configurable)

Scheduling Options
~~~~~~~~~~~~~~~~~~

Edit ``weekly_content.py`` to customize:

.. code-block:: python

   # Schedule for every Sunday at 9:00 AM
   schedule.every().sunday.at("09:00").do(generate_weekly_content)

   # Schedule for every Monday at 2:00 PM
   schedule.every().monday.at("14:00").do(generate_weekly_content)

Output Management
-----------------

File Organization
~~~~~~~~~~~~~~~~~

Generated content is organized as:

.. code-block:: text

   generated_content/
   ├── content_20241007_120000.json
   ├── content_20241014_120000.json
   └── ...

File Naming
~~~~~~~~~~~

Files use timestamp format: ``content_YYYYMMDD_HHMMSS.json`

Batch Processing
~~~~~~~~~~~~~~~~

Process multiple files:

.. code-block:: python

   import json
   from pathlib import Path

   # Load all content files
   content_dir = Path("generated_content")
   all_content = []

   for file_path in content_dir.glob("content_*.json"):
       with open(file_path) as f:
           data = json.load(f)
           all_content.append(data)

Advanced Usage
--------------

Custom Themes
~~~~~~~~~~~~~

Create a themes file:

.. code-block:: json

   {
     "custom_themes": [
       "Faith in Modern Life",
       "Christian Leadership",
       "Biblical Wisdom for Today",
       "Hope in Difficult Times",
       "Power of Prayer"
     ]
   }

Use with CLI:

.. code-block:: bash

   ./content-gen generate --concepts "$(cat themes.json | jq -r '.custom_thems[] | @sh' | tr '\n' ',')"

API Integration
~~~~~~~~~~~~~~~

Use the Python API directly:

.. code-block:: python

   from content_gen_cli import SimpleContentGenerator

   generator = SimpleContentGenerator()

   # Generate with news
   result = generator.generate_content(
       count=3,
       use_news=True
   )

   # Generate custom concepts
   result = generator.generate_content(
       concepts=["Faith", "Hope", "Love"],
       use_news=False
   )

   # Get news analysis
   news_data = generator.gather_news_sentiment()

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Content Generation Fails**
- Check Ollama is running: ``ollama list``
- Verify model is available: ``ollama pull llama3.1:8b``
- Check API URL in configuration

**News Analysis Fails**
- Check internet connection
- Verify RSS feeds are accessible
- Try with ``use_simulation=true``

**Permission Errors**
- Ensure ``content-gen`` is executable: ``chmod +x content-gen``
- Check write permissions in output directory

Debug Mode
~~~~~~~~~~

Enable debug output:

.. code-block:: bash

   ./content-gen generate --debug

Check logs in:
- ``~/.content-gen/`` directory
- Console output with verbose information

Configuration Reset
~~~~~~~~~~~~~~~~~~

Reset to defaults:

.. code-block:: bash

   ./content-gen config --reset

Or manually delete config:

.. code-block:: bash

   rm ~/.content-gen/config.json