Quick Start Guide
=================

This guide will help you get up and running with the Biblical Content Agent quickly.

Installation
------------

1. Clone the repository:
   .. code-block:: bash
   
      git clone <repository-url>
      cd religious-texts

2. Install dependencies:
   .. code-block:: bash
   
      pip install -r requirements.txt

3. Set up the CLI:
   .. code-block:: bash
   
      chmod +x content-gen

Basic Usage
-----------

Generating Content
~~~~~~~~~~~~~~~~~

Generate 3 content options with default themes:
.. code-block:: bash

   ./content-gen generate

Generate 5 content options with news sentiment integration:
.. code-block:: bash

   ./content-gen generate --count 5 --use-news

Generate content with custom themes:
.. code-block:: bash

   ./content-gen generate --concepts "God's Love" "Christian Hope" "Biblical Wisdom"

News Sentiment Analysis
~~~~~~~~~~~~~~~~~~~~~~~

Get current news sentiment and trending topics:
.. code-block:: bash

   ./content-gen news

System Status
~~~~~~~~~~~~~

Check system status and configuration:
.. code-block:: bash

   ./content-gen status

Configuration
~~~~~~~~~~~~~

View current configuration:
.. code-block:: bash

   ./content-gen config --show

Set configuration values:
.. code-block:: bash

   ./content-gen config --set default_count 5
   ./content-gen config --set use_simulation false

Weekly Automation
~~~~~~~~~~~~~~~~~

Set up automated weekly content generation:
.. code-block:: bash

   python weekly_content.py

Examples
--------

Content Generation with News Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example generates content based on current trending topics:

.. code-block:: bash

   ./content-gen generate --count 3 --use-news

Output:
- Generates 3 content options
- Each option has 3 different angles
- Themes are based on current news sentiment
- Results saved to ``generated_content/`` directory

Custom Theme Generation
~~~~~~~~~~~~~~~~~~~~~~~

Generate content for specific themes:

.. code-block:: bash

   ./content-gen generate --concepts "Faith in Crisis" "Christian Community" --count 2

Output:
- 2 content options for each theme
- 3 angles per option
- Total: 6 content pieces

Configuration Examples
~~~~~~~~~~~~~~~~~~~~~~~

Set up for production use:
.. code-block:: bash

   ./content-gen config --set use_simulation false
   ./content-gen config --set model llama3.1:8b
   ./content-gen config --set default_count 5
   ./content-gen config --set auto_news true

Set up for development/testing:
.. code-block:: bash

   ./content-gen config --set use_simulation true
   ./content-gen config --set default_count 2
   ./content-gen config --set angles_per_concept 2

Output Files
------------

Generated content is saved in JSON format with the following structure:

.. code-block:: json

   {
     "timestamp": "2024-10-07T12:00:00",
     "config": {
       "model": "llama3.1:8b",
       "use_news": true
     },
     "content": [
       {
         "concept": "Theme Name",
         "angles": [
           {
             "angle": "Angle Description",
             "content": "Generated content...",
             "scripture_references": ["John 3:16"],
             "word_count": 500
           }
         ]
       }
     ]
   }

Next Steps
----------

- Read the :doc:`userguide` for detailed usage instructions
- Check the :doc:`api` for complete API documentation
- See the :doc:`examples` for advanced use cases
- Read the :doc:`developer` guide for contributing