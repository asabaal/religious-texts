Biblical Content Agent Documentation
=====================================

Welcome to the comprehensive documentation for the Biblical Content Agent - an AI-powered system for generating biblical content with real-time sentiment analysis.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   userguide
   api
   examples
   developer
   testing

Overview
--------

The Biblical Content Agent is a sophisticated content generation system that combines:

* **AI-powered content creation** using Llama 3.1 8B via Ollama
* **Real-time news sentiment analysis** from Christian sources
* **Multi-angle narrative generation** (3 angles per concept)
* **Unified CLI interface** with simple commands
* **Weekly automation** for content planning

Key Features
------------

* **Simple CLI**: ``content-gen generate`` for quick content creation
* **News Integration**: RSS feeds from major Christian news sources
* **Quality Evaluation**: Automated scoring for engagement, clarity, authenticity
* **Audience Targeting**: Demographic-specific messaging
* **Scalable System**: Generate 3-5 concepts with 3 angles each (15 total)

Quick Example
-------------

.. code-block:: bash

   # Generate 3 content concepts with news sentiment
   ./content-gen generate

   # Generate 5 concepts without news
   ./content-gen generate --count 5 --no-news

   # Get current news sentiment analysis
   ./content-gen news

   # Weekly content generation (5 concepts, 15 angles)
   ./weekly_content.py

Documentation Structure
-----------------------

* :doc:`quickstart` - Get started in 5 minutes
* :doc:`userguide` - Comprehensive user manual
* :doc:`api` - Complete API documentation
* :doc:`examples` - Runnable examples and demos
* :doc:`developer` - Developer contribution guide
* :doc:`testing` - Testing framework and guidelines

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`