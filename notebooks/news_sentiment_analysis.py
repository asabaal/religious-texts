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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # News Sentiment Analysis Demo
#
# This notebook demonstrates the news gathering and sentiment analysis capabilities of the Biblical Content Agent system.

# %%
import sys
sys.path.append('..')
from news_gathering import NewsGatherer, get_simulated_news_data
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

print('News sentiment analysis demo ready')

# %%
# Get simulated news data
news_data = get_simulated_news_data()
print(f'Analysis from {news_data["period_days"]} days')
print(f'Overall sentiment: {news_data["sentiment_analysis"]["overall_sentiment"]}')
