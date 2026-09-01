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
# # Worship Term Analysis for Theological Debates
#
# This notebook demonstrates how to use the worship language analysis modules in the `religious-texts` library to analyze worship terminology in theological debates, particularly focusing on terms like "proskuneo" vs. "latreo" which were discussed in the David Wood vs. Alex O'Connor debate.

# %% [markdown]
# ## Setup and Imports

# %%
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to the path so we can import the religious_texts module
sys.path.append('..')

# Import modules
from religious_texts.data_acquisition import loaders
from religious_texts.visualization import heatmaps, distributions
from religious_texts.theological_analysis import worship
from religious_texts.specialized import worship_analysis
from religious_texts.specialized import debate_worship_analysis

# Set plot style
plt.style.use('ggplot')
sns.set_context('notebook')

# %% [markdown]
# ## Loading Biblical Text Data
#
# First, we'll load a Bible text file. The library supports various formats, but for this example, we'll use the King James Version (KJV).

# %%
# Path to the Bible text file - adjust as needed
kjv_path = '../data/kjv.txt'

# Load the Bible text
bible = loaders.load_text(kjv_path)

# Print a sample to verify loading worked correctly
print(f"Loaded {len(bible)} books")
print("Sample from John 3:16:")
print(bible.get('John', {}).get(3, {}).get(16, 'Verse not found'))

# %% [markdown]
# ## Basic Worship Language Analysis
#
# Let's start with a basic analysis of worship language distribution across biblical books.

# %%
# Analyze worship language by book
worship_by_book = worship.worship_language_analysis(bible, unit='book')

# Display the top 10 books with highest worship density
worship_by_book.head(10)[['book', 'total_worship_terms', 'total_words', 'worship_density']]

# %% [markdown]
# ## Visualizing Worship Language Distribution
#
# Now let's create a visualization of the worship language distribution.

# %%
# Prepare data for visualization
top_books = worship_by_book.head(15)[['book', 'worship_density']]

# Create a bar chart
plt.figure(figsize=(12, 8))
sns.barplot(x='worship_density', y='book', data=top_books)
plt.title('Top 15 Biblical Books by Worship Language Density')
plt.xlabel('Worship Terms per 1000 Words')
plt.ylabel('Book')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Analyzing Specific Worship Terms: Proskuneo vs. Latreo
#
# Now let's focus on the specific worship terms that are central to theological debates, particularly "proskuneo" (often translated as "worship") and "latreo" (often translated as "serve").

# %%
# Extract proskuneo terms from WORSHIP_TERMS dictionary
proskuneo_terms = worship_analysis.WORSHIP_TERMS['proskuneo']
latreo_terms = worship_analysis.WORSHIP_TERMS['latreo']

# Print the terms
print("Proskuneo (worship) terms:")
print(proskuneo_terms)
print("\nLatreo (service) terms:")
print(latreo_terms)

# %%
# Analyze distribution of these specific terms
term_distribution = worship.worship_term_distribution(
    bible, 
    terms=proskuneo_terms + latreo_terms
)

# Display top books with these terms
term_distribution.head(10)[['book', 'total_term_count', 'term_density']]

# %% [markdown]
# ## Specialized Analysis of Proskuneo vs. Latreo
#
# Now let's use the specialized worship analysis module to compare proskuneo and latreo usage in detail.

# %%
# Define New Testament books for analysis
nt_books = [
    "Matthew", "Mark", "Luke", "John", "Acts", "Romans", "1 Corinthians", 
    "2 Corinthians", "Galatians", "Ephesians", "Philippians", "Colossians", 
    "1 Thessalonians", "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", 
    "Philemon", "Hebrews", "James", "1 Peter", "2 Peter", "1 John", "2 John", 
    "3 John", "Jude", "Revelation"
]

# Compare proskuneo vs. latreo in the New Testament
comparison = worship_analysis.compare_proskuneo_latreo(bible, books=nt_books)

# Display summary of the comparison
print(f"Total proskuneo instances: {comparison['proskuneo_instances']}")
print(f"Total latreo instances: {comparison['latreo_instances']}")
print("\nProskuneo recipients:")
print(comparison['proskuneo_recipients']['counts'])
print("\nLatreo recipients:")
print(comparison['latreo_recipients']['counts'])

# %% [markdown]
# ## Visualizing Recipient Distribution
#
# Let's visualize the distribution of recipients for both worship terms.

# %%
# Extract recipient counts
proskuneo_recipients = comparison['proskuneo_recipients']['counts']
latreo_recipients = comparison['latreo_recipients']['counts']

# Combine into DataFrame for plotting
recipient_df = pd.DataFrame({
    'Proskuneo': pd.Series(proskuneo_recipients),
    'Latreo': pd.Series(latreo_recipients)
}).fillna(0).astype(int)

# Create grouped bar chart
recipient_df.plot(kind='bar', figsize=(10, 6))
plt.title('Recipient Distribution: Proskuneo vs. Latreo')
plt.xlabel('Recipient Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Identifying Instances of Jesus Worship
#
# One key aspect of the theological debate is whether Jesus receives worship (proskuneo) in a way that indicates his divinity.

# %%
# Analyze instances where Jesus receives proskuneo
jesus_worship = worship_analysis.analyze_worship_recipients(
    bible,
    worship_categories=['proskuneo'],
    recipient='jesus_terms',
    books=nt_books
)

# Display the instances
if not jesus_worship.empty:
    jesus_worship[['reference', 'text', 'worship_term', 'recipient_term']].head(10)
else:
    print("No instances found where Jesus receives proskuneo.")

# %% [markdown]
# ## Analyzing Specific Debate Claims
#
# Now let's use the debate worship analysis module to analyze specific claims made in the David Wood vs. Alex O'Connor debate.

# %%
# List available debate claims
print("Available debate claims for analysis:")
for key, claim in debate_worship_analysis.DEBATE_CLAIMS.items():
    print(f"- {key}: {claim['claim']}")

# %%
# Analyze the proskuneo exclusivity claim
exclusivity_analysis = debate_worship_analysis.analyze_debate_claim(
    bible, 
    "proskuneo_exclusivity"
)

# Display the claim analysis
print(f"Claim: {exclusivity_analysis['claim']}")
print(f"Counter: {exclusivity_analysis['counter']}")
print(f"\nKey passages analyzed: {exclusivity_analysis['total_passages']}")
print("\nSupport statistics:")
print(exclusivity_analysis['support_statistics'])

# %% [markdown]
# ## Examining Specific Passages in Context
#
# Let's analyze specific passages that are often cited in debates about worship terminology.

# %%
# Define key passages from the debate
key_passages = [
    "Matthew 2:11",  # Magi worship infant Jesus
    "Matthew 4:10",  # "Worship the Lord your God and serve him only"
    "Matthew 28:17", # Disciples worship resurrected Jesus
    "John 9:38",     # Healed blind man worships Jesus
    "Hebrews 1:6",   # Angels worship the Son
    "Revelation 19:10", # Angel refuses worship
    "Revelation 22:8-9" # Another angel refuses worship
]

# Analyze these passages in context
context_analysis = debate_worship_analysis.analyze_debate_passages_context(
    bible,
    key_passages,
    context_verses=2
)

# Print summary statistics
print(f"Total passages analyzed: {context_analysis['total_passages']}")
print("\nWorship category counts:")
for category, count in context_analysis['worship_category_counts'].items():
    print(f"- {category}: {count}")

print("\nRecipient category counts:")
for category, count in context_analysis['recipient_category_counts'].items():
    print(f"- {category}: {count}")

print("\nTheological theme counts:")
for theme, count in context_analysis['theme_counts'].items():
    print(f"- {theme}: {count}")

# %% [markdown]
# ## Creating an Evidence Matrix for the Debate
#
# Finally, let's create an evidence matrix that evaluates the biblical support for each debate claim.

# %%
# Create the evidence matrix
evidence_df = debate_worship_analysis.create_worship_evidence_matrix(bible)

# Display the matrix
evidence_df[['claim', 'evidence_category', 'support_count', 'counter_count', 'neutral_count', 'total_passages']]

# %% [markdown]
# ## Visualizing the Evidence Matrix
#
# Let's create a heatmap visualization of the evidence strength for each claim.

# %%
# Create a numeric mapping of evidence categories for visualization
evidence_map = {
    'strong_support': 2,
    'moderate_support': 1,
    'neutral': 0,
    'moderate_counter': -1,
    'strong_counter': -2
}

# Map categories to numeric values
evidence_df['evidence_value'] = evidence_df['evidence_category'].map(evidence_map)

# Prepare data for heatmap
heatmap_data = evidence_df.pivot_table(
    values='evidence_value',
    index='claim',
    columns=['claim_key'],
    aggfunc='first'
)

# Create the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    heatmap_data.T, 
    cmap=sns.diverging_palette(10, 240, n=5), 
    center=0,
    annot=True, 
    fmt=".0f",
    linewidths=.5
)
plt.title('Evidence Matrix for Theological Debate Claims')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Comprehensive Analysis of the Wood vs. O'Connor Debate
#
# Now, let's bring everything together to analyze the specific debate claims from the Wood vs. O'Connor debate regarding worship terminology and Jesus' divinity.

# %%
# Define the specific passages cited in the debate
wood_passages = [
    "Matthew 28:17", # Disciples worship resurrected Jesus
    "John 9:38",     # Healed blind man worships Jesus
    "Hebrews 1:6",   # Angels worship the Son
    "Matthew 4:10"   # "Worship the Lord your God and serve him only"
]

oconnor_passages = [
    "Matthew 2:11",  # Magi worship infant Jesus
    "Revelation 19:10", # Angel refuses worship
    "Revelation 22:8-9", # Another angel refuses worship
    "Romans 1:25"    # Worship and serve the creature
]

# Perform comprehensive analysis of debate worship terms
debate_analysis = debate_worship_analysis.analyze_debate_worship_terms(
    bible,
    proskuneo_passages=wood_passages + oconnor_passages
)

# Display the analysis results
print("DEBATE ANALYSIS: PROSKUNEO VS. LATREO IN THE WOOD-O'CONNOR DEBATE\n")

print("PROSKUNEO (WORSHIP) ANALYSIS:")
print(f"Total passages analyzed: {debate_analysis['proskuneo_analysis']['total_passages']}")
print("Recipient distribution:")
for recipient, count in debate_analysis['proskuneo_analysis']['recipient_distribution'].items():
    print(f"- {recipient}: {count}")

print("\nCOMPARISON WITH COMPLETE NEW TESTAMENT:")
print(f"Total proskuneo instances in NT: {debate_analysis['comparison']['proskuneo_total']}")
print(f"Total latreo instances in NT: {debate_analysis['comparison']['latreo_total']}")
print(f"Instances where Jesus receives proskuneo: {debate_analysis['comparison']['proskuneo_jesus_count']}")
print(f"Instances where Jesus receives latreo: {debate_analysis['comparison']['latreo_jesus_count']}")

print("\nDEBATE SIGNIFICANCE:")
for claim, value in debate_analysis['debate_significance'].items():
    print(f"- {claim}: {'Supported' if value else 'Not supported'}")

# %% [markdown]
# ## Conclusion
#
# This notebook has demonstrated how to use the `religious-texts` library to analyze worship terminology in biblical texts, with a specific focus on the terms discussed in the David Wood vs. Alex O'Connor debate. 
#
# The analysis shows that:
#
# 1. There are clear patterns in how "proskuneo" and "latreo" terms are used in biblical texts.
# 2. Jesus receives "proskuneo" (worship) in multiple New Testament passages.
# 3. The evidence for claims about the exclusivity of these terms is mixed.
# 4. Contextual analysis provides important insights into how these terms are used in relation to different recipients.
#
# This type of analysis provides an objective, data-driven approach to theological debates about worship terminology and divine attribution in biblical texts.
