# This file contains the updated code blocks for the debate_analysis_notebook.ipynb
# Copy these sections into your notebook to implement the visualization improvements

# ===== SECTION 1: IMPORTS AND SETUP =====
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
import os

# Hard code the path to data directory
DATA_DIR = '/home/asabaal/asabaal_ventures/repos/religious-texts/data/'

# Import religious_texts modules
from religious_texts.visualization import heatmaps, distributions
from religious_texts.debate_response import validators, comparisons, context

# Set dark theme aesthetics for the plots
plt.style.use('dark_background')
sns.set_context("notebook", font_scale=1.2)
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.facecolor'] = '#121212'
plt.rcParams['figure.facecolor'] = '#121212'
plt.rcParams['savefig.facecolor'] = '#121212'

# Define colors for the debaters
wood_color = '#1f77b4'   # Blue
oconnor_color = '#ff7f0e'  # Orange

# ===== SECTION 2: LOAD DATA =====
# Load debate metrics from CSV file
debate_metrics = pd.read_csv(os.path.join(DATA_DIR, 'wood_oconnor_debate_metrics.csv'))
print(f"Loaded {len(debate_metrics)} arguments")

# Separate by debater
wood_arguments = debate_metrics[debate_metrics['Debater'] == 'Wood'].drop('Debater', axis=1)
oconnor_arguments = debate_metrics[debate_metrics['Debater'] == 'O\'Connor'].drop('Debater', axis=1)

# Calculate average scores for each debater
wood_avg = wood_arguments.mean(numeric_only=True).to_dict()
oconnor_avg = oconnor_arguments.mean(numeric_only=True).to_dict()

# Overall scores
wood_overall = sum(wood_avg.values()) / len(wood_avg)
oconnor_overall = sum(oconnor_avg.values()) / len(oconnor_avg)

print(f"David Wood's overall score: {wood_overall:.2f}")
print(f"Alex O'Connor's overall score: {oconnor_overall:.2f}")

# ===== SECTION 3: BIBLICAL REFERENCE DISTRIBUTION =====
# Load bible reference distribution data
books_data = pd.read_csv(os.path.join(DATA_DIR, 'biblical_references.csv'))

# Calculate total references by source
books_data['Total'] = books_data['Wood_Citations'] + books_data['OConnor_Citations']
books_data = books_data.sort_values('Total', ascending=False).reset_index(drop=True)

# Group into categories for pie chart
top_books = books_data.head(4)['Book'].tolist()
other_books = books_data.iloc[4:]['Total'].sum()

# Create pie chart data
pie_data = pd.DataFrame({
    'Source': top_books + ['Other Books'],
    'References': list(books_data.head(4)['Total']) + [other_books]
})

# Calculate percentages
total_refs = pie_data['References'].sum()
pie_data['Percentage'] = (pie_data['References'] / total_refs) * 100

# Create a pie chart
plt.figure(figsize=(10, 6))
plt.pie(pie_data['Percentage'], labels=pie_data['Source'], autopct='%1.1f%%',
        startangle=90, shadow=False, explode=(0.1, 0, 0, 0, 0),
        colors=sns.color_palette('Blues_r'))
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.title('Distribution of Biblical References by Source', fontsize=14, color='white')
plt.tight_layout()
plt.savefig('reference_distribution_pie.png', dpi=300, bbox_inches='tight')
plt.show()

# ===== SECTION 4: REFERENCE TYPES BAR CHART =====
# Create reference type data
reference_types = pd.DataFrame({
    'Type': ['Direct quotations', 'Paraphrased references', 'Allusions without citation'],
    'Percentage': [60, 30, 10]
})

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Type', y='Percentage', data=reference_types, palette='Blues_r')
plt.title('Types of Biblical Citations Used', fontsize=14, color='white')
plt.ylabel('Percentage of Total References', fontsize=12, color='white')
plt.xlabel('Reference Type', fontsize=12, color='white')
plt.ylim(0, 100)

# Add percentage labels on bars
for i, p in enumerate(ax.patches):
    ax.annotate(f"{p.get_height():.0f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=12, color='white')

plt.tight_layout()
plt.savefig('reference_types_bar.png', dpi=300, bbox_inches='tight')
plt.show()

# ===== SECTION 5: SIDE-BY-SIDE BOOK CITATIONS =====
# Use top 10 most referenced books
top10_books = books_data.head(10).copy()

# Set up the figure
plt.figure(figsize=(12, 8))

# Set width and positions for bars
bar_width = 0.35
x = np.arange(len(top10_books))

# Plot side-by-side bars
plt.bar(x - bar_width/2, top10_books['Wood_Citations'], bar_width, label='David Wood', color=wood_color)
plt.bar(x + bar_width/2, top10_books['OConnor_Citations'], bar_width, label='Alex O\'Connor', color=oconnor_color)

# Set x-axis ticks and labels
plt.xticks(x, top10_books['Book'], rotation=45, ha='right', color='white')

# Add total counts above each book
for i, book in enumerate(top10_books['Book']):
    total = top10_books['Wood_Citations'].iloc[i] + top10_books['OConnor_Citations'].iloc[i]
    plt.text(i, top10_books[['Wood_Citations', 'OConnor_Citations']].max(axis=1).iloc[i] + 0.3,
             f"Total: {total}", ha='center', color='white')

plt.xlabel('Biblical Book', fontsize=14, color='white')
plt.ylabel('Number of Citations', fontsize=14, color='white')
plt.title('Distribution of Biblical Book Citations by Debater', fontsize=16, color='white')
plt.legend()
plt.tight_layout()
plt.savefig('book_citation_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# ===== SECTION 6: HEATMAPS WITH IMPROVED COLOR SCALE =====
# Create a combined heatmap with both debaters
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Create a custom colormap from red to green
red_green_cmap = sns.diverging_palette(10, 120, as_cmap=True)

# David Wood's heatmap
wood_heatmap_data = wood_arguments.set_index('Argument')
sns.heatmap(wood_heatmap_data, annot=True, cmap=red_green_cmap, vmin=0, vmax=10, 
            cbar_kws={'label': 'Score (0-10)'}, ax=axes[0])
axes[0].set_title("David Wood's Argument Metrics", fontsize=14, color='white')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right', color='white')
axes[0].set_yticklabels(axes[0].get_yticklabels(), color='white')

# Alex O'Connor's heatmap
oconnor_heatmap_data = oconnor_arguments.set_index('Argument')
sns.heatmap(oconnor_heatmap_data, annot=True, cmap=red_green_cmap, vmin=0, vmax=10, 
            cbar_kws={'label': 'Score (0-10)'}, ax=axes[1])
axes[1].set_title("Alex O'Connor's Argument Metrics", fontsize=14, color='white')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right', color='white')
axes[1].set_yticklabels(axes[1].get_yticklabels(), color='white')

# Improve colorbar labels
for ax in axes:
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors='white')
    cbar.set_label('Score (0-10)', color='white')

plt.tight_layout()
plt.savefig('argument_metrics_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# ===== SECTION 7: FIXED RADAR CHART =====
# Function to create radar chart axes manually
def radar_chart(fig, categories, values1, values2, title, label1='Label 1', label2='Label 2',
               color1='blue', color2='orange'):
    # Number of variables
    N = len(categories)
    
    # Compute angle for each category
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    # Make the plot circular by repeating the first angle
    angles += angles[:1]
    
    # Extend values to form a complete polygon
    values1 = values1.tolist()
    values1 += values1[:1]
    values2 = values2.tolist()
    values2 += values2[:1]
    
    # Plot initialization
    ax = fig.add_subplot(111, polar=True)
    
    # Plot values for first dataset
    ax.plot(angles, values1, 'o-', linewidth=2, label=label1, color=color1)
    ax.fill(angles, values1, alpha=0.25, color=color1)
    
    # Plot values for second dataset
    ax.plot(angles, values2, 'o-', linewidth=2, label=label2, color=color2)
    ax.fill(angles, values2, alpha=0.25, color=color2)
    
    # Set labels and beautify
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_ylim(0, 10)
    ax.set_facecolor('#121212')  # Dark background
    
    # Add title and legend
    ax.set_title(title, size=16, color='white', y=1.1)
    ax.legend(loc='upper right', facecolor='#121212')
    
    # Customize tick labels
    ax.tick_params(axis='both', colors='white')
    
    return ax

# Prepare data for radar chart - extract column names excluding 'Argument'
metrics = [col for col in wood_arguments.columns if col != 'Argument']

# Create data arrays
wood_values = np.array([wood_avg[metric] for metric in metrics])
oconnor_values = np.array([oconnor_avg[metric] for metric in metrics])

# Create the radar chart
fig = plt.figure(figsize=(8, 8))
radar_chart(fig, metrics, wood_values, oconnor_values, 'Argument Quality Comparison',
           label1='David Wood', label2="Alex O'Connor", color1=wood_color, color2=oconnor_color)

plt.tight_layout()
plt.savefig('argument_quality_radar.png', dpi=300, bbox_inches='tight')
plt.show()

# ===== SECTION 8: COMBINED STRONGEST/WEAKEST ARGUMENTS CHART =====
# Calculate average scores for each argument
wood_arguments['Average Score'] = wood_arguments[metrics].mean(axis=1)
oconnor_arguments['Average Score'] = oconnor_arguments[metrics].mean(axis=1)

# Add debater column back for identification
wood_arguments['Debater'] = 'Wood'
oconnor_arguments['Debater'] = "O'Connor"

# Combine datasets
combined_args = pd.concat([wood_arguments, oconnor_arguments])

# Sort by average score
sorted_args = combined_args.sort_values('Average Score', ascending=False).reset_index(drop=True)

# Create a color map for the bars
colors = [wood_color if debater == 'Wood' else oconnor_color for debater in sorted_args['Debater']]

# Create a horizontal bar chart
plt.figure(figsize=(12, 10))
bar_plot = plt.barh(sorted_args['Argument'], sorted_args['Average Score'], color=colors)

# Add value labels to the right of each bar
for i, v in enumerate(sorted_args['Average Score']):
    plt.text(v + 0.1, i, f"{v:.2f}", va='center', color='white')

# Add debater labels to the left of each bar
for i, (arg, debater) in enumerate(zip(sorted_args['Argument'], sorted_args['Debater'])):
    plt.text(-0.5, i, f"{debater}", va='center', ha='right', color='white')

# Create custom legend
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, color=wood_color, label='David Wood'),
    plt.Rectangle((0, 0), 1, 1, color=oconnor_color, label="Alex O'Connor")
]
plt.legend(handles=legend_elements, loc='lower right')

# Customize plot
plt.xlabel('Average Score (0-10)', fontsize=14, color='white')
plt.title('Arguments Ranked by Overall Strength', fontsize=16, color='white')
plt.xlim(0, 10)
plt.tight_layout()
plt.savefig('ranked_arguments_combined.png', dpi=300, bbox_inches='tight')
plt.show()

# ===== SECTION 9: IMPROVED OVERALL PERFORMANCE COMPARISON =====
# Prepare data for comparison bar chart
comparison_data = pd.DataFrame({
    'Metric': metrics,
    'David Wood': [wood_avg[metric] for metric in metrics],
    'Alex O\'Connor': [oconnor_avg[metric] for metric in metrics]
})

# Melt the dataframe for easier plotting
melted_data = pd.melt(comparison_data, id_vars=['Metric'], var_name='Debater', value_name='Score')

# Create a grouped bar chart
plt.figure(figsize=(12, 6))
ax = sns.barplot(x='Metric', y='Score', hue='Debater', data=melted_data, 
                palette=[wood_color, oconnor_color])

# Add value labels on bars
for i, bar in enumerate(ax.patches):
    ax.text(bar.get_x() + bar.get_width()/2., 
            bar.get_height() + 0.1,
            f"{bar.get_height():.1f}", 
            ha='center', color='white')

# Customize plot
plt.title('Average Performance by Metric', fontsize=16, color='white')
plt.xlabel('Evaluation Metric', fontsize=14, color='white')
plt.ylabel('Average Score (0-10)', fontsize=14, color='white')
plt.ylim(0, 10)
plt.legend(title='Debater', title_fontsize=12)

# Add overall score text with improved visibility
plt.text(0.01, 0.05, 
         f"Overall Scores:\\nDavid Wood: {wood_overall:.2f}\\nAlex O'Connor: {oconnor_overall:.2f}", 
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='#333333', alpha=0.8, edgecolor='white'),
         fontsize=12, color='white')

plt.tight_layout()
plt.savefig('overall_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ===== SECTION 10: CONTESTED INTERPRETATION CHART WITH DARK THEME =====
# Create data for the disputed interpretation
disputed_area = pd.DataFrame({
    'Interpretation': ['Divine Figure', 'Human Representative', 'Collective Symbol', 'Messianic But Not Divine'],
    'Wood Support': [8, 3, 2, 4],
    'O\'Connor Support': [3, 7, 6, 8]
})

# Create a grouped bar chart for the disputed area
plt.figure(figsize=(10, 6))
x = np.arange(len(disputed_area['Interpretation']))
width = 0.35

plt.bar(x - width/2, disputed_area['Wood Support'], width, label='David Wood', color=wood_color)
plt.bar(x + width/2, disputed_area['O\'Connor Support'], width, label='Alex O\'Connor', color=oconnor_color)

plt.xlabel('Interpretive Framework', fontsize=14, color='white')
plt.ylabel('Support Level (0-10)', fontsize=14, color='white')
plt.title('Contested Interpretations of "Son of Man" in Daniel 7', fontsize=16, color='white')
plt.xticks(x, disputed_area['Interpretation'], rotation=45, ha='right', color='white')
plt.yticks(color='white')
plt.legend(title='Debater')
plt.ylim(0, 10)

plt.tight_layout()
plt.savefig('contested_interpretation.png', dpi=300, bbox_inches='tight')
plt.show()
