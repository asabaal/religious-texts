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
# # Biblical Argument Analysis: Wood vs. O'Connor Debate
#
# This notebook provides a data-driven visualization and analysis of the biblical arguments presented in a theological debate between David Wood and Alex O'Connor. We'll examine their reference patterns, argument quality, and overall performance based on objective metrics.

# %%
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
import sys

# Set aesthetics for the plots - Dark Mode Theme
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

# Import religious_texts modules
repo_path = '/home/asabaal/religious-texts/religious_texts'  # Update this to your actual path
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)
from religious_texts.visualization import heatmaps, distributions
from religious_texts.debate_response import validators, comparisons, context

# %% [markdown]
# ## 1. Load Debate Data
#
# We'll start by loading the debate analysis data, which includes metrics on biblical citations and argument quality for both debaters.

# %%
# Load debate metrics from CSV file
debate_metrics = pd.read_csv('/home/asabaal/asabaal_ventures/repos/religious-texts/data/wood_oconnor_debate_metrics.csv')
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

# %% [markdown]
# ## 2. Biblical Reference Distribution
#
# Let's create a visualization of the biblical references used in the debate.

# %%
# Load bible reference distribution data
books_data = pd.read_csv('/home/asabaal/asabaal_ventures/repos/religious-texts/data/biblical_references.csv')

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

# %%
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

# %% [markdown]
# ## 3. Book Citations by Debater
#
# Let's visualize how each debater used different biblical books.

# %%
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

# %% [markdown]
# ## 4. Key Phrase Analysis
#
# Let's analyze the frequency of key biblical phrases and concepts used in the debate.

# %%
# Create dataframes for son of god references, I am statements, and worship references
son_of_god = pd.DataFrame({
    'Source': ['John', 'Mark', 'Matthew', 'Luke'],
    'Occurrences': [10, 3, 9, 6]
})

i_am_statements = pd.DataFrame({
    'Source': ['John', 'Synoptics'],
    'Occurrences': [21, 3]
})

worship_refs = pd.DataFrame({
    'Source': ['Matthew', 'Mark', 'Luke', 'John'],
    'Occurrences': [5, 2, 0, 1]
})

# Create a multiple bar chart
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Son of God references
sns.barplot(x='Source', y='Occurrences', data=son_of_god, ax=axes[0], palette='Blues_r')
axes[0].set_title('\"Son of God\" References', fontsize=14, color='white')
axes[0].set_ylabel('Number of Occurrences', fontsize=12, color='white')
axes[0].set_xlabel('Source', fontsize=12, color='white')
axes[0].set_ylim(0, 25)
axes[0].tick_params(axis='x', colors='white')
axes[0].tick_params(axis='y', colors='white')

# Plot 2: I Am statements
sns.barplot(x='Source', y='Occurrences', data=i_am_statements, ax=axes[1], palette='Oranges_r')
axes[1].set_title('\"I Am\" Statements', fontsize=14, color='white')
axes[1].set_ylabel('Number of Occurrences', fontsize=12, color='white')
axes[1].set_xlabel('Source', fontsize=12, color='white')
axes[1].set_ylim(0, 25)
axes[1].tick_params(axis='x', colors='white')
axes[1].tick_params(axis='y', colors='white')

# Plot 3: Worship references
sns.barplot(x='Source', y='Occurrences', data=worship_refs, ax=axes[2], palette='Greens_r')
axes[2].set_title('Worship (proskuneo) References', fontsize=14, color='white')
axes[2].set_ylabel('Number of Occurrences', fontsize=12, color='white')
axes[2].set_xlabel('Source', fontsize=12, color='white')
axes[2].set_ylim(0, 25)
axes[2].tick_params(axis='x', colors='white')
axes[2].tick_params(axis='y', colors='white')

plt.tight_layout()
plt.savefig('key_phrase_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 5. Argument Quality Heatmaps
#
# Let's create heatmaps to visualize each debater's performance across different metrics.

# %%
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


# %% [markdown]
# ## 6. Argument Quality Radar Chart
#
# Let's create radar charts to compare the quality of arguments between the two debaters.

# %%
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
metrics = [col for col in wood_arguments.columns if col != 'Argument' and col != 'Average Score']

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

# %% [markdown]
# ## 7. Strongest and Weakest Arguments
#
# Now, let's create bar charts to highlight each debater's strongest and weakest arguments.

# %%
# Calculate average scores for each argument
if 'Average Score' not in wood_arguments.columns:
    wood_arguments['Average Score'] = wood_arguments[[col for col in wood_arguments.columns 
                                                  if col != 'Argument']].mean(axis=1)
    
if 'Average Score' not in oconnor_arguments.columns:
    oconnor_arguments['Average Score'] = oconnor_arguments[[col for col in oconnor_arguments.columns 
                                                         if col != 'Argument']].mean(axis=1)

# Add debater column back for identification
wood_arguments['Debater'] = 'Wood'
oconnor_arguments['Debater'] = "O'Connor"

# Combine datasets
combined_args = pd.concat([wood_arguments, oconnor_arguments])

# Sort by average score
sorted_args = combined_args.sort_values('Average Score', ascending=False).reset_index(drop=True)

# Create labeled arguments with debater prefix
sorted_args['Labeled_Argument'] = sorted_args['Debater'] + ': ' + sorted_args['Argument']

# Create a color map for the bars
colors = [wood_color if debater == 'Wood' else oconnor_color for debater in sorted_args['Debater']]

# Create a horizontal bar chart with better spacing
plt.figure(figsize=(12, 10))
bar_plot = plt.barh(y=sorted_args['Labeled_Argument'], width=sorted_args['Average Score'], color=colors)

# Add value labels to the right of each bar
for i, v in enumerate(sorted_args['Average Score']):
    plt.text(v + 0.1, i, f"{v:.2f}", va='center', color='white')

# Customize plot
plt.xlabel('Average Score (0-10)', fontsize=14, color='white')
plt.ylabel('', fontsize=14)  # No need for y-label since we've combined debater and argument
plt.title('Arguments Ranked by Overall Strength', fontsize=16, color='white')
plt.xlim(0, 10)
plt.yticks(fontsize=10, color='white')  # Ensure y-tick labels are visible

# Create custom legend
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, color=wood_color, label='David Wood'),
    plt.Rectangle((0, 0), 1, 1, color=oconnor_color, label="Alex O'Connor")
]
plt.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig('ranked_arguments_combined.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 8. Overall Performance Comparison
#
# Finally, let's create a summary bar chart comparing the average performance of both debaters across each metric.

# %%
# Prepare data for comparison bar chart
# Make sure we're only using metrics that existed in the original notebook
metrics = [col for col in wood_arguments.columns if col != 'Argument' and col != 'Average Score' and col != 'Debater']

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
         f"Overall Scores:\nDavid Wood: {wood_overall:.2f}\nAlex O'Connor: {oconnor_overall:.2f}", 
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='#333333', alpha=0.8, edgecolor='white'),
         fontsize=12, color='white')

plt.tight_layout()
plt.savefig('overall_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 9. Most Disputed Area Analysis
#
# Let's create a visualization showing the most disputed area of the debate - the "Son of Man" references from Daniel 7.

# %%
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

# %% [markdown]
# ## 10. Conclusion
#
# Based on our statistical analysis and visualizations, we can draw several insights about the debate:
#
# 1. **Overall Performance**: Alex O'Connor scored slightly higher overall (7.3/10) compared to David Wood (6.95/10).
#
# 2. **Areas of Strength**:
#    - Wood's strongest argument was on "Early Christian Beliefs" with high evidence quality.
#    - O'Connor's strongest argument was the "Delegation of Divine Powers" with high evidence quality.
#
# 3. **Areas of Weakness**:
#    - Wood's "Two Powers Framework" had the lowest scholarly support score.
#    - O'Connor's "Divine Name Bearer Model" was his weakest argument area.
#
# 4. **Biblical References**:
#    - John's Gospel was the most cited source (40% of all references).
#    - Both debaters heavily cited the Gospel of John, but O'Connor cited it slightly more.
#    - Wood relied more heavily on Genesis and Philippians.
#    - O'Connor relied more on Psalms and Exodus.
#
# 5. **Key Differences**:
#    - The biggest gap in performance was in contextual accuracy, where O'Connor scored higher (7.8 vs 6.6).
#    - The most contested area was the interpretation of the "Son of Man" references in Daniel 7, with both debaters offering substantially different frameworks.
#
# The visualizations created in this notebook provide a data-driven approach to evaluating the debate, allowing for objective assessment of the biblical arguments presented by both debaters.

# %%

# %%

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
import matplotlib.patheffects as path_effects
import ipywidgets as widgets
from IPython.display import display, clear_output

# Dark mode colors
dark_bg = "#121212"
text_color = "#e0e0e0"
central_color = "#bb86fc"  # Purple
ot_color = "#03dac6"       # Teal
jesus_color = "#cf6679"    # Pink/Red
disciples_color = "#64dd17" # Light Green
enemies_color = "#ffab00"  # Amber

def create_mindmap(show_ot=True, show_jesus=True, show_disciples=True, show_enemies=True):
    plt.figure(figsize=(25, 11))
    ax = plt.gca()
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 700)
    ax.axis('off')
    
    # Dark background
    ax.add_patch(Rectangle((0, 0), 1000, 700, facecolor=dark_bg, edgecolor='none'))
    
    # Title - moved to bottom
    plt.text(500, 650, "Two Powers in Heaven: Mind Map", 
             fontsize=28, fontweight='bold', ha='center', va='center', color=text_color)
    
    # Function to add a node
    def add_node(x, y, radius, color, text1, text2=None, alpha=0.9):
        circle = Circle((x, y), radius, facecolor=color, alpha=alpha)
        ax.add_patch(circle)
        
        text_obj = plt.text(x, y-5 if text2 else y, text1, 
                 fontsize=16, fontweight='bold', ha='center', va='center', color='#121212')
        
        # Add outline to text for better visibility
        text_obj.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
        
        if text2:
            text_obj2 = plt.text(x, y+15, text2, 
                     fontsize=16, fontweight='bold', ha='center', va='center', color='#121212')
            text_obj2.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
            
    # Function to add text box
    def add_text_box(x, y, width, height, text1, text2=None, border_color='black'):
        rect = Rectangle((x, y), width, height, facecolor='#2d2d2d', edgecolor=border_color, linewidth=2, alpha=0.9, zorder=2)
        ax.add_patch(rect)
        
        plt.text(x + width/2, y + height/2 - 10 if text2 else y + height/2, text1, 
                 fontsize=14, ha='center', va='center', color=text_color)
        
        if text2:
            plt.text(x + width/2, y + height/2 + 10, text2, 
                     fontsize=14, ha='center', va='center', color=text_color)
    
    # Function to add connection
    def add_connection(x1, y1, x2, y2, color, linewidth=2):
        arrow = FancyArrowPatch((x1, y1), (x2, y2), connectionstyle="arc3,rad=0.1", 
                             arrowstyle='-', color=color, linewidth=linewidth, zorder=1)
        ax.add_patch(arrow)
    
    # Central node
    add_node(500, 350, 80, central_color, "Two Powers", "in Heaven")
    
    # Legend - moved to top left
    legend_x = 30
    legend_y = 550
    legend_spacing = 40
    
    # Add legend title
    plt.text(legend_x + 80, legend_y - 30, "Legend:", fontsize=18, fontweight='bold', color=text_color)
    
    # Add legend entries
    categories = []
    if show_ot:
        categories.append(("Old Testament Evidence", ot_color))
    if show_jesus:
        categories.append(("Jesus's Claims", jesus_color))
    if show_disciples:
        categories.append(("Disciples' Understanding", disciples_color))
    if show_enemies:
        categories.append(("Opponents' Reactions", enemies_color))
    
    for i, (text, color) in enumerate(categories):
        circle = Circle((legend_x, legend_y + i*legend_spacing), 12, facecolor=color)
        ax.add_patch(circle)
        plt.text(legend_x + 120, legend_y + i*legend_spacing, text, fontsize=16, color=text_color, va='center', ha='center')
    
    # ---- OLD TESTAMENT APPEARANCES ----
    if show_ot:
        # Main node - moved to avoid overlap
        add_node(250, 150, 60, ot_color, "Old Testament", "Appearances")
        
        # Connect to central node
        add_connection(420, 350, 310, 150, ot_color, 3)
        
        # Genesis - repositioned
        add_node(130, 70, 40, ot_color, "Genesis", alpha=0.7)
        add_connection(250, 100, 170, 80, ot_color)
        add_text_box(200, 25, 170, 60, 'the LORD out of heaven"', '"LORD rained... from', ot_color)
        
        # Zechariah - repositioned
        add_node(130, 200, 40, ot_color, "Zechariah", alpha=0.7)
        add_connection(210, 170, 170, 200, ot_color)
        add_text_box(10, 250, 170, 60, 'has sent me"', '"The LORD Almighty', ot_color)
        
        # Angel of the Lord - repositioned
        add_node(230, 300, 40, ot_color, "Angel of", "the LORD", alpha=0.7)
        add_connection(250, 210, 230, 260, ot_color)
        add_text_box(100, 350, 170, 60, 'of the LORD face to face"', '"I have seen the angel', ot_color)
    
    # ---- JESUS CLAIMS ----
    if show_jesus:
        # Main node - repositioned
        add_node(700, 250, 60, jesus_color, "Jesus", "Claims")
        
        # Connect to central node
        add_connection(580, 350, 640, 250, jesus_color, 3)
        
        # Son of God - repositioned
        add_node(630, 150, 40, jesus_color, "Son of God", alpha=0.7)
        add_connection(680, 210, 650, 180, jesus_color)
        add_text_box(580, 50, 170, 60, 'except the Son"', '"No one knows the Father', jesus_color)
        
        # Lord of David - repositioned
        add_node(800, 200, 40, jesus_color, "David", "Lord of", alpha=0.7)
        add_connection(740, 230, 760, 210, jesus_color)
        add_text_box(730, 100, 170, 60, 'sit at my right hand"', '"The Lord said to my Lord', jesus_color)
        
        # Son of Man - repositioned
        add_node(780, 350, 40, jesus_color, "Man", "Son of", alpha=0.7)
        add_connection(730, 300, 760, 330, jesus_color)
        add_text_box(830, 330, 170, 60, 'clouds of heaven"', '"Coming with the', jesus_color)
    
    # ---- DISCIPLES' UNDERSTANDING ----
    if show_disciples:
        # Main node - repositioned
        add_node(450, 500, 60, disciples_color, "Understanding", "Disciples'")
        
        # Connect to central node
        add_connection(500, 430, 450, 440, disciples_color, 3)
        
        # Philippians - repositioned
        add_node(300, 440, 40, disciples_color, "Philippians", alpha=0.7)
        add_connection(390, 470, 340, 450, disciples_color)
        add_text_box(200, 480, 170, 60, 'very nature God"', '"Who being in', disciples_color)
        
        # John 1 - repositioned
        add_node(450, 600, 40, disciples_color, "John 1", alpha=0.7)
        add_connection(450, 560, 450, 580, disciples_color)
        add_text_box(500, 580, 170, 60, 'and the Word was God"', '"The Word was with God', disciples_color)
        
        # Early Church - repositioned
        add_node(600, 440, 40, disciples_color, "Church", "Early", alpha=0.7)
        add_connection(510, 480, 560, 450, disciples_color)
        add_text_box(525, 490, 170, 60, 'the name of Jesus"', '"Everyone who calls on', disciples_color)
    
    # ---- REACTIONS OF ENEMIES ----
    if show_enemies:
        # Main node - repositioned
        add_node(700, 430, 50, enemies_color, "of Enemies", "Reactions")
        
        # Connect to central node
        add_connection(580, 350, 650, 410, enemies_color, 3)
        
        # Blasphemy - repositioned
        add_node(820, 480, 40, enemies_color, "Blasphemy", alpha=0.7)
        add_connection(750, 450, 780, 470, enemies_color)
        add_text_box(720, 525, 170, 60, 'blasphemy"', '"You have heard his', enemies_color)
    
    plt.tight_layout()
    plt.show()

# Set up the side panel of controls
from IPython.display import display

def create_ui_with_chart():
    # Create the checkboxes with larger text and more spacing
    ot_checkbox = widgets.Checkbox(
        value=True, 
        description='Old Testament Evidence', 
        layout=widgets.Layout(width='350px'),
        style={'description_width': 'initial', 'font_size': '16px'}
    )
    
    jesus_checkbox = widgets.Checkbox(
        value=True, 
        description='Jesus Claims',
        layout=widgets.Layout(width='350px'),
        style={'description_width': 'initial', 'font_size': '16px'}
    )
    
    disciples_checkbox = widgets.Checkbox(
        value=True, 
        description='Disciples\' Understanding',
        layout=widgets.Layout(width='350px'),
        style={'description_width': 'initial', 'font_size': '16px'}
    )
    
    enemies_checkbox = widgets.Checkbox(
        value=True, 
        description='Opponents\' Reactions',
        layout=widgets.Layout(width='350px'),
        style={'description_width': 'initial', 'font_size': '16px'}
    )
    
    # Create vertical box for checkboxes
    controls = widgets.VBox([
        widgets.HTML("<h3>Toggle Evidence Categories:</h3>"),
        widgets.VBox([
            ot_checkbox, 
            jesus_checkbox, 
            disciples_checkbox, 
            enemies_checkbox
        ], layout=widgets.Layout(margin='10px 0px 10px 0px'))
    ], layout=widgets.Layout(width='350px'))
    
    # Create output widget for the plot
    output = widgets.Output()
    
    # Function to update the plot
    def update_plot(*args):
        output.clear_output(wait=True)
        with output:
            create_mindmap(
                show_ot=ot_checkbox.value,
                show_jesus=jesus_checkbox.value,
                show_disciples=disciples_checkbox.value,
                show_enemies=enemies_checkbox.value
            )
    
    # Register observers
    ot_checkbox.observe(update_plot, 'value')
    jesus_checkbox.observe(update_plot, 'value')
    disciples_checkbox.observe(update_plot, 'value')
    enemies_checkbox.observe(update_plot, 'value')
    
    # Create the layout with controls on the side
    ui = widgets.HBox([
        controls,
        output
    ], layout=widgets.Layout(width='100%'))
    
    # Initial display
    display(ui)
    update_plot()

# Call the function to create the UI and chart
create_ui_with_chart()

# %%

# %%

# %%

# %%
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.tokenize import sent_tokenize
import PyPDF2
import langdetect
from langdetect import DetectorFactory
import pycld2 as cld2

# Set plotting style
plt.style.use('dark_background')
sns.set_context("notebook", font_scale=1.2)
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

# Ensure consistent language detection results
DetectorFactory.seed = 0

class BibleTextAnalyzer:
    def __init__(self, bible_pdf_path, language='english'):
        """
        Initialize the analyzer with path to a Bible PDF file
        
        Parameters:
        -----------
        bible_pdf_path: str
            Path to the Bible PDF file
        language: str
            Language of the Bible text ('english', 'hebrew', or 'greek')
        """
        self.language = language.lower()
        self.pdf_path = bible_pdf_path
        self.text = ""
        self.book_content = {}
        
        # Language codes for detection
        self.lang_codes = {
            'english': ['en'],
            'hebrew': ['he', 'iw'],  # Both ISO codes for Hebrew
            'greek': ['el', 'grc']   # Modern and Ancient Greek
        }
        
        # Load and preprocess the text
        self._load_and_preprocess_pdf()
        self.parse_books()
        
    def _load_and_preprocess_pdf(self):
        """Load text from PDF and filter by language"""
        print(f"Loading PDF: {self.pdf_path}")
        
        try:
            pdf_reader = PyPDF2.PdfReader(self.pdf_path)
            num_pages = len(pdf_reader.pages)
            print(f"Total pages: {num_pages}")
            
            # Extract text from each page and filter by language
            all_text = []
            for i in range(num_pages):
                if i % 10 == 0:
                    print(f"Processing page {i+1}/{num_pages}...")
                
                page = pdf_reader.pages[i]
                page_text = page.extract_text()
                
                # Split page into paragraphs for language filtering
                paragraphs = page_text.split('\n\n')
                valid_paragraphs = []
                
                for para in paragraphs:
                    if len(para.strip()) < 20:  # Skip very short paragraphs
                        continue
                    
                    if self._is_target_language(para):
                        valid_paragraphs.append(para)
                
                all_text.extend(valid_paragraphs)
            
            self.text = '\n\n'.join(all_text)
            print(f"Extracted and filtered text length: {len(self.text)} characters")
            
        except Exception as e:
            print(f"Error loading PDF: {e}")
            self.text = ""
    
    def _is_target_language(self, text):
        """Check if text is in the target language"""
        text = text.strip()
        if not text:
            return False
            
        # For short texts, use character set heuristics
        if len(text) < 50:
            if self.language == 'hebrew' and any('\u0590' <= c <= '\u05FF' for c in text):
                return True
            if self.language == 'greek' and any('\u0370' <= c <= '\u03FF' for c in text):
                return True
            if self.language == 'english' and text.isascii():
                return True
            return False
        
        # For longer texts, use language detection
        try:
            # First try pycld2 which handles Hebrew and Greek better
            _, _, detected_langs = cld2.detect(text, returnVectors=True)
            if detected_langs[0][1] in self.lang_codes[self.language]:
                return True
                
            # Fallback to langdetect
            detected = langdetect.detect(text)
            return detected in self.lang_codes[self.language]
            
        except (langdetect.lang_detect_exception.LangDetectException, cld2.error, Exception):
            # Fallback to character set heuristics if detection fails
            if self.language == 'hebrew' and any('\u0590' <= c <= '\u05FF' for c in text):
                return True
            if self.language == 'greek' and any('\u0370' <= c <= '\u03FF' for c in text):
                return True
            if self.language == 'english' and text.isascii():
                return True
            return False
            
    def parse_books(self):
        """Parse the Bible text into individual books"""
        print("Parsing books...")
        
        # Patterns for different languages
        book_patterns = {
            'english': r'(?:The )?Book of ([A-Za-z\s]+)(.*?)(?=(?:The )?Book of|\Z)',
            'hebrew': r'ספר ([א-ת\s]+)(.*?)(?=ספר|\Z)',  # Adjust for Hebrew Bible format
            'greek': r'(?:Βίβλος|Βιβλίον) ([Α-Ωα-ω\s]+)(.*?)(?=(?:Βίβλος|Βιβλίον)|\Z)'  # Adjust for Greek Bible format
        }
        
        # Use appropriate pattern based on language
        pattern = book_patterns.get(self.language, book_patterns['english'])
        
        # Add fallback patterns for each language
        fallback_patterns = []
        if self.language == 'english':
            # Common Bible book names in English
            books = ["Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges", "Ruth", 
                    "1 Samuel", "2 Samuel", "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah", 
                    "Esther", "Job", "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon", "Isaiah", "Jeremiah", 
                    "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah", "Nahum", 
                    "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi", "Matthew", "Mark", "Luke", "John", 
                    "Acts", "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians", "Philippians", 
                    "Colossians", "1 Thessalonians", "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", "Philemon", 
                    "Hebrews", "James", "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude", "Revelation"]
            for book in books:
                fallback_patterns.append(f"(?:{book})\s+Chapter\s+([0-9]+)(.*?)(?=Chapter|$)")
                
        elif self.language == 'hebrew':
            # Add Hebrew Bible book names and patterns here
            hebrew_books = [
                "בראשית", "שמות", "ויקרא", "במדבר", "דברים", "יהושע", "שופטים", "שמואל א", 
                "שמואל ב", "מלכים א", "מלכים ב", "ישעיהו", "ירמיהו", "יחזקאל", "הושע", 
                "יואל", "עמוס", "עובדיה", "יונה", "מיכה", "נחום", "חבקוק", "צפניה", "חגי", 
                "זכריה", "מלאכי", "תהלים", "משלי", "איוב", "שיר השירים", "רות", "איכה", 
                "קהלת", "אסתר", "דניאל", "עזרא", "נחמיה", "דברי הימים א", "דברי הימים ב"
            ]
            for book in hebrew_books:
                fallback_patterns.append(f"(?:{book})\s+פרק\s+([0-9א-ת]+)(.*?)(?=פרק|$)")
                
        elif self.language == 'greek':
            # Add Greek Bible book names and patterns here
            greek_books = [
                "Γένεσις", "Έξοδος", "Λευιτικόν", "Αριθμοί", "Δευτερονόμιον", 
                "Ιησούς του Ναυή", "Κριταί", "Ρουθ", "Βασιλειών Α", "Βασιλειών Β", 
                "Βασιλειών Γ", "Βασιλειών Δ", "Παραλειπομένων Α", "Παραλειπομένων Β", 
                "Έσδρας", "Νεεμίας", "Εσθήρ", "Ιώβ", "Ψαλμοί", "Παροιμίαι", 
                "Εκκλησιαστής", "Άσμα Ασμάτων", "Ησαΐας", "Ιερεμίας", "Θρήνοι", 
                "Ιεζεκιήλ", "Δανιήλ", "Ωσηέ", "Ιωήλ", "Αμώς", "Αβδιού", "Ιωνάς", 
                "Μιχαίας", "Ναούμ", "Αββακούμ", "Σοφονίας", "Αγγαίος", "Ζαχαρίας", "Μαλαχίας",
                "Κατά Ματθαίον", "Κατά Μάρκον", "Κατά Λουκάν", "Κατά Ιωάννην", "Πράξεις",
                "Προς Ρωμαίους", "Προς Κορινθίους Α", "Προς Κορινθίους Β", "Προς Γαλάτας",
                "Προς Εφεσίους", "Προς Φιλιππησίους", "Προς Κολοσσαείς", "Προς Θεσσαλονικείς Α",
                "Προς Θεσσαλονικείς Β", "Προς Τιμόθεον Α", "Προς Τιμόθεον Β", "Προς Τίτον",
                "Προς Φιλήμονα", "Προς Εβραίους", "Ιακώβου", "Πέτρου Α", "Πέτρου Β",
                "Ιωάννου Α", "Ιωάννου Β", "Ιωάννου Γ", "Ιούδα", "Αποκάλυψις"
            ]
            for book in greek_books:
                fallback_patterns.append(f"(?:{book})\s+κεφάλαιο\s+([0-9α-ω]+)(.*?)(?=κεφάλαιο|$)")
        
        # Try the main pattern first
        books = re.findall(pattern, self.text, re.DOTALL)
        
        # If few or no books found, try fallback patterns
        if len(books) < 5:
            print("Main pattern didn't find enough books, trying fallback patterns...")
            for fallback_pattern in fallback_patterns:
                chapters = re.findall(fallback_pattern, self.text, re.DOTALL)
                if chapters:
                    book_name = fallback_pattern.split("(?:")[1].split(")")[0]
                    if book_name not in self.book_content:
                        self.book_content[book_name] = ""
                    
                    for chapter_num, chapter_content in chapters:
                        self.book_content[book_name] += f"Chapter {chapter_num}\n{chapter_content}\n"
        else:
            # Process books found with the main pattern
            for book_name, content in books:
                self.book_content[book_name.strip()] = content.strip()
        
        print(f"Found {len(self.book_content)} books")
        if len(self.book_content) > 0:
            print(f"Books found: {', '.join(list(self.book_content.keys())[:5])}{'...' if len(self.book_content) > 5 else ''}")
        else:
            print("Warning: No books were parsed!")

    # The rest of your methods remain largely unchanged...
    def find_angel_of_lord_instances(self):
        """Find all instances of 'angel of the Lord' in the text"""
        angel_patterns = {
            'english': r'\b(?:the )?angel of the Lord\b',
            'hebrew': r'מלאך יהוה',  # Angel of the LORD in Hebrew
            'greek': r'(?:ὁ )?ἄγγελος (?:τοῦ )?κυρίου'  # Angel of the Lord in Greek
        }
        
        pattern = angel_patterns.get(self.language, angel_patterns['english'])
        angel_instances = []
        
        for book, content in self.book_content.items():
            # Split content into sentences for better context
            sentences = sent_tokenize(content)
            
            for i, sentence in enumerate(sentences):
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    # Get context (surrounding sentences for analysis)
                    start_idx = max(0, i-1)
                    end_idx = min(len(sentences), i+2)
                    context = " ".join(sentences[start_idx:end_idx])
                    
                    # Determine if this instance refers to God
                    is_god_reference = self._is_god_reference(context)
                    
                    angel_instances.append({
                        'book': book,
                        'context': context,
                        'is_god_reference': is_god_reference
                    })
                    
        return angel_instances
    
    def _is_god_reference(self, context):
        """
        Determine if an instance of 'angel of the Lord' in the given context refers to God
        
        This is a simplified approach - in practice, you would want to use more sophisticated NLP
        or develop a comprehensive set of rules based on theological research.
        """
        # Adjust patterns based on language
        god_indicators = []
        
        if self.language == 'english':
            god_indicators = [
                # Direct equation
                r'angel of the Lord.*\bsaid\b.*\bI am\b.*\bGod\b',
                r'angel of the Lord.*\bspoke as\b.*\bGod\b',
                # Authority indicators
                r'angel of the Lord.*\bcommanded\b',
                r'angel of the Lord.*\bswore\b',
                r'angel of the Lord.*\bpromised\b',
                # Identity switching
                r'angel of the Lord[^.]*\..*\bLord\b.*\bsaid\b',
                r'angel of the Lord[^.]*then the Lord',
                # Worship context
                r'worshipped.*angel of the Lord',
                r'bowed.*angel of the Lord',
                r'angel of the Lord.*\bname\b',
            ]
        elif self.language == 'hebrew':
            god_indicators = [
                r'מלאך יהוה.*אמר.*אני.*אלהים',
                r'מלאך יהוה.*צוה',
                r'מלאך יהוה.*נשבע',
                r'מלאך יהוה.*הבטיח',
                r'מלאך יהוה.*יהוה.*אמר',
                r'השתחוה.*מלאך יהוה',
                r'מלאך יהוה.*שם',
            ]
        elif self.language == 'greek':
            god_indicators = [
                r'ἄγγελος κυρίου.*εἶπεν.*ἐγώ εἰμι.*θεός',
                r'ἄγγελος κυρίου.*ἐντέλλομαι',
                r'ἄγγελος κυρίου.*ὤμοσεν',
                r'ἄγγελος κυρίου.*ὑπέσχετο',
                r'ἄγγελος κυρίου.*κύριος.*εἶπεν',
                r'προσεκύνησεν.*ἄγγελος κυρίου',
                r'ἄγγελος κυρίου.*ὄνομα',
            ]
        
        for indicator in god_indicators:
            if re.search(indicator, context, re.IGNORECASE):
                return True
                
        # Additional check: if the sentence has "but" after Angel of the Lord, likely differentiation
        if self.language == 'english' and re.search(r'angel of the Lord\b.*\bbut\b.*\bGod\b', context, re.IGNORECASE):
            return False
        elif self.language == 'hebrew' and re.search(r'מלאך יהוה.*אבל.*אלהים', context, re.IGNORECASE):
            return False
        elif self.language == 'greek' and re.search(r'ἄγγελος κυρίου.*ἀλλὰ.*θεός', context, re.IGNORECASE):
            return False
            
        # More advanced analysis would go here
        return False  # Default: conservative approach, mark as not referring to God
    
    def _is_god_reference(self, context):
        """
        Determine if an instance of 'angel of the Lord' in the given context refers to God
        
        This is a simplified approach - in practice, you would want to use more sophisticated NLP
        or develop a comprehensive set of rules based on theological research.
        """
        # Key indicators where angel of the Lord appears to be equated with God
        god_indicators = [
            # Direct equation
            r'angel of the Lord.*\bsaid\b.*\bI am\b.*\bGod\b',
            r'angel of the Lord.*\bspoke as\b.*\bGod\b',
            # Authority indicators
            r'angel of the Lord.*\bcommanded\b',
            r'angel of the Lord.*\bswore\b',
            r'angel of the Lord.*\bpromised\b',
            # Identity switching
            r'angel of the Lord[^.]*\..*\bLord\b.*\bsaid\b',
            r'angel of the Lord[^.]*then the Lord',
            # Worship context
            r'worshipped.*angel of the Lord',
            r'bowed.*angel of the Lord',
            r'angel of the Lord.*\bname\b',
        ]
        
        for indicator in god_indicators:
            if re.search(indicator, context, re.IGNORECASE):
                return True
                
        # Additional check: if the sentence has "but" after Angel of the Lord, likely differentiation
        if re.search(r'angel of the Lord\b.*\bbut\b.*\bGod\b', context, re.IGNORECASE):
            return False
            
        # More advanced analysis would go here
        return False  # Default: conservative approach, mark as not referring to God
    
    def find_divine_titles(self):
        """Find instances of various divine titles in the text"""
        divine_titles = {
            'english': [
                r'\bLord God\b', 
                r'\bLord GOD\b',
                r'\bLord of hosts\b',
                r'\bGod Almighty\b',
                r'\bThe Almighty\b',
                r'\bI AM\b',
                r'\bI am that I am\b',
                r'\bThe Most High\b',
                r'\bKing of kings\b',
                r'\bAlpha and Omega\b'
            ],
            'hebrew': [
                r'יהוה אלהים',
                r'אדני יהוה',
                r'יהוה צבאות',
                r'אל שדי',
                r'עליון',
                r'אהיה אשר אהיה'
            ],
            'greek': [
                r'κύριος ὁ θεός',
                r'κύριος παντοκράτωρ',
                r'ὁ ὢν',
                r'θεὸς ὁ ὕψιστος'
            ]
        }
        
        patterns = divine_titles.get(self.language, divine_titles['english'])
        title_counts = {pattern: 0 for pattern in patterns}
        title_books = {pattern: Counter() for pattern in patterns}
        
        for book, content in self.book_content.items():
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                title_counts[pattern] += len(matches)
                title_books[pattern][book] += len(matches)
                
        return title_counts, title_books
    
    def compare_with_angel_references(self, angel_instances):
        """Compare 'angel of the Lord' references with other divine titles"""
        god_as_angel_count = sum(1 for instance in angel_instances if instance['is_god_reference'])
        not_god_as_angel_count = len(angel_instances) - god_as_angel_count
        
        title_counts, _ = self.find_divine_titles()
        
        # Add angel counts to title counts for comparison
        all_counts = title_counts.copy()
        all_counts['Angel of the Lord (as God)'] = god_as_angel_count
        all_counts['Angel of the Lord (not as God)'] = not_god_as_angel_count
        
        return all_counts
        
    def analyze_books_with_angel_references(self, angel_instances):
        """Analyze which books have 'angel of the Lord' references and their distribution"""
        book_counts = Counter()
        god_ref_by_book = Counter()
        not_god_ref_by_book = Counter()
        
        for instance in angel_instances:
            book = instance['book']
            book_counts[book] += 1
            if instance['is_god_reference']:
                god_ref_by_book[book] += 1
            else:
                not_god_ref_by_book[book] += 1
                
        return book_counts, god_ref_by_book, not_god_ref_by_book
    
    def visualize_angel_references(self, angel_instances):
        """Visualize the analysis of 'angel of the Lord' references"""
        # Count godlike references vs. non-godlike
        god_refs = sum(1 for instance in angel_instances if instance['is_god_reference'])
        non_god_refs = len(angel_instances) - god_refs
        
        # Check if we have any data to visualize
        if len(angel_instances) == 0:
            print("No 'angel of the Lord' references found to visualize")
            return
            
        try:
            # Create a pie chart - remove shadow parameter which is causing issues
            plt.figure(figsize=(10, 6))
            plt.pie([god_refs, non_god_refs], 
                    labels=['References to God', 'Not References to God'], 
                    autopct='%1.1f%%',
                    colors=['#1f77b4', '#ff7f0e'],
                    explode=(0.1, 0),
                    startangle=90)  # Removed shadow=True which causes the error
            plt.title('Angel of the Lord References: God vs. Not God', fontsize=16)
            plt.tight_layout()
            plt.savefig('angel_references_pie.png', dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure to prevent display issues
        except Exception as e:
            print(f"Error creating pie chart: {e}")
        
        # Book distribution
        book_counts, god_ref_by_book, not_god_ref_by_book = self.analyze_books_with_angel_references(angel_instances)
        
        # Only attempt visualization if we have books
        top_books = [book for book, _ in book_counts.most_common(10) if book_counts[book] > 0]
        if not top_books:
            print("No books with 'angel of the Lord' references to visualize")
            return
            
        try:
            # Create stacked bar chart
            plt.figure(figsize=(12, 8))
            
            god_refs_data = [god_ref_by_book[book] for book in top_books]
            not_god_refs_data = [not_god_ref_by_book[book] for book in top_books]
            
            bar_width = 0.8
            x = np.arange(len(top_books))
            
            plt.bar(x, god_refs_data, bar_width, label='References to God', color='#1f77b4')
            plt.bar(x, not_god_refs_data, bar_width, bottom=god_refs_data, label='Not References to God', color='#ff7f0e')
            
            plt.xlabel('Book of the Bible', fontsize=14)
            plt.ylabel('Number of References', fontsize=14)
            plt.title('Distribution of "Angel of the Lord" References by Book', fontsize=16)
            plt.xticks(x, top_books, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig('angel_references_by_book.png', dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure
        except Exception as e:
            print(f"Error creating bar chart: {e}")    

    def visualize_divine_title_comparison(self, all_counts):
        """Visualize comparison between 'angel of the Lord' and other divine titles"""
        # Check if we have any data
        if not all_counts:
            print("No divine titles found to visualize")
            return
        
        try:
            # Sort by counts
            sorted_titles = dict(sorted(all_counts.items(), key=lambda x: x[1], reverse=True))
            
            # Format title labels
            formatted_titles = {re.sub(r'\\b|[()]', '', title).replace('\\', ''): count 
                            for title, count in sorted_titles.items()}
            
            # Create bar chart
            plt.figure(figsize=(14, 8))
            bars = plt.bar(range(len(formatted_titles)), list(formatted_titles.values()), 
                        color=sns.color_palette("Blues_r", len(formatted_titles)))
            
            # Highlight angel references
            for i, (title, _) in enumerate(formatted_titles.items()):
                if 'Angel of the Lord' in title:
                    bars[i].set_color('#ff7f0e' if 'not as God' in title else '#1f77b4')
            
            plt.xlabel('Divine Title', fontsize=14)
            plt.ylabel('Number of Occurrences', fontsize=14)
            plt.title('Comparison of Divine Titles in the Bible', fontsize=16)
            plt.xticks(range(len(formatted_titles)), list(formatted_titles.keys()), rotation=45, ha='right')
            
            # Add counts above bars
            for i, v in enumerate(formatted_titles.values()):
                plt.text(i, v + 0.5, str(v), ha='center', fontsize=10, color='white')
                
            plt.tight_layout()
            plt.savefig('divine_title_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure
        except Exception as e:
            print(f"Error creating divine title comparison chart: {e}")
        
    def analyze_and_visualize(self):
        """Run full analysis and visualization pipeline"""
        print("Finding angel of the Lord instances...")
        angel_instances = self.find_angel_of_lord_instances()
        print(f"Found {len(angel_instances)} instances of 'angel of the Lord'")
        
        print("\nAnalyzing divine titles...")
        all_counts = self.compare_with_angel_references(angel_instances)
        
        print("\nVisualizing results...")
        self.visualize_angel_references(angel_instances)
        self.visualize_divine_title_comparison(all_counts)
        
        return angel_instances, all_counts



# Change these paths to your downloaded Bible files
english_bible_path = "/home/asabaal/asabaal_ventures/asabaals_academic_adventures/data_science_of_religious_texts/data/nrsv.pdf"
hebrew_bible_path = "/home/asabaal/asabaal_ventures/asabaals_academic_adventures/data_science_of_religious_texts/data/masoretic.pdf"
greek_bible_path = "/home/asabaal/asabaal_ventures/asabaals_academic_adventures/data_science_of_religious_texts/greek.pdf"

print("Analyzing English Bible text...")
english_analyzer = BibleTextAnalyzer(english_bible_path, 'english')
english_instances, english_counts = english_analyzer.analyze_and_visualize()

# Export the analysis results to CSV for further exploration
instances_df = pd.DataFrame(english_instances)
instances_df.to_csv('angel_lord_instances.csv', index=False)

counts_df = pd.DataFrame([english_counts]).T.reset_index()
counts_df.columns = ['Title', 'Count']
counts_df.to_csv('divine_title_counts.csv', index=False)

print("\nAnalysis complete! Results saved to CSV files and visualizations.")

# Optional: run analysis on Hebrew and Greek texts
if os.path.exists(hebrew_bible_path):
    print("\nAnalyzing Hebrew Bible text...")
    hebrew_analyzer = BibleTextAnalyzer(hebrew_bible_path, 'hebrew')
    hebrew_analyzer.analyze_and_visualize()

if os.path.exists(greek_bible_path):
    print("\nAnalyzing Greek Bible text...")
    greek_analyzer = BibleTextAnalyzer(greek_bible_path, 'greek')
    greek_analyzer.analyze_and_visualize()

# %%
os.listdir()

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Process the divine_title_counts.csv
def visualize_divine_titles_csv():
    df = pd.read_csv('divine_title_counts.csv')
    
    # Sort by count value
    df = df.sort_values('Count', ascending=False)
    
    # Create a horizontal bar chart (easier to read long titles)
    plt.figure(figsize=(12, 8))
    bars = plt.barh(df['Title'], df['Count'], color=sns.color_palette("Blues_r", len(df)))
    
    # Highlight angel references if they exist
    for i, title in enumerate(df['Title']):
        if isinstance(title, str) and 'Angel of the Lord' in title:
            bars[i].set_color('#ff7f0e' if 'not as God' in title else '#1f77b4')
    
    plt.xlabel('Number of Occurrences', fontsize=14)
    plt.ylabel('Divine Title', fontsize=14)
    plt.title('Divine Titles in the Bible by Frequency', fontsize=16)
    
    # Add count labels inside bars
    for i, v in enumerate(df['Count']):
        plt.text(v/2, i, str(v), va='center', fontsize=10, color='white', weight='bold')
    
    plt.tight_layout()
    plt.savefig('divine_titles_csv_viz.png', dpi=300)
    plt.show()

# 2. Process the angel_lord_instances.csv
def visualize_angel_instances_csv():
    df = pd.read_csv('angel_lord_instances.csv')
    
    # Count by book and is_god_reference
    book_counts = df.groupby(['book', 'is_god_reference']).size().unstack(fill_value=0)
    
    # If the columns are missing, create them
    if True not in book_counts.columns:
        book_counts[True] = 0
    if False not in book_counts.columns:
        book_counts[False] = 0
    
    # Rename for clarity
    book_counts = book_counts.rename(columns={True: 'References to God', False: 'Not References to God'})
    
    # Create stacked bar chart
    book_counts.plot(kind='bar', stacked=True, figsize=(12, 6), 
                    color=['#1f77b4', '#ff7f0e'])
    
    plt.xlabel('Book of the Bible', fontsize=14)
    plt.ylabel('Number of References', fontsize=14)
    plt.title('Distribution of "Angel of the Lord" References by Book', fontsize=16)
    plt.legend(title='Reference Type')
    plt.xticks(rotation=45, ha='right')
    
    # Add total counts as text
    for i, book in enumerate(book_counts.index):
        god_refs = book_counts.loc[book, 'References to God']
        not_god_refs = book_counts.loc[book, 'Not References to God']
        total = god_refs + not_god_refs
        plt.text(i, total + 0.1, f'Total: {total}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('angel_instances_by_book.png', dpi=300)
    plt.show()
    
    # Also create a wordcloud of contexts (optional)
    try:
        from wordcloud import WordCloud
        
        all_contexts = ' '.join(df['context'].tolist())
        
        plt.figure(figsize=(12, 8))
        wordcloud = WordCloud(width=800, height=400, background_color='black',
                            max_words=100, colormap='Blues').generate(all_contexts)
        
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Words Surrounding "Angel of the Lord" References', fontsize=16)
        plt.tight_layout()
        plt.savefig('angel_contexts_wordcloud.png', dpi=300)
        plt.show()
    except:
        print("WordCloud visualization requires the wordcloud package")


# %%
visualize_divine_titles_csv()

# %%
visualize_angel_instances_csv()

# %%

# %%

# %%
english_analyzer = BibleTextAnalyzer(english_bible_path, 'english')

# %%
english_analyzer.text[-500:]

# %%
pd.Series(english_analyzer.text.split("\n")).to_csv("nrsv.txt", index=False, header=False)

# %% [markdown]
# # The "Only Permits Blessing" Framework
#
# ## Biblical Foundation
#
# The "Only Permits Blessing" framework is rooted in several key scriptural passages:
#
# Galatians 6:1: "Brothers and sisters, if someone is caught in a sin, you who live by the Spirit should restore that person gently. But watch yourselves, or you also may be tempted."
#
# Ephesians 4:29: "Let no corrupting talk come out of your mouths, but only such as is good for building up, as fits the occasion, that it may give grace to those who hear."
#
# James 3:9-10: "With the tongue we praise our Lord and Father, and with it we curse human beings, who have been made in God's likeness. Out of the same mouth come praise and cursing. My brothers and sisters, this should not be."
#
# Colossians 4:6: "Let your speech always be gracious, seasoned with salt, so that you may know how you ought to answer each person."
#
# This biblical foundation emphasizes that our speech should exclusively build up and edify others, not tear down. The framework transforms theological discourse by prioritizing restoration over condemnation, especially when engaging differences in biblical interpretation, doctrine, and ethics.
#
# ## Core Principles
#
# ### 1. Gentle Restoration vs. Heresy Labeling
#
# Gentle Restoration: Approaches theological differences with questions, context, and an assumption of the other's sincere faith journey
# Heresy Labeling: Dismisses theological differences by attaching stigmatizing labels that end dialogue
# ### 2. Criteria for Different Approaches
#
# Scripture provides distinct approaches for different situations:
#
# Gentle Restoration (Galatians 6:1) is appropriate for:
# Sincere seekers who may misunderstand
# Those with different interpretations but good intentions
# Communities wrestling with complex theological questions
#
# Strong Condemnation (seen in places like Galatians 1:8-9) is reserved for:
# Exploitation of vulnerable people
# Deliberate deception for personal gain
# Teaching that causes active harm
# Those creating division to serve their own interests
#
# ### 3. Practical Elements of Blessing-Based Engagement
#
# The framework employs specific practices:
#
# Affirm shared values before addressing differences
# Ask questions that invite deeper reflection rather than making accusations
# Add context rather than simply contradicting
# Accept that growth takes time
# Affirm relationship regardless of agreement
# ## Contrasting Approaches
#
# | Heresy-Labeling Approach | Blessing-Based Approach |
# |--------------------------|-------------------------|
# | "That view is heretical and dangerous to the faith" | "I see how you're trying to honor Scripture; I understand this passage differently" |
# | "No true Christian would believe that" | "Christians throughout history have wrestled with this question in different ways" |
# | "You're twisting Scripture to fit your agenda" | "I'm curious how you reconcile this passage with other relevant texts" |
# | "That's liberal theology that undermines truth" | "I wonder what theological values are most important to you in this interpretation?" |
#
# ## Application for Queer Christian Contexts
#
# This framework is particularly powerful for queer Christians who:
#
# Experience being labeled "heretical" for affirming theology
# Navigate complex relationships with traditional faith communities
# Seek to engage rather than dismiss traditional perspectives
# Model Christ-like engagement even amid significant theological difference
# When addressing harmful theology, the framework helps distinguish between:
# - The person (who deserves dignity and gentle engagement)
# - The idea (which may require firm but respectful critique)
# - The system (which may need structural transformation)
#
# ## Historical Perspective
#
# Throughout church history, theological positions have moved from "heresy" to "orthodoxy" through respectful engagement:
# - Gentile inclusion without circumcision
# - Vernacular Bible translations
# - Recognition of scientific discoveries
# - Abolition of slavery
#
# This historical pattern suggests humility about pronouncing final judgments and openness to how the Spirit might be working through marginalized voices today.
#
# ## Transformative Impact
#
# When we choose blessing-based engagement:
# 1. Defenses lower, creating openness to new perspectives
# 2. Relationships continue, allowing ongoing growth
# 3. Nuance emerges beyond simplistic positions
# 4. Learning becomes mutual rather than one-directional
# 5. The Spirit has room to work through relationship
#
# ## Conclusion
#
# The "Only Permits Blessing" framework doesn't avoid addressing error or harmful theology, but transforms how we engage difference. By following the biblical model of gentle restoration, we create space for genuine transformation while maintaining both truth and love.
#
# As modeled by Jesus himself (who was labeled a blasphemer by religious authorities), this approach recognizes that faithful discipleship sometimes means being misunderstood while continuing to extend blessing.

# %% [markdown]
#

# %% [markdown]
#
