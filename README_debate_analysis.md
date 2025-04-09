# Biblical Debate Analysis Visualization

This project provides a comprehensive statistical analysis and visualization of biblical arguments presented in theological debates. The accompanying Jupyter notebook (`debate_analysis_notebook.ipynb`) demonstrates how to use the `religious-texts` library to create informative visualizations of debate performance metrics.

## Overview

The notebook uses data from a theological debate between David Wood and Alex O'Connor to demonstrate:

1. Quantitative evaluation of debate performance
2. Distribution analysis of biblical references
3. Argument strength visualization
4. Comparative analysis of debate tactics

## Features

- **Performance radar charts**: Compare debater performance across multiple metrics
- **Heatmap visualizations**: Show argument strength by category
- **Reference distribution analysis**: Analyze which biblical books and passages were cited
- **Key phrase tracking**: Identify and visualize important theological terms
- **Statistical significance testing**: Evaluate the validity of theological claims

## Requirements

- Python 3.8+
- Jupyter Notebook
- Dependencies:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - religious_texts (local package)

## Usage

1. Clone this repository
2. Install the required dependencies
3. Open and run the Jupyter notebook:
```bash
jupyter notebook debate_analysis_notebook.ipynb
```

## Data Format

The notebook uses structured data about the debate, including:

- Arguments presented by each debater
- Performance metrics for each argument (Evidence Quality, Contextual Accuracy, Scholarly Support, Logical Coherence)
- Biblical references cited
- Distribution of references across biblical books

## Customizing for Other Debates

To analyze a different debate:

1. Replace the argument evaluation data
2. Update the biblical reference counts
3. Modify the key phrases section to reflect the theological concepts relevant to the new debate
4. Run all cells to generate updated visualizations

## Example Visualizations

The notebook generates several visualizations:

1. **Radar chart** comparing overall argument quality
2. **Pie chart** showing distribution of biblical references
3. **Bar charts** displaying strongest and weakest arguments
4. **Heatmaps** visualizing performance metrics
5. **Stacked bar charts** showing biblical book citation patterns
6. **Comparison charts** for contested theological interpretations

## License

MIT
