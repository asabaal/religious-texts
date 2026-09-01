#!/usr/bin/env python3
"""
Convert content agent JSON results to HTML using existing demo styling.
"""

import json
import os
import glob
from datetime import datetime
from pathlib import Path


def load_latest_content_json():
    """Load the most recent content agent JSON file."""
    json_files = glob.glob("generated_content/content_*.json")
    if not json_files:
        raise FileNotFoundError("No content JSON files found in generated_content/")
    
    latest_file = max(json_files, key=os.path.getctime)
    print(f"Loading latest content file: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)


def load_multi_angle_content():
    """Load all multi-angle content files."""
    content_files = glob.glob("multi_angle_content_*.json")
    content_data = {}
    
    for file_path in content_files:
        # Skip the summary file
        if 'summary' in file_path:
            continue
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            if 'concept' in data:
                content_data[data['concept']] = data
    
    return content_data


def get_sentiment_color(sentiment):
    """Get color for sentiment value."""
    if sentiment > 0.5:
        return "#00ff88"  # Green for positive
    elif sentiment > 0:
        return "#ffaa00"  # Orange for neutral-positive
    elif sentiment > -0.5:
        return "#ff6600"  # Red-orange for neutral-negative
    else:
        return "#ff3366"  # Red for negative


def generate_html(content_json, multi_angle_data):
    """Generate HTML report from content data."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Content Agent Report - {content_json['timestamp'][:10]}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}

        header {{
            text-align: center;
            padding: 40px 0;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            margin-bottom: 30px;
            backdrop-filter: blur(10px);
        }}

        h1 {{
            font-size: 2.5em;
            background: linear-gradient(45deg, #00d4ff, #7b2ff7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}

        .subtitle {{
            font-size: 1.2em;
            color: #a0a0a0;
            margin-bottom: 20px;
        }}

        .timestamp {{
            color: #7b2ff7;
            font-size: 0.9em;
        }}

        .section {{
            background: rgba(255, 255, 255, 0.08);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            backdrop-filter: blur(5px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}

        h2 {{
            color: #00d4ff;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}

        h3 {{
            color: #7b2ff7;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}

        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}

        .config-item {{
            background: rgba(123, 47, 247, 0.1);
            border: 1px solid rgba(123, 47, 247, 0.3);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }}

        .config-label {{
            color: #a0a0a0;
            font-size: 0.9em;
            margin-bottom: 5px;
        }}

        .config-value {{
            color: #00d4ff;
            font-size: 1.1em;
            font-weight: bold;
        }}

        .topics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}

        .topic-card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.3s ease;
        }}

        .topic-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(123, 47, 247, 0.2);
        }}

        .topic-name {{
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 10px;
            color: #00d4ff;
        }}

        .topic-metrics {{
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
        }}

        .metric {{
            text-align: center;
        }}

        .metric-label {{
            color: #a0a0a0;
            font-size: 0.8em;
        }}

        .metric-value {{
            font-weight: bold;
            font-size: 1.1em;
        }}

        .sentiment-bar {{
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            margin: 10px 0;
            overflow: hidden;
        }}

        .sentiment-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }}

        .concepts-list {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}

        .concept-item {{
            background: rgba(0, 212, 255, 0.1);
            border-left: 4px solid #00d4ff;
            padding: 20px;
            border-radius: 8px;
        }}

        .content-item {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 25px;
            margin: 30px 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}

        .passage-header {{
            background: rgba(123, 47, 247, 0.2);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 25px;
            text-align: center;
        }}

        .passage-text {{
            font-style: italic;
            font-size: 1.1em;
            color: #e0e0e0;
            margin: 15px 0;
        }}

        .angles-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}

        .angle-card {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
        }}

        .angle-title {{
            color: #00ff88;
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}

        .target-audience {{
            color: #ffaa00;
            font-size: 0.9em;
            margin-bottom: 15px;
        }}

        .script-content {{
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            white-space: pre-wrap;
        }}

        .evaluation-scores {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-top: 15px;
        }}

        .score-item {{
            display: flex;
            justify-content: space-between;
            padding: 5px 10px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 5px;
        }}

        .score-label {{
            color: #a0a0a0;
            font-size: 0.9em;
        }}

        .score-value {{
            color: #00d4ff;
            font-weight: bold;
        }}

        .concerns-list {{
            background: rgba(255, 102, 0, 0.1);
            border-left: 4px solid #ff6600;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}

        .concern-item {{
            margin: 10px 0;
            padding-left: 20px;
            position: relative;
        }}

        .concern-item:before {{
            content: "•";
            color: #ff6600;
            position: absolute;
            left: 0;
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}
            
            .config-grid,
            .topics-grid,
            .concepts-list,
            .angles-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Content Agent Report</h1>
            <div class="subtitle">Biblical Content Generation Analysis</div>
            <div class="timestamp">Generated: {content_json['timestamp']}</div>
        </header>

        <div class="section">
            <h2>Configuration</h2>
            <div class="config-grid">
                <div class="config-item">
                    <div class="config-label">Concepts Requested</div>
                    <div class="config-value">{content_json['config']['count']}</div>
                </div>
                <div class="config-item">
                    <div class="config-label">Model Used</div>
                    <div class="config-value">{content_json['config']['model']}</div>
                </div>
                <div class="config-item">
                    <div class="config-label">News Analysis</div>
                    <div class="config-value">{'Enabled' if content_json['config']['use_news'] else 'Disabled'}</div>
                </div>
                <div class="config-item">
                    <div class="config-label">Generation Status</div>
                    <div class="config-value">{'Success' if content_json['generation_result']['success'] else 'Failed'}</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>News Sentiment Analysis</h2>
            <div class="topics-grid">
"""

    # Add trending topics
    for topic in content_json['news_sentiment']['trending_topics']:
        sentiment_color = get_sentiment_color(topic['sentiment'])
        sentiment_width = abs(topic['sentiment']) * 100
        
        html_template += f"""
                <div class="topic-card">
                    <div class="topic-name">{topic['topic'].title()}</div>
                    <div class="topic-metrics">
                        <div class="metric">
                            <div class="metric-label">Frequency</div>
                            <div class="metric-value">{topic['frequency']}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Sentiment</div>
                            <div class="metric-value" style="color: {sentiment_color}">{topic['sentiment']:.1f}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Relevance</div>
                            <div class="metric-value">{topic['relevance']:.1f}</div>
                        </div>
                    </div>
                    <div class="sentiment-bar">
                        <div class="sentiment-fill" style="width: {sentiment_width}%; background-color: {sentiment_color}"></div>
                    </div>
                </div>
"""

    html_template += """
            </div>
            
            <h3>Recommended Themes</h3>
            <div class="concepts-list">
"""

    # Add recommended themes
    for theme in content_json['news_sentiment']['recommended_themes']:
        html_template += f"""
                <div class="concept-item">
                    <div>{theme}</div>
                </div>
"""

    html_template += """
            </div>
            
            <h3>Audience Concerns</h3>
            <div class="concerns-list">
"""

    # Add audience concerns
    for concern in content_json['news_sentiment']['audience_concerns']:
        html_template += f"""
                <div class="concern-item">{concern}</div>
"""

    html_template += """
            </div>
        </div>

        <div class="section">
            <h2>Generated Content</h2>
"""

    # Add multi-angle content if available
    if multi_angle_data:
        for concept_name, concept_data in multi_angle_data.items():
            html_template += f"""
            <div class="content-item">
                <div class="passage-header">
                    <h3>{concept_data['concept']}</h3>
                    <div class="passage-text">"{concept_data['text']}"</div>
                    <div style="color: #a0a0a0; margin-top: 10px;">{concept_data['passage']} ({concept_data.get('translation', 'ESV')})</div>
                </div>
                
                <div class="angles-grid">
"""

            for angle in concept_data['narrative_angles']:
                total_score = sum(angle['evaluation']['scores'].values())
                
                html_template += f"""
                    <div class="angle-card">
                        <div class="angle-title">{angle['angle_info']['angle']}</div>
                        <div class="target-audience">Target: {angle['angle_info']['target_audience']}</div>
                        <div style="color: #a0a0a0; font-size: 0.9em; margin-bottom: 10px;">
                            {angle['angle_info']['justification']}
                        </div>
                        <div class="script-content">{angle['script']}</div>
                        <div style="color: #00ff88; font-weight: bold; margin: 10px 0;">
                            Total Score: {total_score}/28
                        </div>
                        <div class="evaluation-scores">
"""

                for metric, score in angle['evaluation']['scores'].items():
                    html_template += f"""
                            <div class="score-item">
                                <span class="score-label">{metric.replace('_', ' ')}</span>
                                <span class="score-value">{score}/7</span>
                            </div>
"""

                html_template += """
                        </div>
                    </div>
"""

            html_template += """
                </div>
            </div>
"""

    html_template += f"""
        </div>

        <div class="section">
            <h2>Generation Summary</h2>
            <div class="config-grid">
                <div class="config-item">
                    <div class="config-label">Concepts Generated</div>
                    <div class="config-value">{len(content_json['summary']['concepts_used'])}</div>
                </div>
                <div class="config-item">
                    <div class="config-label">Overall Success</div>
                    <div class="config-value">{'✓' if content_json['summary']['success'] else '✗'}</div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

    return html_template


def main():
    """Main conversion function."""
    try:
        # Load data
        content_json = load_latest_content_json()
        multi_angle_data = load_multi_angle_content()
        
        # Generate HTML
        html_content = generate_html(content_json, multi_angle_data)
        
        # Save HTML file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"content_report_{timestamp}.html"
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"✓ HTML report generated: {output_file}")
        print(f"✓ Loaded {len(multi_angle_data)} multi-angle content files")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())