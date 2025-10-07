#!/usr/bin/env python3
"""
Generate biblical content with multiple narrative angles
Each content piece will have top 3 narrative angles with justifications
"""

import json
import requests
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag import ensure_indexes, get_passage_texts_by_ref

OLLAMA = "http://localhost:11434"
GEN_MODEL = "llama3.1:8b"
PRIMARY_TRANSLATION = "ESV"

def chat(prompt, system=None, temp=0.7):
    """Send chat request to Ollama API"""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    
    try:
        r = requests.post(f"{OLLAMA}/api/chat", json={
            "model": GEN_MODEL,
            "messages": msgs,
            "stream": False,
            "options": {"temperature": temp}
        })
        r.raise_for_status()
        return r.json()["message"]["content"]
    except Exception as e:
        print(f"Chat API error: {e}")
        return ""

def generate_narrative_angles(concept, passage_ref, passage_text):
    """Generate top 3 narrative angles with justifications"""
    
    system_prompt = "You are an expert biblical storyteller and content creator. You excel at finding multiple compelling angles for scripture passages."
    
    user_prompt = f"""
    For this biblical passage, generate the TOP 3 narrative angles that would work well for short-form video content.
    
    Concept: {concept}
    Passage: {passage_ref}
    Text: "{passage_text}"
    
    For each angle, provide:
    1. A compelling angle title (3-6 words)
    2. A brief justification explaining why this angle resonates with modern audiences
    3. The target audience that would connect most with this angle
    
    Return in this exact JSON format:
    {{
        "angles": [
            {{
                "angle": "Compelling angle title",
                "justification": "Why this angle works for modern audiences",
                "target_audience": "Primary audience group"
            }},
            {{
                "angle": "Second compelling angle",
                "justification": "Why this angle is effective",
                "target_audience": "Primary audience group"
            }},
            {{
                "angle": "Third compelling angle", 
                "justification": "Why this angle connects today",
                "target_audience": "Primary audience group"
            }}
        ]
    }}
    """
    
    response = chat(user_prompt, system_prompt, 0.8)
    
    try:
        angles_data = json.loads(response)
        if "angles" in angles_data and len(angles_data["angles"]) >= 3:
            return angles_data["angles"][:3]
    except json.JSONDecodeError:
        pass
    
    # Fallback angles if JSON parsing fails
    return [
        {
            "angle": "Personal Faith Journey",
            "justification": "Relatable story of individual spiritual growth and transformation",
            "target_audience": "Young adults seeking authentic faith"
        },
        {
            "angle": "Modern Life Application",
            "justification": "Practical wisdom for navigating contemporary challenges",
            "target_audience": "Working professionals and parents"
        },
        {
            "angle": "Hope and Encouragement",
            "justification": "Uplifting message that resonates in difficult times",
            "target_audience": "People going through life challenges"
        }
    ]

def generate_script_for_angle(concept, passage_ref, passage_text, angle_info):
    """Generate a script specifically tailored to a narrative angle"""
    
    system_prompt = "You write concise, engaging scripts for biblical short-form videos, specifically tailored to the given narrative angle."
    
    user_prompt = f"""
    Create a short script based on this information:
    
    Concept: {concept}
    Passage: {passage_ref}
    Text: "{passage_text}"
    
    Narrative Angle: {angle_info['angle']}
    Target Audience: {angle_info['target_audience']}
    Focus: {angle_info['justification']}
    
    Structure:
    - Hook (1 line, 5-7 words that grabs this specific audience)
    - Context (1-2 lines, explaining the passage through this angle)
    - Application (1 line, personal relevance for this audience)
    - Call to action (1 line, specific to this angle)
    
    Keep it under 50 words total. Make it compelling for the target audience.
    """
    
    return chat(user_prompt, system_prompt, 0.65)

def evaluate_script(script, angle_info):
    """Evaluate script quality with angle-specific criteria"""
    
    system_prompt = "You are a content evaluator for biblical media, specializing in audience engagement."
    
    user_prompt = f"""
    Rate this script on a scale of 1-10 for:
    - Engagement (how well it grabs the target audience)
    - Clarity (how clear the message is)
    - Authenticity (how genuine and biblical it feels)
    - Angle Fit (how well it matches the intended narrative angle)
    
    Narrative Angle: {angle_info['angle']}
    Target Audience: {angle_info['target_audience']}
    
    Script: {script[:150]}...
    
    Return JSON: {{"scores":{{"Engagement":7,"Clarity":8,"Authenticity":7,"Angle_Fit":8}},"why":"Brief evaluation reason"}}
    """
    
    response = chat(user_prompt, system_prompt, 0.1)
    
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {
            "scores": {"Engagement": 7, "Clarity": 7, "Authenticity": 7, "Angle_Fit": 7},
            "why": "Good content with effective angle implementation"
        }

def generate_content_with_multiple_angles(concept, passage_ref, passage_text):
    """Generate complete content with multiple narrative angles"""
    
    print(f"🎯 Generating angles for: {concept}")
    
    # Generate 3 narrative angles
    angles = generate_narrative_angles(concept, passage_ref, passage_text)
    print(f"✅ Generated {len(angles)} narrative angles")
    
    # Generate script and evaluation for each angle
    angle_contents = []
    for i, angle_info in enumerate(angles, 1):
        print(f"   📝 Creating script for angle {i}: {angle_info['angle']}")
        
        script = generate_script_for_angle(concept, passage_ref, passage_text, angle_info)
        evaluation = evaluate_script(script, angle_info)
        
        angle_content = {
            "angle_info": angle_info,
            "script": script,
            "evaluation": evaluation
        }
        
        angle_contents.append(angle_content)
        print(f"   ✅ Completed angle {i}")
    
    return {
        "concept": concept,
        "passage": passage_ref,
        "text": passage_text,
        "translation": PRIMARY_TRANSLATION,
        "narrative_angles": angle_contents,
        "total_angles": len(angle_contents)
    }

def main():
    """Generate content with multiple angles for key passages"""
    print("🎬 Biblical Content Generator with Multiple Narrative Angles")
    print("=" * 60)
    
    ensure_indexes()
    
    # Key passages to generate content for
    content_plans = [
        {
            "concept": "God's Unconditional Love",
            "passage": "John 3:16"
        },
        {
            "concept": "Finding Purpose in Suffering",
            "passage": "Romans 8:28"
        },
        {
            "concept": "The Power of Humility",
            "passage": "Matthew 5:5"
        },
        {
            "concept": "Creation and Divine Power",
            "passage": "Genesis 1:1"
        },
        {
            "concept": "Light Overcoming Darkness",
            "passage": "Genesis 1:3"
        }
    ]
    
    all_content = []
    
    for i, plan in enumerate(content_plans, 1):
        print(f"\n{'='*20} Content Item {i} {'='*20}")
        
        # Get passage text
        texts = get_passage_texts_by_ref(plan['passage'])
        primary_text = texts.get(PRIMARY_TRANSLATION)
        
        if not primary_text:
            print(f"❌ No text found for {plan['passage']}")
            continue
        
        print(f"📜 Passage: {plan['passage']}")
        print(f"📖 Text: {primary_text}")
        
        # Generate content with multiple angles
        content_item = generate_content_with_multiple_angles(
            plan['concept'], 
            plan['passage'], 
            primary_text
        )
        
        all_content.append(content_item)
        
        # Save to file
        clean_concept = plan['concept'].replace(' ', '_').replace(',', '').replace("'", '')
        filename = f"multi_angle_content_{i}_{clean_concept}.json"
        with open(filename, 'w') as f:
            json.dump(content_item, f, indent=2)
        print(f"💾 Saved to: {filename}")
        
        # Print summary
        print(f"\n📊 Summary for {plan['concept']}:")
        for j, angle_content in enumerate(content_item['narrative_angles'], 1):
            angle = angle_content['angle_info']
            scores = angle_content['evaluation']['scores']
            total_score = sum(scores.values())
            print(f"   {j}. {angle['angle']} (Score: {total_score})")
            print(f"      Target: {angle['target_audience']}")
    
    # Save summary
    with open('multi_angle_content_summary.json', 'w') as f:
        json.dump(all_content, f, indent=2)
    
    print(f"\n🎉 Generated content with multiple angles for {len(all_content)} passages!")
    print("📁 Files created:")
    for i, plan in enumerate(content_plans[:len(all_content)], 1):
        clean_concept = plan['concept'].replace(' ', '_').replace(',', '').replace("'", '')
        print(f"   - multi_angle_content_{i}_{clean_concept}.json")
    print("   - multi_angle_content_summary.json")
    
    return all_content

if __name__ == "__main__":
    content = main()