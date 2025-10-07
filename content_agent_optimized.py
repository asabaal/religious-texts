#!/usr/bin/env python3
"""
Optimized Content Agent for Biblical Video Content Generation

This script uses Llama 3.1 8B model via Ollama to generate biblical content ideas,
angles, and scripts for short-form videos.
"""

import json
import random
import requests
import sys
import os
from collections import defaultdict

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag import ensure_indexes, search_passages_for_concept, get_passage_texts_by_ref

# Configuration
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

def ideate_concepts(n=5):
    """Generate biblical content concepts"""
    system = "You are a biblical content creator specializing in short-form video content."
    user_prompt = (
        f"Generate {n} compelling biblical concepts for short videos (3-5 words each). "
        "Focus on themes that resonate with modern audiences. "
        "Return only a JSON array of strings."
    )
    
    response = chat(user_prompt, system, 0.8)
    
    try:
        concepts = json.loads(response)
        if isinstance(concepts, list):
            return [str(c).strip() for c in concepts if str(c).strip()][:n]
    except json.JSONDecodeError:
        pass
    
    # Fallback: extract lines that look like concepts
    lines = [line.strip().strip('"\'') for line in response.split('\n') if line.strip()]
    return [line for line in lines if 3 <= len(line.split()) <= 5][:n]

def generate_angles(concept, passages):
    """Generate narrative angles for a concept"""
    system = "You create engaging narrative angles for biblical content."
    user_prompt = (
        f"Concept: {concept}\nBible Passages: {', '.join(passages)}\n\n"
        "Generate 1 compelling angle with a brief explanation. "
        "Return JSON: {\"angle\":\"...\",\"why\":\"...\"}"
    )
    
    response = chat(user_prompt, system, 0.7)
    
    try:
        result = json.loads(response)
        if isinstance(result, dict) and "angle" in result:
            return [result]
    except json.JSONDecodeError:
        pass
    
    # Fallback angle
    return [{"angle": "Personal faith journey", "why": "Relatable and inspiring"}]

def draft_script(concept, ref, primary_text, xrefs):
    """Draft a script for the content"""
    system = "You write concise, engaging scripts for biblical short-form videos."
    user_prompt = (
        f"Create a short script based on:\n"
        f"Concept: {concept}\n"
        f"Passage: {ref}\n"
        f"Text: {primary_text}\n\n"
        "Structure:\n"
        "- Hook (1 line, 5-7 words)\n"
        "- Context (1-2 lines, explaining the passage)\n"
        "- Application (1 line, personal relevance)\n"
        "- Call to action (1 line)\n\n"
        "Keep it under 50 words total. Be engaging and authentic."
    )
    
    return chat(user_prompt, system, 0.65)

def evaluate_script(draft):
    """Evaluate the generated script"""
    system = "You are a content evaluator for biblical media."
    user_prompt = (
        "Rate this script on a scale of 1-10 for:\n"
        "- Engagement\n"
        "- Clarity\n"
        "- Authenticity\n\n"
        f"Script: {draft[:100]}...\n\n"
        "Return JSON: {\"scores\":{\"Engagement\":7,\"Clarity\":8,\"Authenticity\":7},\"why\":\"Brief reason\"}"
    )
    
    response = chat(user_prompt, system, 0.1)
    
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"scores": {"Engagement": 7, "Clarity": 7, "Authenticity": 7}, "why": "Good content"}

def generate_content_options(k=3):
    """Generate complete content options"""
    print("Initializing content generation system...")
    ensure_indexes()
    
    print("Generating concepts...")
    concepts = ideate_concepts(min(k*2, 8))
    if not concepts:
        print("No concepts generated. Using fallback concepts.")
        concepts = ["God's Unconditional Love", "Faith in Difficult Times", "Power of Forgiveness"]
    
    print(f"Generated {len(concepts)} concepts: {concepts}")
    
    candidates = []
    
    for i, concept in enumerate(concepts[:k+1]):
        print(f"Processing concept {i+1}: {concept}")
        
        # Find relevant passages
        passages = search_passages_for_concept(concept, top_k=3)
        if not passages:
            print(f"No passages found for '{concept}'")
            continue
        
        # Select diverse passages
        selected_passages = []
        seen_books = set()
        for passage in passages:
            book = passage.split()[0]
            if book not in seen_books:
                selected_passages.append(passage)
                seen_books.add(book)
                if len(selected_passages) >= 2:
                    break
        
        if not selected_passages:
            continue
        
        # Generate angles
        angles = generate_angles(concept, selected_passages)
        if not angles:
            continue
        
        # Process each angle
        for angle in angles[:1]:  # Limit to 1 angle per concept
            ref = selected_passages[0]
            
            # Get passage text
            texts = get_passage_texts_by_ref(ref)
            primary_text = texts.get(PRIMARY_TRANSLATION)
            if not primary_text:
                continue
            
            # Generate script
            print(f"Generating script for {concept} - {angle['angle']}")
            script = draft_script(concept, ref, primary_text, [])
            
            # Evaluate script
            evaluation = evaluate_script(script)
            
            candidates.append({
                "concept": concept,
                "angle": angle,
                "reference": ref,
                "script": script,
                "evaluation": evaluation,
                "translation": PRIMARY_TRANSLATION
            })
            
            print(f"✓ Generated option for '{concept}'")
            
            # Stop if we have enough candidates
            if len(candidates) >= k:
                break
        
        if len(candidates) >= k:
            break
    
    # Sort by total score
    def total_score(candidate):
        scores = candidate["evaluation"].get("scores", {})
        return sum(scores.values())
    
    candidates.sort(key=total_score, reverse=True)
    return candidates[:k]

def display_options(options):
    """Display the generated content options"""
    print("\n" + "="*60)
    print("BIBLICAL CONTENT OPTIONS")
    print("="*60)
    
    for i, option in enumerate(options, 1):
        print(f"\n--- OPTION {i} ---")
        print(f"Concept: {option['concept']}")
        print(f"Angle: {option['angle']['angle']}")
        print(f"Why: {option['angle'].get('why', 'N/A')}")
        print(f"Passage: {option['reference']} ({option['translation']})")
        
        scores = option['evaluation'].get('scores', {})
        total = sum(scores.values())
        print(f"Scores: {scores} (Total: {total})")
        print(f"Evaluation: {option['evaluation'].get('why', 'N/A')}")
        
        print(f"\nScript:\n{option['script']}")
        print("-" * 40)

def main():
    """Main function"""
    print("🎬 Biblical Content Agent")
    print("Powered by Llama 3.1 8B")
    print("="*40)
    
    try:
        # Test Ollama connection
        test_response = requests.get(f"{OLLAMA}/api/tags")
        if test_response.status_code != 200:
            print("❌ Cannot connect to Ollama. Make sure it's running.")
            return
        
        models = [m['name'] for m in test_response.json().get('models', [])]
        if GEN_MODEL not in models:
            print(f"❌ Model {GEN_MODEL} not found. Available: {models}")
            return
        
        print(f"✓ Connected to Ollama with {GEN_MODEL}")
        
        # Generate content options
        options = generate_content_options(k=3)
        
        if options:
            display_options(options)
            print(f"\n✅ Successfully generated {len(options)} content options!")
        else:
            print("❌ No content options generated.")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()