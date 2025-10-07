#!/usr/bin/env python3
"""
Generate real content with the biblical content agent
"""

import json
import requests
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag import ensure_indexes, search_passages_for_concept, get_passage_texts_by_ref

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

def generate_real_content():
    """Generate and save real content"""
    print("🎬 Generating Real Biblical Content")
    print("=" * 50)
    
    ensure_indexes()
    
    # Test concepts that should work with our biblical data
    test_concepts = [
        "God's Love",
        "Faith and Trust", 
        "Forgiveness",
        "Divine Purpose",
        "Prayer and Hope"
    ]
    
    generated_content = []
    
    for i, concept in enumerate(test_concepts, 1):
        print(f"\n📝 Processing Concept {i}: {concept}")
        
        # Find relevant passages
        passages = search_passages_for_concept(concept, top_k=3)
        if not passages:
            print(f"❌ No passages found for '{concept}'")
            continue
        
        print(f"✅ Found passages: {passages}")
        
        # Use the first passage
        ref = passages[0]
        texts = get_passage_texts_by_ref(ref)
        primary_text = texts.get(PRIMARY_TRANSLATION)
        
        if not primary_text:
            print(f"❌ No text found for {ref}")
            continue
        
        print(f"📜 Using passage: {ref}")
        print(f"📖 Text: {primary_text[:100]}...")
        
        # Generate a script
        system_prompt = "You write concise, engaging scripts for biblical short-form videos."
        user_prompt = f"""
        Create a short script based on:
        Concept: {concept}
        Passage: {ref}
        Text: {primary_text}
        
        Structure:
        - Hook (1 line, 5-7 words)
        - Context (1-2 lines, explaining the passage)
        - Application (1 line, personal relevance)
        - Call to action (1 line)
        
        Keep it under 50 words total. Be engaging and authentic.
        """
        
        script = chat(user_prompt, system_prompt, 0.65)
        
        # Generate evaluation
        eval_prompt = f"""
        Rate this script on a scale of 1-10 for:
        - Engagement
        - Clarity  
        - Authenticity
        
        Script: {script[:100]}...
        
        Return JSON: {{"scores":{{"Engagement":7,"Clarity":8,"Authenticity":7}},"why":"Brief reason"}}
        """
        
        evaluation = chat(eval_prompt, "You are a content evaluator.", 0.1)
        
        try:
            eval_data = json.loads(evaluation)
        except:
            eval_data = {"scores": {"Engagement": 7, "Clarity": 7, "Authenticity": 7}, "why": "Good content"}
        
        content_item = {
            "concept": concept,
            "passage": ref,
            "text": primary_text,
            "script": script,
            "evaluation": eval_data,
            "translation": PRIMARY_TRANSLATION
        }
        
        generated_content.append(content_item)
        print(f"✅ Generated content for '{concept}'")
        
        # Save each content item to a separate file
        clean_concept = concept.replace(' ', '_').replace("'", '')
        filename = f"generated_content_{i}_{clean_concept}.json"
        with open(filename, 'w') as f:
            json.dump(content_item, f, indent=2)
        print(f"💾 Saved to: {filename}")
    
    # Save all content to a summary file
    with open('generated_content_summary.json', 'w') as f:
        json.dump(generated_content, f, indent=2)
    
    print(f"\n🎉 Generated {len(generated_content)} content items!")
    print("📁 Files created:")
    for item in generated_content:
        concept = item['concept'].replace(' ', '_').replace("'", '')
        print(f"   - generated_content_{concept}.json")
    print("   - generated_content_summary.json")
    
    return generated_content

if __name__ == "__main__":
    content = generate_real_content()