#!/usr/bin/env python3
"""
Debug concept generation
"""

import json
import requests

OLLAMA = "http://localhost:11434"
GEN_MODEL = "llama3.1:8b"

def test_simple_concept():
    """Test concept generation with very simple prompt"""
    
    sys = "You are a helpful assistant."
    usr = "List 3 biblical concepts. Return only a JSON array of strings."
    
    msgs = [{"role":"system","content":sys}, {"role":"user","content":usr}]
    
    try:
        r = requests.post(f"{OLLAMA}/api/chat", json={
            "model": GEN_MODEL, "messages": msgs, "stream": False,
            "options": {"temperature": 0.7}
        })
        r.raise_for_status()
        response = r.json()["message"]["content"]
        print(f"Raw response: {repr(response)}")
        
        # Try to parse as JSON
        try:
            concepts = json.loads(response)
            print(f"Parsed JSON: {concepts}")
            return concepts
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return None
            
    except Exception as e:
        print(f"Request error: {e}")
        return None

def test_even_simpler():
    """Test with even simpler prompt"""
    
    msgs = [{"role":"user","content": 'Respond with only: ["faith", "love", "hope"]'}]
    
    try:
        r = requests.post(f"{OLLAMA}/api/chat", json={
            "model": GEN_MODEL, "messages": msgs, "stream": False,
            "options": {"temperature": 0.1}
        })
        r.raise_for_status()
        response = r.json()["message"]["content"]
        print(f"Simple response: {repr(response)}")
        
        try:
            concepts = json.loads(response)
            print(f"Parsed simple: {concepts}")
            return concepts
        except json.JSONDecodeError as e:
            print(f"Simple JSON parse error: {e}")
            return None
            
    except Exception as e:
        print(f"Simple request error: {e}")
        return None

if __name__ == "__main__":
    print("Testing concept generation...")
    
    print("\n1. Testing simple concept prompt:")
    test_simple_concept()
    
    print("\n2. Testing even simpler prompt:")
    test_even_simpler()