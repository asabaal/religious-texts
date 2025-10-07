#!/usr/bin/env python3
"""
Simple test script for the content agent functionality
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

def test_chat():
    """Test basic chat functionality"""
    print("Testing basic chat functionality...")
    
    msgs = [{"role": "user", "content": "Hello, respond with just 'CHAT_TEST'"}]
    
    try:
        r = requests.post(f"{OLLAMA}/api/chat", json={
            "model": GEN_MODEL, "messages": msgs, "stream": False,
            "options": {"temperature": 0.7}
        })
        r.raise_for_status()
        response = r.json()["message"]["content"]
        print(f"Chat response: {response}")
        return True
    except Exception as e:
        print(f"Chat test failed: {e}")
        return False

def test_rag_functions():
    """Test RAG module functions"""
    print("\nTesting RAG functions...")
    
    try:
        # Test indexing
        ensure_indexes()
        print("✓ Indexing works")
        
        # Test search
        results = search_passages_for_concept("love", top_k=3)
        print(f"✓ Search results: {results}")
        
        # Test passage retrieval
        if results:
            texts = get_passage_texts_by_ref(results[0])
            print(f"✓ Passage texts for {results[0]}: {list(texts.keys())}")
        
        return True
    except Exception as e:
        print(f"RAG test failed: {e}")
        return False

def test_concept_ideation():
    """Test concept ideation with simple prompt"""
    print("\nTesting concept ideation...")
    
    sys = "You are a biblical content ideator. Be faithful, concise, and diverse."
    usr = (
        "Propose 3 short, distinct biblical concepts (3–5 words each). "
        "Return JSON list only."
    )
    
    msgs = []
    if sys: msgs.append({"role":"system","content":sys})
    msgs.append({"role":"user","content":usr})
    
    try:
        r = requests.post(f"{OLLAMA}/api/chat", json={
            "model": GEN_MODEL, "messages": msgs, "stream": False,
            "options": {"temperature": 0.8}
        })
        r.raise_for_status()
        response = r.json()["message"]["content"]
        print(f"Concept ideation response: {response}")
        
        # Try to parse as JSON
        try:
            concepts = json.loads(response)
            print(f"✓ Parsed concepts: {concepts}")
            return True
        except json.JSONDecodeError:
            print("✗ Could not parse response as JSON")
            return False
            
    except Exception as e:
        print(f"Concept ideation test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Content Agent Components")
    print("=" * 40)
    
    # Test components individually
    chat_ok = test_chat()
    rag_ok = test_rag_functions()
    ideation_ok = test_concept_ideation()
    
    print("\n" + "=" * 40)
    print("Test Results:")
    print(f"Chat API: {'✓' if chat_ok else '✗'}")
    print(f"RAG Functions: {'✓' if rag_ok else '✗'}")
    print(f"Concept Ideation: {'✓' if ideation_ok else '✗'}")
    
    if all([chat_ok, rag_ok, ideation_ok]):
        print("\n✓ All tests passed! The content agent should work.")
    else:
        print("\n✗ Some tests failed. Check the errors above.")