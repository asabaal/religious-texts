#!/usr/bin/env python3
"""
Test content agent with smaller parameters
"""

import json, random, requests
from collections import defaultdict
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag import ensure_indexes, search_passages_for_concept, get_passage_texts_by_ref

OLLAMA = "http://localhost:11434"
GEN_MODEL = "llama3.1:8b"
PRIMARY_TRANSLATION = "ESV"

def chat(prompt, system=None, temp=0.7):
    msgs = []
    if system: msgs.append({"role":"system","content":system})
    msgs.append({"role":"user","content":prompt})
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": GEN_MODEL, "messages": msgs, "stream": False,
        "options": {"temperature": temp}
    })
    r.raise_for_status()
    return r.json()["message"]["content"]

def ideate_concepts(n=3):
    sys = "You are a biblical content ideator."
    usr = (
      f"List {n} biblical concepts for videos (3-5 words each). "
      "Return only a JSON array of strings."
    )
    out = chat(usr, sys, 0.8)
    try:
        items = json.loads(out)
        items = [s.strip() for s in items if isinstance(s, str)]
        return items[:n]
    except:
        # fallback: split lines
        return [l.strip("-• ").strip() for l in out.splitlines() if l.strip()][:n]

def generate_angles(concept, passages):
    sys = "You craft distinct narrative angles for short faith videos."
    usr = (
        f"Concept: {concept}\nPassages: {', '.join(passages)}\n"
        "Generate 1 angle with 1-line descriptor. "
        "Return JSON: {\"angle\":\"...\",\"why\":\"...\"}"
    )
    out = chat(usr, sys, 0.7)
    try:
        result = json.loads(out)
        return [result] if isinstance(result, dict) else result[:1]
    except:
        return [{"angle":"Personal faith journey","why":"Relatable and inspiring"}]

def draft_script(concept, ref, primary_text, xrefs):
    sys = ("You write concise, faithful scripts. Structure:\n"
           "Hook (5 words)\nContext (10 words)\n"
           "Application (5 words)\nCTA (3 words)")
    usr = (
        f"Concept: {concept}\nPassage: {ref}\n"
        f"Text: {primary_text[:50]}...\n\n"
        "Write one short script."
    )
    return chat(usr, sys, 0.65)

def evaluate(draft):
    sys = "You are a strict evaluator returning compact JSON."
    usr = (
        "Score 0-10 for engagement and clarity. "
        "Return JSON: {\"scores\":{\"Engagement\":7,\"Clarity\":8},\"why\":\"Good\"}\n\n"
        f"Draft: {draft[:100]}..."
    )
    try:
        return json.loads(chat(usr, sys, 0.1))
    except:
        return {"scores":{"Engagement":7,"Clarity":8}, "why":"Good content"}

def generate_options(k=2):
    ensure_indexes()
    concepts = ideate_concepts()
    print(f"Concepts: {concepts}")
    
    if not concepts:
        return []

    candidates = []
    for concept in concepts[:2]:  # Limit to 2 concepts
        hits = search_passages_for_concept(concept, top_k=2)
        if not hits: continue
        
        picked = hits[:1]  # Just take first passage
        if not picked: continue

        angles = generate_angles(concept, picked)
        if not angles: continue

        ref = picked[0]
        texts = get_passage_texts_by_ref(ref)
        primary_text = texts.get(PRIMARY_TRANSLATION)
        if not primary_text: continue

        xrefs = []  # Skip cross-refs for simplicity

        for ang in angles[:1]:  # Just first angle
            draft = draft_script(f"{concept} — {ang['angle']}", ref, primary_text, xrefs)
            score = evaluate(draft)
            candidates.append({
                "concept": concept,
                "angle": ang,
                "ref": ref,
                "draft": draft,
                "eval": score
            })
            print(f"Generated candidate for {concept}")
        
        if len(candidates) >= k:
            break

    return candidates[:k]

if __name__ == "__main__":
    print("Generating content options (simplified)...")
    opts = generate_options(k=2)
    
    for i, o in enumerate(opts, 1):
        print(f"\n=== OPTION {i} ===")
        print(f"Concept: {o['concept']}")
        print(f"Angle: {o['angle']['angle']} ({o['angle'].get('why','')})")
        print(f"Passage: {o['ref']}")
        print("Scores:", o["eval"].get("scores"))
        print("Why:", o["eval"].get("why"))
        print("\nDRAFT:\n" + o['draft'][:200] + "...")
    
    print(f"\nGenerated {len(opts)} options successfully!")