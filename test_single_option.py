#!/usr/bin/env python3
"""
Test single option generation from content agent
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
    sys = "You are a biblical content ideator. Be faithful, concise, and diverse."
    usr = (
        f"Propose {n} short, distinct biblical concepts (3–5 words each). "
        "Return JSON list only."
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
        "Generate 1 distinct angle with 1-line descriptor. "
        "Return JSON: {\"angle\":\"...\",\"why\":\"...\"}"
    )
    out = chat(usr, sys, 0.7)
    try:
        result = json.loads(out)
        return [result] if isinstance(result, dict) else result[:1]
    except:
        return [{"angle":"Personal faith vs. religious rules","why":"High tension and compassion"}]

def draft_script(concept, ref, primary_text, xrefs):
    sys = ("You write concise, faithful scripts. Structure strictly:\n"
           "Hook (<=7s)\nScripture Context (15–20s)\n"
           "[Personal Connection: leave blank; add 3 prompting bullets]\n"
           "Application (8–10s)\nCTA (<=5s)\nPromo (<=3s)\n"
           "Cite verses by ref. Avoid doctrinal overreach.")
    usr = (
        f"Concept: {concept}\nPrimary Passage: {ref}\n"
        f"Primary Text ({PRIMARY_TRANSLATION}):\n{primary_text}\n\n"
        f"Optional Cross-Refs: {', '.join(xrefs) if xrefs else 'None'}\n\n"
        "Write ONE complete draft now."
    )
    return chat(usr, sys, 0.65)

def evaluate(draft):
    sys = "You are a strict evaluator returning compact JSON."
    usr = (
        "Score 0–10 for Emotional, Hook, Engagement, ScripturalClarity, VoiceFit. "
        "Also include 'why' as a short sentence. "
        "Return JSON: {\"scores\":{...},\"why\":\"...\"}\n\nDraft:\n" + draft
    )
    try:
        return json.loads(chat(usr, sys, 0.1))
    except:
        return {"scores":{}, "why":"n/a"}

def generate_single_option():
    ensure_indexes()
    concepts = ideate_concepts(3)
    print(f"Generated concepts: {concepts}")
    
    if not concepts:
        print("No concepts generated")
        return None
    
    concept = concepts[0]
    print(f"Using concept: {concept}")
    
    # Search for passages
    hits = search_passages_for_concept(concept, top_k=4)
    if not hits:
        print("No passages found")
        return None
    
    # Pick diverse passages
    picked = []
    seen_books = set()
    for ref in hits:
        book = ref.split()[0]
        if book in seen_books: continue
        seen_books.add(book)
        picked.append(ref)
        if len(picked) == 2: break
    
    if not picked:
        print("No diverse passages found")
        return None
    
    print(f"Selected passages: {picked}")
    
    # Generate angles
    angles = generate_angles(concept, picked)
    print(f"Generated angles: {angles}")
    
    if not angles:
        print("No angles generated")
        return None
    
    # Use first passage and first angle
    ref = picked[0]
    angle = angles[0]
    
    # Get passage text
    texts = get_passage_texts_by_ref(ref)
    primary_text = texts.get(PRIMARY_TRANSLATION)
    if not primary_text:
        print(f"No {PRIMARY_TRANSLATION} text for {ref}")
        return None
    
    # Get cross-refs
    xrefs = [r for r in picked if r != ref][:1]
    
    # Draft script
    print("Generating script...")
    draft = draft_script(f"{concept} — {angle['angle']}", ref, primary_text, xrefs)
    print("Script generated!")
    
    # Evaluate
    print("Evaluating script...")
    score = evaluate(draft)
    
    return {
        "concept": concept,
        "angle": angle,
        "ref": ref,
        "draft": draft,
        "eval": score
    }

if __name__ == "__main__":
    print("Generating single content option...")
    option = generate_single_option()
    
    if option:
        print("\n" + "="*50)
        print("GENERATED OPTION:")
        print(f"Concept: {option['concept']}")
        print(f"Angle: {option['angle']['angle']} ({option['angle'].get('why','')})")
        print(f"Passage: {option['ref']} (Primary: {PRIMARY_TRANSLATION})")
        print("Scores:", option["eval"].get("scores"))
        print("Why:", option["eval"].get("why"))
        print("\nDRAFT:")
        print(option["draft"])
    else:
        print("Failed to generate option")