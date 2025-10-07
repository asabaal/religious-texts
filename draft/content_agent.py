# content_agent/app_autonomous.py
import json, random, requests
from collections import defaultdict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import ensure_indexes, search_passages_for_concept, get_passage_texts_by_ref

OLLAMA = "http://localhost:11434"
GEN_MODEL = "llama3.1:8b"      # or "qwen2.5:7b"
PRIMARY_TRANSLATION = "ESV"    # change as you like

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

def ideate_concepts(n=12):
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
      "Generate 2 distinct angles with 1-line descriptors. "
      "Return JSON: [{\"angle\":\"...\",\"why\":\"...\"}, ...]"
    )
    out = chat(usr, sys, 0.7)
    try:
        return json.loads(out)[:2]
    except:
        return [{"angle":"Personal faith vs. religious rules","why":"High tension and compassion"},
                {"angle":"Jesus renames identity","why":"Strong emotional payoff"}]

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

def generate_options(k=5):
    ensure_indexes()
    concepts = ideate_concepts()
    random.shuffle(concepts)

    candidates = []
    for concept in concepts:
        # 1) semantic search across the whole Bible corpus (all translations aligned)
        hits = search_passages_for_concept(concept, top_k=4)  # returns canonical refs
        if not hits: continue
        # de-dup: enforce diversity across books
        picked = []
        seen_books = set()
        for ref in hits:
            book = ref.split()[0]
            if book in seen_books: continue
            seen_books.add(book)
            picked.append(ref)
            if len(picked) == 2: break

        if not picked: continue

        # 2) create angles
        angles = generate_angles(concept, picked)

        # 3) pull text for primary translation + fetch 1–2 cross refs for color
        for ref in picked:
            texts = get_passage_texts_by_ref(ref)          # dict: {translation: text}
            primary_text = texts.get(PRIMARY_TRANSLATION)  # required for drafting
            if not primary_text: continue
            # naive cross-refs: other translation wordings or nearest pericope neighbors
            xrefs = [r for r in picked if r != ref][:1]

            # 4) draft for each angle
            for ang in angles:
                draft = draft_script(f"{concept} — {ang['angle']}", ref, primary_text, xrefs)
                score = evaluate(draft)
                candidates.append({
                    "concept": concept,
                    "angle": ang,
                    "ref": ref,
                    "draft": draft,
                    "eval": score
                })
        # stop early if we already have many candidates
        if len(candidates) >= 12:
            break

    # 5) rank & de-duplicate (simple: by sum of scores; then book diversity)
    def score_sum(c):
        s = c["eval"].get("scores", {})
        return sum(s.get(k,0) for k in ["Emotional","Hook","Engagement","ScripturalClarity","VoiceFit"])
    candidates.sort(key=score_sum, reverse=True)

    # keep diversity
    final, seen = [], set()
    for c in candidates:
        key = (c["ref"].split()[0], c["concept"].split("—")[0])  # book + base concept
        if key in seen: continue
        seen.add(key)
        final.append(c)
        if len(final) == k: break
    return final

if __name__ == "__main__":
    opts = generate_options(k=5)
    for i, o in enumerate(opts, 1):
        print(f"\n=== OPTION {i} ===")
        print(f"Concept: {o['concept']}")
        print(f"Angle: {o['angle']['angle']}  ({o['angle'].get('why','')})")
        print(f"Passage: {o['ref']}  (Primary: {PRIMARY_TRANSLATION})")
        print("Scores:", o["eval"].get("scores"))
        print("Why:", o["eval"].get("why"))
        print("\nDRAFT:\n" + o["draft"])

