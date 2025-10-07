#!/usr/bin/env python3
"""
Launch the Biblical Content Agent Demo in the browser
"""

import webbrowser
import os
import sys

def main():
    # Get the absolute path to the demo file
    demo_path = os.path.abspath("demo.html")
    
    if not os.path.exists(demo_path):
        print("❌ Demo file not found!")
        return
    
    print("🎬 Launching Biblical Content Agent Demo...")
    print(f"📍 Opening: {demo_path}")
    print("🌐 Opening in your default browser...")
    
    # Open in browser
    webbrowser.open(f"file://{demo_path}")
    
    print("✅ Demo launched! Check your browser.")
    print("\n📁 Key files you can view:")
    print("   • rag.py - Core search module")
    print("   • content_agent_optimized.py - Production agent")
    print("   • draft/content_agent.py - Original draft")
    print("   • test_content_agent.py - Testing framework")
    print("   • CONTENT_AGENT_REVIEW.md - Full documentation")

if __name__ == "__main__":
    main()