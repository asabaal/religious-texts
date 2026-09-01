#!/usr/bin/env python3
"""
CLI Examples Demo Script

This script demonstrates various CLI commands for the Biblical Content Agent.
Run this script to see examples of all major CLI functionality.
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

def run_command(cmd, description):
    """Run a CLI command and display results."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"📝 Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ SUCCESS")
            if result.stdout.strip():
                print("📤 Output:")
                print(result.stdout)
        else:
            print("❌ FAILED")
            print(f"Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT (Command took too long)")
    except Exception as e:
        print(f"💥 ERROR: {e}")

def main():
    """Run all CLI demo examples."""
    print("🚀 Biblical Content Agent - CLI Demo")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Change to project directory
    project_dir = Path(__file__).parent
    import os
    os.chdir(project_dir)
    
    # Demo commands
    demos = [
        # Help commands
        (["python", "content_gen_cli.py", "--help"], "Show main help"),
        (["python", "content_gen_cli.py", "generate", "--help"], "Show generate help"),
        
        # Status commands
        (["python", "content_gen_cli.py", "status"], "Check system status"),
        
        # Configuration commands
        (["python", "content_gen_cli.py", "config", "--show"], "Show current configuration"),
        
        # News analysis
        (["python", "content_gen_cli.py", "news"], "Get news sentiment analysis"),
        
        # Content generation (small demo)
        (["python", "content_gen_cli.py", "generate", "--count", "1", "--no-news"], 
         "Generate 1 content concept without news"),
    ]
    
    # Run each demo
    for cmd, description in demos:
        run_command(cmd, description)
        
        # Ask user if they want to continue
        try:
            response = input("\nPress Enter to continue to next demo, or 'q' to quit: ")
            if response.lower() == 'q':
                break
        except KeyboardInterrupt:
            print("\n\n👋 Demo interrupted by user")
            break
    
    print(f"\n🎉 Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("💡 Try running these commands directly in your terminal!")

if __name__ == "__main__":
    main()