#!/usr/bin/env python3
"""
Simple test script to verify tool usage and Langfuse tracing
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from examples.web_search import run_web_search
from langfuse_setup import get_langfuse_status

def test_tool_usage():
    """Test if tools are being used by the agent."""
    
    print("Testing tool usage with Langfuse tracing...")
    
    # Check Langfuse status
    status = get_langfuse_status()
    print(f"Langfuse enabled: {status['enabled']}")
    print(f"Langfuse host: {status['host']}")
    
    # Test queries that should definitely use tools
    test_queries = [
        "What time is it right now?",
        "What's the current weather in New York?",
        "What are the latest news headlines?",
        "What is the current date?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i}: {query} ---")
        try:
            response = run_web_search(query, "duckduckgo")
            print(f"Response: {response}")
            
            # Check if response shows tool usage
            if any(keyword in response.lower() for keyword in ['action:', 'observation:', 'thought:']):
                print("✅ Tool usage detected!")
            else:
                print("❌ No tool usage detected - agent answered from training data")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_tool_usage() 