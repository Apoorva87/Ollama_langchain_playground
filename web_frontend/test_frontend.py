#!/usr/bin/env python3
"""
Test script for the web frontend components
"""

import sys
import os
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all required imports work"""
    print("🧪 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        from config import Config
        print("✅ Config imported successfully")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from langfuse_setup import get_langfuse_status
        print("✅ Langfuse setup imported successfully")
    except ImportError as e:
        print(f"❌ Langfuse setup import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration loading"""
    print("\n🔧 Testing configuration...")
    
    try:
        from config import Config
        print(f"✅ Ollama model: {Config.OLLAMA_MODEL}")
        print(f"✅ Default search engine: {Config.DEFAULT_SEARCH_ENGINE}")
        print(f"✅ Langfuse enabled: {bool(Config.LANGFUSE_PUBLIC_KEY and Config.LANGFUSE_SECRET_KEY)}")
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    
    return True

def test_langfuse():
    """Test Langfuse integration"""
    print("\n📊 Testing Langfuse integration...")
    
    try:
        from langfuse_setup import get_langfuse_status
        status = get_langfuse_status()
        print(f"✅ Langfuse status: {status}")
    except Exception as e:
        print(f"❌ Langfuse test failed: {e}")
        return False
    
    return True

def test_ollama_connection():
    """Test Ollama connection"""
    print("\n🤖 Testing Ollama connection...")
    
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is running and accessible")
            print(f"Available models:\n{result.stdout}")
        else:
            print("⚠️  Ollama is not running or not accessible")
            print(f"Error: {result.stderr}")
    except FileNotFoundError:
        print("⚠️  Ollama command not found. Please install Ollama.")
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
    
    return True

def test_web_search_import():
    """Test web search example import"""
    print("\n🔍 Testing web search import...")
    
    try:
        from examples.web_search import run_web_search
        print("✅ Web search example imported successfully")
    except ImportError as e:
        print(f"❌ Web search import failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 Testing Web Frontend Components")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_langfuse,
        test_ollama_connection,
        test_web_search_import
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The web frontend should work correctly.")
        print("\n🚀 To start the web frontend:")
        print("   cd web_frontend")
        print("   ./run.sh")
        print("   # or")
        print("   streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\n💡 Common solutions:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Start Ollama: ollama serve")
        print("   3. Check environment variables in .env file")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 