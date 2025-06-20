import pytest
from pathlib import Path
import sys
import os
from typing import Generator
import asyncio

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import our sample scripts
from examples.basic_chat import run_basic_chat
from examples.document_qa import run_document_qa
from examples.chain_of_thought import run_chain_of_thought
from examples.memory_chat import run_memory_chat
from examples.web_search import run_web_search

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

def test_basic_chat():
    """Test the basic chat functionality."""
    response = run_basic_chat("What is 2+2?")
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0

def test_document_qa():
    """Test the document QA functionality."""
    # Create a test document
    test_doc = """
    Python is a high-level programming language.
    It was created by Guido van Rossum and first released in 1991.
    Python is known for its simplicity and readability.
    """
    
    # Save test document
    doc_path = project_root / "test_doc.txt"
    with open(doc_path, "w") as f:
        f.write(test_doc)
    
    try:
        response = run_document_qa(str(doc_path), "Who created Python?")
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        assert "Guido" in response or "Rossum" in response
    finally:
        # Cleanup
        if doc_path.exists():
            os.remove(doc_path)

def test_chain_of_thought():
    """Test the chain of thought functionality."""
    response = run_chain_of_thought("If I have 3 apples and give 2 to my friend, how many do I have left?")
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    assert "1" in response or "one" in response.lower()

def test_memory_chat():
    """Test the memory chat functionality."""
    # Test first message
    response1 = run_memory_chat("My name is Test User")
    assert response1 is not None
    assert isinstance(response1, str)
    assert len(response1) > 0
    
    # Test follow-up message that should use memory
    response2 = run_memory_chat("What's my name?")
    assert response2 is not None
    assert isinstance(response2, str)
    assert len(response2) > 0
    assert "Test User" in response2

class TestWebSearch:
    """Test suite for web search functionality."""
    
    def test_basic_fact(self):
        """Test searching for a basic factual question."""
        response = run_web_search("What is the capital of France?")
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        assert "Paris" in response
    
    def test_current_events(self):
        """Test searching for current events."""
        response = run_web_search("What are the latest developments in AI?")
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        assert any(word in response.lower() for word in ["ai", "artificial", "intelligence"])
    
    def test_complex_query(self):
        """Test a more complex query requiring multiple searches."""
        response = run_web_search("Compare Python and JavaScript programming languages")
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        assert any(word in response.lower() for word in ["python", "javascript"])
    
    def test_error_handling(self):
        """Test handling of invalid or empty queries."""
        with pytest.raises(Exception):
            run_web_search("")
        
        with pytest.raises(Exception):
            run_web_search("   ")
    
    def test_response_format(self):
        """Test that responses are properly formatted."""
        response = run_web_search("What is the weather like today?")
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        # Check for common weather-related terms
        assert any(word in response.lower() for word in ["weather", "temperature", "forecast", "degrees"])

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 