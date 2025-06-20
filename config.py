import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration settings
class Config:
    # Google API settings
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
    
    # Ollama model settings
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    
    # Search engine settings
    DEFAULT_SEARCH_ENGINE = os.getenv("DEFAULT_SEARCH_ENGINE", "duckduckgo")  # Options: "duckduckgo" or "google"
    
    # Langfuse settings
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    
    @classmethod
    def validate(cls):
        """Validate that all required configuration is present."""
        if cls.DEFAULT_SEARCH_ENGINE == "google":
            if not cls.GOOGLE_API_KEY:
                raise ValueError(
                    "GOOGLE_API_KEY environment variable is required when using Google search. "
                    "Please set it in your .env file."
                )
            if not cls.GOOGLE_CSE_ID:
                raise ValueError(
                    "GOOGLE_CSE_ID environment variable is required when using Google search. "
                    "Please set it in your .env file. You can get this from the Google Custom Search Console."
                ) 