"""
Langfuse Setup Package

This package contains all Langfuse-related configuration and setup files
for the LangChain web search application.
"""

from .langfuse_config import setup_langfuse, get_langfuse_callback, get_langfuse_status

__all__ = ['setup_langfuse', 'get_langfuse_callback', 'get_langfuse_status'] 