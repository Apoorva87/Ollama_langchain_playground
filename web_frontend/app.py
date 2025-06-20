"""
Streamlit Web Frontend for Ollama LangChain Playground

A comprehensive web interface for interacting with various LLM models
and running different examples from the playground.
"""

import streamlit as st
import sys
import os
import logging
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Literal

# Add the parent directory to Python path to import from the main project
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from the main project
from config import Config
from langfuse_setup import get_langfuse_callback, get_langfuse_status

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Ollama LangChain Playground",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitLogger:
    """Custom logger that writes to Streamlit"""
    
    def __init__(self, container):
        self.container = container
        self.logs = []
    
    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.logs.append(log_entry)
        self.container.text_area("Logs", value="\n".join(self.logs[-50:]), height=200)

class LLMManager:
    """Manages different LLM providers"""
    
    def __init__(self):
        self.llm = None
        self.provider = None
    
    def setup_ollama(self, model: str) -> bool:
        """Setup Ollama LLM"""
        try:
            from langchain_ollama import OllamaLLM
            self.llm = OllamaLLM(model=model)
            self.provider = "ollama"
            return True
        except Exception as e:
            st.error(f"Failed to setup Ollama: {e}")
            return False
    
    def setup_openai(self, api_key: str, model: str) -> bool:
        """Setup OpenAI LLM"""
        try:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                openai_api_key=api_key,
                model=model,
                temperature=0.7
            )
            self.provider = "openai"
            return True
        except Exception as e:
            st.error(f"Failed to setup OpenAI: {e}")
            return False
    
    def setup_anthropic(self, api_key: str, model: str) -> bool:
        """Setup Anthropic LLM"""
        try:
            from langchain_anthropic import ChatAnthropic
            self.llm = ChatAnthropic(
                anthropic_api_key=api_key,
                model=model,
                temperature=0.7
            )
            self.provider = "anthropic"
            return True
        except Exception as e:
            st.error(f"Failed to setup Anthropic: {e}")
            return False
    
    def get_llm(self):
        return self.llm

class ChatManager:
    """Manages chat history and interactions"""
    
    def __init__(self):
        self.messages = []
        self.thinking_logs = []
    
    def add_message(self, role: str, content: str, thinking: Optional[str] = None):
        """Add a message to the chat history"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        if thinking:
            self.thinking_logs.append({
                "content": thinking,
                "timestamp": datetime.now().isoformat()
            })
    
    def get_messages(self):
        return self.messages
    
    def get_thinking_logs(self):
        return self.thinking_logs
    
    def clear_history(self):
        self.messages = []
        self.thinking_logs = []

class ExampleRunner:
    """Runs examples from the examples folder"""
    
    def __init__(self, llm_manager: LLMManager, logger: StreamlitLogger):
        self.llm_manager = llm_manager
        self.logger = logger
    
    def run_basic_chat(self, message: str) -> str:
        """Run basic chat example"""
        try:
            llm = self.llm_manager.get_llm()
            if not llm:
                return "Error: No LLM configured"
            
            response = llm.invoke(message)
            # Handle different response types
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, str):
                return response
            else:
                return str(response)
        except Exception as e:
            self.logger.log(f"Error in basic chat: {e}", "ERROR")
            return f"Error: {e}"
    
    def run_web_search(self, query: str, search_engine: Literal["duckduckgo", "google"] = "duckduckgo") -> str:
        """Run web search example"""
        try:
            from examples.web_search import run_web_search
            response = run_web_search(query, search_engine)
            return response
        except Exception as e:
            self.logger.log(f"Error in web search: {e}", "ERROR")
            return f"Error: {e}"
    
    def run_document_qa(self, question: str, file_content: str) -> str:
        """Run document QA example"""
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain.embeddings import OllamaEmbeddings
            from langchain.vectorstores import FAISS
            from langchain.chains import RetrievalQA
            
            llm = self.llm_manager.get_llm()
            if not llm:
                return "Error: No LLM configured"
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_text(file_content)
            
            # Create embeddings and vector store
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vectorstore = FAISS.from_texts(chunks, embeddings)
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever()
            )
            
            response = qa_chain.invoke({"query": question})
            return response["result"]
        except Exception as e:
            self.logger.log(f"Error in document QA: {e}", "ERROR")
            return f"Error: {e}"

def main():
    """Main application function"""
    
    # Initialize session state
    if 'chat_manager' not in st.session_state:
        st.session_state.chat_manager = ChatManager()
    if 'llm_manager' not in st.session_state:
        st.session_state.llm_manager = LLMManager()
    if 'logger' not in st.session_state:
        st.session_state.logger = StreamlitLogger(st.empty())
    
    # Header
    st.title("🤖 Ollama LangChain Playground")
    st.markdown("A comprehensive web interface for experimenting with LLM models and LangChain")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # LLM Provider Selection
        provider = st.selectbox(
            "Select LLM Provider",
            ["Ollama", "OpenAI", "Anthropic"]
        )
        
        if provider == "Ollama":
            model = st.text_input("Ollama Model", value="qwen3:8b")
            if st.button("Setup Ollama"):
                if st.session_state.llm_manager.setup_ollama(model):
                    st.success(f"Ollama {model} configured successfully!")
                    st.session_state.logger.log(f"Ollama {model} configured")
        
        elif provider == "OpenAI":
            api_key = st.text_input("OpenAI API Key", type="password")
            model = st.selectbox("OpenAI Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"])
            if st.button("Setup OpenAI"):
                if st.session_state.llm_manager.setup_openai(api_key, model):
                    st.success(f"OpenAI {model} configured successfully!")
                    st.session_state.logger.log(f"OpenAI {model} configured")
        
        elif provider == "Anthropic":
            api_key = st.text_input("Anthropic API Key", type="password")
            model = st.selectbox("Anthropic Model", ["claude-3-sonnet-20240229", "claude-3-opus-20240229", "claude-3-haiku-20240307"])
            if st.button("Setup Anthropic"):
                if st.session_state.llm_manager.setup_anthropic(api_key, model):
                    st.success(f"Anthropic {model} configured successfully!")
                    st.session_state.logger.log(f"Anthropic {model} configured")
        
        # Langfuse Status
        st.header("📊 Langfuse Status")
        langfuse_status = get_langfuse_status()
        if langfuse_status["enabled"]:
            st.success("✅ Langfuse Connected")
        else:
            st.warning("⚠️ Langfuse Not Configured")
        
        # Clear Chat Button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_manager.clear_history()
            st.session_state.logger.log("Chat history cleared")
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Example selection
        example_type = st.selectbox(
            "Select Example Type",
            ["Basic Chat", "Web Search", "Document QA"]
        )
        
        # Chat input
        user_input = st.text_area("Your Message", height=100)
        
        # File upload for Document QA
        uploaded_file = None
        if example_type == "Document QA":
            uploaded_file = st.file_uploader(
                "Upload Document (PDF, TXT)",
                type=['pdf', 'txt']
            )
        
        # Search engine selection for Web Search
        search_engine: Literal["duckduckgo", "google"] = "duckduckgo"
        if example_type == "Web Search":
            search_engine = st.selectbox("Search Engine", ["duckduckgo", "google"])
        
        # Send button
        if st.button("🚀 Send", type="primary"):
            if user_input.strip():
                # Add user message
                st.session_state.chat_manager.add_message("user", user_input)
                
                # Initialize example runner
                example_runner = ExampleRunner(
                    st.session_state.llm_manager,
                    st.session_state.logger
                )
                
                # Process based on example type
                with st.spinner("Processing..."):
                    if example_type == "Basic Chat":
                        response = example_runner.run_basic_chat(user_input)
                    elif example_type == "Web Search":
                        response = example_runner.run_web_search(user_input, search_engine)
                    elif example_type == "Document QA":
                        if uploaded_file:
                            file_content = uploaded_file.read().decode('utf-8')
                            response = example_runner.run_document_qa(user_input, file_content)
                        else:
                            response = "Please upload a document first."
                    
                    # Add assistant response
                    st.session_state.chat_manager.add_message("assistant", response)
                    st.session_state.logger.log(f"Generated response for: {user_input[:50]}...")
                
                st.rerun()
        
        # Display chat history
        st.subheader("Chat History")
        for message in st.session_state.chat_manager.get_messages():
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Assistant:** {message['content']}")
            st.markdown("---")
    
    with col2:
        st.header("🧠 Thinking Window")
        
        # Display thinking logs
        thinking_logs = st.session_state.chat_manager.get_thinking_logs()
        if thinking_logs:
            for log in thinking_logs[-5:]:  # Show last 5 thinking logs
                st.text_area(
                    f"Thinking ({log['timestamp'][11:19]})",
                    value=log['content'],
                    height=100,
                    disabled=True
                )
        else:
            st.info("No thinking logs yet. Thinking models will show their reasoning here.")
        
        st.header("📋 Logs")
        
        # Create a container for logs
        log_container = st.container()
        st.session_state.logger.container = log_container

if __name__ == "__main__":
    main() 