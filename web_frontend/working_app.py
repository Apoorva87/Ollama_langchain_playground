import streamlit as st
import sys
import os
import re
from typing import List

# Add the parent directory to the Python path to import from examples and config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.schema import Document
from config import Config
from utils import load_css

# Page configuration
st.set_page_config(
    page_title="Ollama LangChain Playground",
    page_icon="🤖",
    layout="wide"
)

# Load and apply custom CSS
css_content = load_css("styles.css")
if css_content:
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize session state for LLM
if 'llm' not in st.session_state:
    st.session_state.llm = None

# Initialize session state for input clearing
if 'clear_input_flag' not in st.session_state:
    st.session_state.clear_input_flag = False

# Initialize session state for logs
if 'logs' not in st.session_state:
    st.session_state.logs = ["Application started. Ready to chat!"]

# Initialize session state for thinking content
if 'thinking_content' not in st.session_state:
    st.session_state.thinking_content = "Thinking content will appear here when the model provides thinking steps."

# Initialize session state for uploaded files
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}  # Dictionary to store multiple files: {filename: content}

# Initialize session state for vector store
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# Initialize session state for vector store status
if 'vectorstore_status' not in st.session_state:
    st.session_state.vectorstore_status = "not_created"  # not_created, building, ready, error

def initialize_llm(model_name: str = "qwen3:8b"):
    """Initialize the LLM with the specified model."""
    try:
        log_message(f"Initializing LLM with model: {model_name}")
        llm = Ollama(
            model=model_name,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )
        log_message(f"✅ LLM initialized successfully with model: {model_name}")
        return llm
    except Exception as e:
        error_msg = f"Failed to initialize LLM: {str(e)}"
        log_message(f"❌ {error_msg}")
        st.error(error_msg)
        return None

def run_basic_chat(user_input: str, model_name: str = "qwen3:8b") -> str:
    """Run a basic chat interaction with the model.
    
    Args:
        user_input: The user's input message
        model_name: The model to use
        
    Returns:
        The model's response as a string
    """
    try:
        log_message(f"Processing user input: '{user_input[:50]}...'")
        
        # Initialize LLM if not already done
        if st.session_state.llm is None:
            log_message("LLM not initialized, initializing now...")
            st.session_state.llm = initialize_llm(model_name)
        
        if st.session_state.llm is None:
            error_msg = "Could not initialize the language model. Please check your Ollama setup."
            log_message(f"❌ {error_msg}")
            return f"Error: {error_msg}"
        
        # Get response from model
        log_message("Sending request to LLM...")
        response = st.session_state.llm.invoke(user_input)
        log_message(f"✅ Received response from LLM (length: {len(response)} chars)")
        return response
    except Exception as e:
        error_msg = f"Error during chat: {str(e)}"
        log_message(f"❌ {error_msg}")
        return f"Error: {str(e)}"

def run_chain_of_thought(question: str, model_name: str = "qwen3:8b") -> str:
    """Run a chain of thought reasoning process.
    
    Args:
        question: The question to reason about
        model_name: The model to use
        
    Returns:
        The model's reasoning and answer as a string
    """
    try:
        log_message(f"Processing chain of thought question: '{question[:50]}...'")
        
        # Initialize LLM if not already done
        if st.session_state.llm is None:
            log_message("LLM not initialized, initializing now...")
            st.session_state.llm = initialize_llm(model_name)
        
        if st.session_state.llm is None:
            error_msg = "Could not initialize the language model. Please check your Ollama setup."
            log_message(f"❌ {error_msg}")
            return f"Error: {error_msg}"
        
        # Create a prompt template for chain of thought reasoning
        prompt = PromptTemplate(
            input_variables=["question"],
            template="""Let's solve this step by step:

Question: {question}

Let's think about this carefully:
1. First, let's understand what the question is asking
2. Then, let's break down the problem into smaller parts
3. Finally, let's arrive at a conclusion

Please provide your reasoning and answer:"""
        )

        # Create the chain
        chain = LLMChain(llm=st.session_state.llm, prompt=prompt)

        # Get response
        log_message("Running chain of thought reasoning...")
        response = chain.run(question)
        log_message(f"✅ Received chain of thought response (length: {len(response)} chars)")
        return response
    except Exception as e:
        error_msg = f"Error during chain of thought: {str(e)}"
        log_message(f"❌ {error_msg}")
        return f"Error: {str(e)}"

def log_message(message: str):
    """Add a message to the logs."""
    st.session_state.logs.append(f"[{st.session_state.get('timestamp', 'now')}] {message}")
    # Keep only last 50 log entries
    if len(st.session_state.logs) > 50:
        st.session_state.logs = st.session_state.logs[-50:]

def extract_thinking_content(response: str) -> tuple[str, str]:
    """Extract thinking content from response and return (thinking_content, clean_response).
    
    Args:
        response: The full response from the LLM
        
    Returns:
        Tuple of (thinking_content, clean_response) where thinking_content is the content
        between <think></think> tags, and clean_response is the response with thinking tags removed
    """
    # Find all thinking content
    thinking_pattern = r'<think>(.*?)</think>'
    thinking_matches = re.findall(thinking_pattern, response, re.DOTALL)
    
    # Extract thinking content
    thinking_content = '\n\n'.join(thinking_matches) if thinking_matches else ""
    
    # Remove thinking tags from response
    clean_response = re.sub(thinking_pattern, '', response, flags=re.DOTALL).strip()
    
    return thinking_content, clean_response

def load_document(file_path: str) -> List[Document]:
    """Load a document from file.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        List of loaded documents
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document file not found: {file_path}")
    
    loader = TextLoader(file_path)
    documents = loader.load()
    return documents

def process_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Process documents by splitting them into chunks.
    
    Args:
        documents: List of documents to process
        chunk_size: Size of each text chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of processed document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    return texts

def create_vectorstore(documents: List[Document], model_name: str = "qwen3:8b") -> FAISS:
    """Create a vector store from documents.
    
    Args:
        documents: List of documents to index
        model_name: Name of the embedding model
        
    Returns:
        FAISS vector store
    """
    embeddings = OllamaEmbeddings(model=model_name)
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def create_qa_chain(llm, retriever) -> RetrievalQA:
    """Create a QA chain with the given LLM and retriever.
    
    Args:
        llm: Language model instance
        retriever: Document retriever
        
    Returns:
        RetrievalQA chain
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

def run_document_qa_with_vectorstore(question: str, model_name: str = "qwen3:8b") -> str:
    """Run document QA using pre-built vector store.
    
    Args:
        question: The question to ask about the documents
        model_name: Name of the model to use
        
    Returns:
        The model's answer as a string
    """
    try:
        log_message(f"Processing document QA question: '{question[:50]}...'")
        log_message(f"Vector store status: {st.session_state.vectorstore_status}")
        
        # Check if vector store is ready
        if st.session_state.vectorstore_status != "ready":
            if st.session_state.vectorstore_status == "building":
                return "⏳ Please wait, vector store is still being built..."
            elif st.session_state.vectorstore_status == "error":
                return "❌ Vector store creation failed. Please try uploading files again."
            else:
                return "❌ No vector store available. Please upload documents first."
        
        # Check if vector store object exists
        if st.session_state.vectorstore is None:
            log_message("❌ Vector store object is None despite ready status")
            return "❌ Vector store not properly initialized. Please try uploading files again."
        
        # Initialize LLM if not already done
        if st.session_state.llm is None:
            log_message("LLM not initialized, initializing now...")
            st.session_state.llm = initialize_llm(model_name)
        
        if st.session_state.llm is None:
            error_msg = "Could not initialize the language model. Please check your Ollama setup."
            log_message(f"❌ {error_msg}")
            return f"Error: {error_msg}"
        
        # Use pre-built vector store
        vectorstore = st.session_state.vectorstore
        retriever = vectorstore.as_retriever()

        # Create QA chain
        log_message("Creating QA chain...")
        qa_chain = create_qa_chain(st.session_state.llm, retriever)

        # Get answer
        log_message("Retrieving answer from documents...")
        result = qa_chain({"query": question})
        log_message(f"✅ Document QA completed (answer length: {len(result['result'])} chars)")
        return result["result"]
    except Exception as e:
        error_msg = f"Error during document QA: {str(e)}"
        log_message(f"❌ {error_msg}")
        return f"Error: {str(e)}"

def build_vectorstore_from_files(file_contents: dict, model_name: str = "qwen3:8b") -> bool:
    """Build vector store from uploaded files.
    
    Args:
        file_contents: Dictionary of {filename: content} for uploaded files
        model_name: Name of the model to use for embeddings
        
    Returns:
        True if successful, False otherwise
    """
    try:
        log_message("🔄 Starting vector store creation...")
        st.session_state.vectorstore_status = "building"
        
        # Create documents from uploaded content
        documents = []
        for filename, content in file_contents.items():
            document = Document(page_content=content, metadata={"source": filename})
            documents.append(document)
        
        # Process documents
        log_message(f"📄 Processing {len(documents)} documents...")
        texts = process_documents(documents)

        # Create vector store
        log_message("🔍 Creating FAISS vector store...")
        vectorstore = create_vectorstore(texts, model_name)
        
        # Store vector store in session state
        st.session_state.vectorstore = vectorstore
        st.session_state.vectorstore_status = "ready"
        
        log_message(f"✅ Vector store created successfully! Indexed {len(texts)} chunks from {len(documents)} documents.")
        return True
        
    except Exception as e:
        error_msg = f"Error building vector store: {str(e)}"
        log_message(f"❌ {error_msg}")
        st.session_state.vectorstore_status = "error"
        return False

# Main content
st.title("🤖 Ollama LangChain Playground")
st.markdown("A comprehensive web interface for experimenting with LLM models and LangChain")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # LLM Provider Selection
    provider = st.selectbox(
        "Select LLM Provider",
        ["Ollama", "OpenAI", "Anthropic"]
    )
    
    if provider == "Ollama":
        model = st.text_input("Ollama Model", value="qwen3:8b")
        
        # Model status indicator
        if st.session_state.llm is not None:
            st.success("✅ Model Ready")
        else:
            st.info("⏳ Model not initialized")
        
        if st.button("Setup Ollama"):
            with st.spinner("Initializing Ollama..."):
                st.session_state.llm = initialize_llm(model)
                if st.session_state.llm is not None:
                    st.success(f"✅ Ollama {model} configured successfully!")
                else:
                    st.error("❌ Failed to initialize Ollama. Please check if Ollama is running.")
    
    elif provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password")
        model = st.selectbox("OpenAI Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"])
        st.info("🔧 OpenAI integration coming soon...")
        if st.button("Setup OpenAI"):
            st.info("OpenAI integration is not yet implemented.")
    
    elif provider == "Anthropic":
        api_key = st.text_input("Anthropic API Key", type="password")
        model = st.selectbox("Anthropic Model", ["claude-3-sonnet-20240229", "claude-3-opus-20240229", "claude-3-haiku-20240307"])
        st.info("🔧 Anthropic integration coming soon...")
        if st.button("Setup Anthropic"):
            st.info("Anthropic integration is not yet implemented.")
    
    # Status section
    st.markdown("---")
    st.subheader("📊 Status")
    
    if st.session_state.llm is not None:
        st.success("✅ LLM Ready")
        st.info(f"Model: {model if provider == 'Ollama' else 'Not set'}")
    else:
        st.warning("⚠️ LLM Not Ready")
        st.info("Please configure your LLM provider above.")
    
    # Vector store status
    st.markdown("---")
    st.subheader("📚 Vector Store")
    
    if st.session_state.vectorstore_status == "ready":
        st.success("✅ Vector Store Ready")
        if st.session_state.uploaded_files:
            st.info(f"Documents: {len(st.session_state.uploaded_files)} files loaded")
    elif st.session_state.vectorstore_status == "building":
        st.warning("🔄 Building Vector Store")
        st.info("Please wait while documents are being processed...")
    elif st.session_state.vectorstore_status == "error":
        st.error("❌ Vector Store Error")
        st.info("Failed to create vector store. Try uploading files again.")
    else:
        st.info("📄 No Vector Store")
        st.info("Upload documents to create vector store.")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Chat Interface")
    
    # Example selection
    example_type = st.selectbox(
        "Select Example Type",
        ["Basic Chat", "Chain of Thought", "Web Search", "Document QA"]
    )
    
    # File upload for Document QA (moved here)
    if example_type == "Document QA":
        uploaded_files = st.file_uploader(
            "Upload Documents (PDF, TXT)",
            type=['pdf', 'txt'],
            accept_multiple_files=True
        )
        
        # Process uploaded files
        if uploaded_files:
            try:
                # Read file contents
                file_contents = {file.name: file.read().decode('utf-8') for file in uploaded_files}
                st.session_state.uploaded_files = file_contents
                
                # Show file info
                #for filename, content in file_contents.items():
                #    file_size = len(content)
                #    st.success(f"✅ File uploaded successfully! Size: {file_size} characters")
                #    st.info(f"📄 File: {filename}")
                
                # Show uploaded files summary
                total_size = sum(len(content) for content in file_contents.values())
                #st.success(f"✅ {len(file_contents)} files uploaded successfully! Total size: {total_size} characters")
                #for filename in file_contents.keys():
                #    st.info(f"📄 {filename}")
                
                # Automatically build vector store
                model_name = model if provider == "Ollama" else "qwen3:8b"
                
                # Only build if not already building or ready
                if st.session_state.vectorstore_status not in ["building", "ready"]:
                    with st.spinner("🔄 Building vector store..."):
                        success = build_vectorstore_from_files(file_contents, model_name)
                        if success:
                            st.success("✅ Vector store ready! You can now ask questions about your documents.")
                        else:
                            st.error("❌ Failed to build vector store. Please try uploading files again.")
                else:
                    st.info("🔄 Vector store is already being built or ready.")
                
            except Exception as e:
                st.error(f"❌ Error reading files: {str(e)}")
                st.session_state.uploaded_files = {}
                st.session_state.vectorstore_status = "error"
        else:
            st.session_state.uploaded_files = {}
            st.session_state.vectorstore = None
            st.session_state.vectorstore_status = "not_created"
            st.info("📄 Please upload documents to ask questions about them.")
    
    # Search engine selection for Web Search
    if example_type == "Web Search":
        search_engine = st.selectbox("Search Engine", ["duckduckgo", "google"])
    
    # Chat history display (scrollable)
    st.subheader("Chat History")
    
    if not st.session_state.chat_history:
        st.info("No messages yet. Start a conversation!")
    else:
        # Use an expander to create a scrollable area for chat history
        with st.expander("Chat Messages", expanded=True):
            # Display each message using native Streamlit chat components
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(message["content"])
    
    # Chat input section
    st.subheader("Send Message")
    
    # Use different key if we need to clear the input
    input_key = "user_input_cleared" if st.session_state.clear_input_flag else "user_input"
    
    # Set appropriate placeholder based on example type
    if example_type == "Chain of Thought":
        placeholder = "Ask a question that requires step-by-step reasoning..."
    elif example_type == "Document QA":
        placeholder = "Ask a question about your uploaded document..."
    elif example_type == "Web Search":
        placeholder = "Ask a question that requires web search..."
    else:
        placeholder = "Type your message here..."
    
    user_input = st.text_area("Your Message", height=80, key=input_key, placeholder=placeholder)
    
    # Reset the clear flag after using it
    if st.session_state.clear_input_flag:
        st.session_state.clear_input_flag = False
    
    # Send button
    col_send1, col_send2, col_send3 = st.columns([1, 1, 2])
    with col_send1:
        if st.button("🚀 Send", type="primary"):
            if user_input.strip():
                # Add user message to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": "now"
                })
                
                # Show processing status
                with st.spinner("🤖 Thinking..."):
                    # Get model name from sidebar
                    model_name = model if provider == "Ollama" else "qwen3:8b"
                    
                    # Get actual response from LLM
                    if example_type == "Basic Chat":
                        raw_response = run_basic_chat(user_input, model_name)
                        
                        # Extract thinking content and clean response
                        thinking_content, clean_response = extract_thinking_content(raw_response)
                        
                        # Update thinking window if thinking content was found
                        if thinking_content:
                            log_message(f"Found thinking content (length: {len(thinking_content)} chars)")
                            st.session_state.thinking_content = thinking_content
                        else:
                            st.session_state.thinking_content = "No thinking content found in this response."
                        
                        assistant_response = clean_response
                    elif example_type == "Chain of Thought":
                        raw_response = run_chain_of_thought(user_input, model_name)
                        
                        thinking_content, clean_response = extract_thinking_content(raw_response)

                        if thinking_content:
                            log_message(f"chain of thought: Found thinking content (length: {len(thinking_content)} chars)")
                            st.session_state.thinking_content = thinking_content
                        else:
                            st.session_state.thinking_content = "{chain of thought} No thinking content found in this response."
                        
                        assistant_response = clean_response
                        
                        log_message(f"Chain of thought reasoning completed (length: {len(raw_response)} chars)")
                    elif example_type == "Document QA":
                        if not st.session_state.uploaded_files:
                            assistant_response = "❌ Please upload documents first before asking questions."
                            st.session_state.thinking_content = "No documents uploaded. Please upload documents to enable document QA."
                        else:
                            raw_response = run_document_qa_with_vectorstore(
                                user_input, 
                                model_name
                            )
                            
                            thinking_content, clean_response = extract_thinking_content(raw_response)

                            if thinking_content:
                                log_message(f"Document QA: Found thinking content (length: {len(thinking_content)} chars)")
                                st.session_state.thinking_content = thinking_content
                            else:
                                st.session_state.thinking_content = "{Document QA} No thinking content found in this response."
                            
                            assistant_response = clean_response
                            # For document QA, show the answer in chat and processing info in thinking window 
                            log_message(f"Document QA completed (answer length: {len(raw_response)} chars)")
                    else:
                        # For other example types, show placeholder
                        assistant_response = f"This is a test response for {example_type}. You said: '{user_input}'. In a real implementation, this would be the actual LLM response."
                        st.session_state.thinking_content = "Thinking content will appear here for thinking models."
                
                # Add assistant message to chat history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": assistant_response,
                    "timestamp": "now"
                })
                
                # Clear the input by setting session state
                st.session_state.clear_input_flag = True
                st.rerun()
            else:
                st.warning("Please enter a message first.")
    
    with col_send2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

with col2:
    st.header("🤖 Thinking Window")
    
    # Thinking window controls
    col_thinking1, col_thinking2 = st.columns([3, 1])
    with col_thinking2:
        if st.button("🗑️ Clear Thinking", key="clear_thinking"):
            st.session_state.thinking_content = "Thinking content cleared."
            st.rerun()
    
    thinking_content = st.text_area(
        "Thinking Process", 
        value=st.session_state.thinking_content,
        height=300,
        key="thinking_window"
    )
    
    st.header("📋 Logs")
    logs_content = st.text_area(
        "Application Logs", 
        value="\n".join(st.session_state.logs),
        height=300,
        key="logs_window"
    )

# Footer
st.markdown("---")
st.markdown("**Status:** ✅ Working correctly!")
st.markdown("**Next Steps:** Configure your LLM provider in the sidebar and start chatting!") 