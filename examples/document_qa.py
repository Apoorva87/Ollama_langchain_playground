from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.schema import Document
from typing import List, Dict, Any
import os

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

def run_document_qa(doc_path: str, question: str, model_name: str = "qwen3:8b") -> str:
    """Run a document QA interaction with the model.
    
    Args:
        doc_path: Path to the document file
        question: The question to ask about the document
        model_name: Name of the model to use
        
    Returns:
        The model's answer as a string
    """
    # Initialize Ollama
    llm = Ollama(model=model_name)

    # Load and process the document
    documents = load_document(doc_path)
    texts = process_documents(documents)

    # Create vector store
    vectorstore = create_vectorstore(texts, model_name)
    retriever = vectorstore.as_retriever()

    # Create QA chain
    qa_chain = create_qa_chain(llm, retriever)

    # Get answer
    result = qa_chain({"query": question})
    return result["result"]

def run_document_qa_with_uploaded_file(file_content: str, question: str, model_name: str = "qwen3:8b") -> str:
    """Run document QA with uploaded file content.
    
    Args:
        file_content: Content of the uploaded file
        question: The question to ask about the document
        model_name: Name of the model to use
        
    Returns:
        The model's answer as a string
    """
    # Initialize Ollama
    llm = Ollama(model=model_name)

    # Create document from uploaded content
    document = Document(page_content=file_content, metadata={"source": "uploaded_file"})
    documents = [document]
    
    # Process documents
    texts = process_documents(documents)

    # Create vector store
    vectorstore = create_vectorstore(texts, model_name)
    retriever = vectorstore.as_retriever()

    # Create QA chain
    qa_chain = create_qa_chain(llm, retriever)

    # Get answer
    result = qa_chain({"query": question})
    return result["result"]

def main():
    # Interactive QA loop
    print("Document QA System (type 'quit' to exit)")
    print("-" * 50)

    while True:
        question = input("\nYour question: ")
        if question.lower() == 'quit':
            break

        result = run_document_qa("sample.txt", question)
        print("\nAnswer:", result)
        print("-" * 50)

if __name__ == "__main__":
    main() 