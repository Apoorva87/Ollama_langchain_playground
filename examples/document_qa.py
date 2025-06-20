from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

def run_document_qa(doc_path: str, question: str) -> str:
    """Run a document QA interaction with the model.
    
    Args:
        doc_path: Path to the document file
        question: The question to ask about the document
        
    Returns:
        The model's answer as a string
    """
    # Initialize Ollama
    llm = Ollama(model="qwen3:8b")
    embeddings = OllamaEmbeddings(model="qwen3:8b")

    # Load and process the document
    loader = TextLoader(doc_path)
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)

    # Create vector store
    vectorstore = FAISS.from_documents(texts, embeddings)
    retriever = vectorstore.as_retriever()

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

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