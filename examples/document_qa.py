from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

def main():
    # Initialize Ollama
    llm = Ollama(model="qwen3:8b")
    embeddings = OllamaEmbeddings(model="qwen3:8b")

    # Load and process the document
    loader = TextLoader("sample.txt")
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

    # Interactive QA loop
    print("Document QA System (type 'quit' to exit)")
    print("-" * 50)

    while True:
        question = input("\nYour question: ")
        if question.lower() == 'quit':
            break

        result = qa_chain({"query": question})
        print("\nAnswer:", result["result"])
        print("\nSource documents:")
        for doc in result["source_documents"]:
            print("-" * 30)
            print(doc.page_content[:200] + "...")
        print("-" * 50)

if __name__ == "__main__":
    main() 