from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def run_chain_of_thought(question: str) -> str:
    """Run a chain of thought reasoning process.
    
    Args:
        question: The question to reason about
        
    Returns:
        The model's reasoning and answer as a string
    """
    # Initialize Ollama
    llm = Ollama(model="qwen3:8b")

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
    chain = LLMChain(llm=llm, prompt=prompt)

    # Get response
    response = chain.run(question)
    return response

def main():
    # Interactive reasoning loop
    print("Chain of Thought Reasoning System (type 'quit' to exit)")
    print("-" * 50)

    while True:
        question = input("\nYour question: ")
        if question.lower() == 'quit':
            break

        print("\nReasoning process:")
        print("-" * 30)
        response = run_chain_of_thought(question)
        print(response)
        print("-" * 50)

if __name__ == "__main__":
    main() 