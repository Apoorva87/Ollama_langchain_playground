from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

def main():
    # Initialize Ollama
    llm = Ollama(model="qwen3:8b")

    # Create a custom prompt template
    template = """The following is a friendly conversation between a human and an AI. 
    The AI is helpful, creative, clever, and very friendly. The AI remembers the conversation history.

    Current conversation:
    {history}
    Human: {input}
    AI:"""

    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )

    # Initialize memory
    memory = ConversationBufferMemory()

    # Create conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=True
    )

    # Interactive chat loop
    print("Memory Chat System (type 'quit' to exit)")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break

        print("\nAssistant: ", end="")
        response = conversation.predict(input=user_input)
        print(response)
        print("-" * 50)

if __name__ == "__main__":
    main() 