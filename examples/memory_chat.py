from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# Initialize global conversation chain
_llm = Ollama(model="qwen3:8b")
_template = """The following is a friendly conversation between a human and an AI. 
The AI is helpful, creative, clever, and very friendly. The AI remembers the conversation history.

Current conversation:
{history}
Human: {input}
AI:"""

_prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=_template
)

_memory = ConversationBufferMemory()
_conversation = ConversationChain(
    llm=_llm,
    memory=_memory,
    prompt=_prompt,
    verbose=True
)

def run_memory_chat(user_input: str) -> str:
    """Run a memory-enabled chat interaction.
    
    Args:
        user_input: The user's input message
        
    Returns:
        The model's response as a string
    """
    response = _conversation.predict(input=user_input)
    return response

def main():
    # Interactive chat loop
    print("Memory Chat System (type 'quit' to exit)")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break

        print("\nAssistant: ", end="")
        response = run_memory_chat(user_input)
        print(response)
        print("-" * 50)

if __name__ == "__main__":
    main() 