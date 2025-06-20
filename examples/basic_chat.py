from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def run_basic_chat(user_input: str) -> str:
    """Run a basic chat interaction with the model.
    
    Args:
        user_input: The user's input message
        
    Returns:
        The model's response as a string
    """
    # Initialize Ollama with qwen3:8b model
    llm = Ollama(
        model="qwen3:8b",
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )
    
    # Get response from model
    response = llm.invoke(user_input)
    return response

def main():
    # Example conversation
    print("Starting a conversation with qwen3:8b (type 'quit' to exit)")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break

        print("\nAssistant: ", end="")
        response = run_basic_chat(user_input)
        print("\n" + "-" * 50)

if __name__ == "__main__":
    main() 