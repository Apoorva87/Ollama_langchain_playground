from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def main():
    # Initialize Ollama with qwen3:8b model
    llm = Ollama(
        model="qwen3:8b",
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

    # Example conversation
    print("Starting a conversation with qwen3:8b (type 'quit' to exit)")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break

        print("\nAssistant: ", end="")
        response = llm.invoke(user_input)
        print("\n" + "-" * 50)

if __name__ == "__main__":
    main() 