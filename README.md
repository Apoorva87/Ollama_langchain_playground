# Ollama LangChain Playground

This repository contains examples and experiments using Ollama models with LangChain. The project demonstrates various use cases and integrations of Ollama's local LLM capabilities with LangChain's powerful framework.

## Prerequisites

- Python 3.8+
- Ollama installed and running locally
- qwen3:8b model pulled in Ollama

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Apoorva87/Ollama_langchain_playground.git
cd Ollama_langchain_playground
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure Ollama is running with qwen3:8b model:
```bash
ollama run qwen3:8b
```

## Examples

The repository contains several examples demonstrating different use cases:

1. Basic Chat - Simple conversation with the model
2. Document QA - Question answering over documents
3. Chain of Thought - Complex reasoning tasks
4. Memory Chat - Chat with memory retention

## Project Structure

```
.
├── examples/
│   ├── basic_chat.py
│   ├── document_qa.py
│   ├── chain_of_thought.py
│   └── memory_chat.py
├── requirements.txt
└── README.md
```

## Usage

Each example can be run independently. For instance:

```bash
python examples/basic_chat.py
```

## License

MIT 