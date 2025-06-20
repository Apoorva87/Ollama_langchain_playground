# Ollama LangChain Playground

A comprehensive playground for experimenting with Ollama models and LangChain, featuring web search capabilities, document QA, chain of thought reasoning, memory-enhanced conversations, and full observability with Langfuse.

## 🚀 Features

- **🤖 Multiple LLM Models**: Support for various Ollama models (qwen3:8b, llama3, etc.)
- **🔍 Web Search**: DuckDuckGo and Google search integration with intelligent query processing
- **📄 Document QA**: Question answering from documents
- **🧠 Chain of Thought**: Step-by-step reasoning capabilities
- **💾 Memory Chat**: Conversational memory with chat history
- **📊 Full Observability**: Comprehensive Langfuse integration for tracing, monitoring, and debugging
- **🛠️ Multiple Tools**: Search, datetime, and extensible tool system
- **🔧 Environment Configuration**: Flexible configuration via environment variables
- **📝 Detailed Logging**: Comprehensive logging with JSON and human-readable formats

## 📁 Project Structure

```
Ollama_langchain_playground/
├── examples/                 # Example scripts
│   ├── basic_chat.py        # Basic chat with Ollama
│   ├── document_qa.py       # Document question answering
│   ├── chain_of_thought.py  # Chain of thought reasoning
│   ├── memory_chat.py       # Memory-enhanced chat
│   └── web_search.py        # Advanced web search with observability
├── langfuse_setup/          # Langfuse observability setup
│   ├── docker-compose.yml   # Langfuse Docker configuration
│   ├── langfuse_config.py   # Python configuration module
│   ├── manage_langfuse.sh   # Management script
│   └── README.md           # Langfuse setup guide
├── logs/                    # Application logs
│   ├── langchain_debug.log # Human-readable logs
│   └── langchain_json.log  # JSON-formatted logs
├── config.py               # Configuration settings
├── env.example             # Environment variables template
├── requirements.txt        # Python dependencies
├── test_samples.py         # Test suite
└── README.md              # This file
```

## 🛠️ Quick Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file (or copy from `env.example`):

```env
# Ollama Configuration
OLLAMA_MODEL=qwen3:8b

# Search Engine Configuration
DEFAULT_SEARCH_ENGINE=duckduckgo

# Google Search (optional - only needed if using Google search)
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_google_cse_id

# Langfuse Observability (optional)
LANGFUSE_PUBLIC_KEY=pk-lf-1234567890abcdef
LANGFUSE_SECRET_KEY=sk-lf-1234567890abcdef
LANGFUSE_HOST=http://localhost:3000
```

### 3. Start Langfuse (Optional but Recommended)

```bash
cd langfuse_setup
./manage_langfuse.sh start
```

### 4. Run Examples

```bash
# Basic chat
python3 examples/basic_chat.py

# Web search with full observability
python3 examples/web_search.py

# Document QA
python3 examples/document_qa.py

# Chain of thought
python3 examples/chain_of_thought.py

# Memory chat
python3 examples/memory_chat.py
```

## 🔍 Enhanced Web Search Features

The `web_search.py` example is the most comprehensive demonstration, featuring:

### 🎯 **Core Capabilities**
- **DuckDuckGo Integration**: Free, no API key required, privacy-focused
- **Google Search Integration**: Advanced search with API key and CSE ID
- **DateTime Tool**: Get current date/time for context-aware responses
- **Intelligent Query Processing**: Automatic tool selection and reasoning
- **ReAct Agent**: Step-by-step reasoning with tool usage

### 🔧 **Advanced Features**
- **Environment-based Configuration**: All settings via environment variables
- **Comprehensive Logging**: Detailed debug logs and JSON logging
- **Error Handling**: Robust error handling and validation
- **Flexible Search Engines**: Easy switching between DuckDuckGo and Google
- **Observability Integration**: Full tracing with Langfuse

### 📊 **Observability Features**
- **Complete Trace Logging**: Every step of the agent's reasoning process
- **Tool Execution Tracking**: Detailed logs of search and datetime tool usage
- **Performance Metrics**: Response times, token usage, and costs
- **Error Tracking**: Comprehensive error logging and debugging
- **Real-time Monitoring**: Live dashboard at http://localhost:3000

### 🚀 **Usage Examples**

```bash
# Run with default settings (DuckDuckGo)
python3 examples/web_search.py

# Ask questions like:
# - "What's the weather today?"
# - "What time is it?"
# - "Latest news about AI"
# - "Current events in technology"
# - "How to make a cake"
```

### 🔍 **What You'll See**

1. **Interactive Query Input**: Enter your question
2. **Agent Reasoning**: Watch the agent think through the problem
3. **Tool Execution**: See search and datetime tools in action
4. **Final Response**: Get a comprehensive answer
5. **Langfuse Dashboard**: View detailed traces at http://localhost:3000

## 📊 Observability with Langfuse

Langfuse provides comprehensive tracing and monitoring for your LLM applications:

### 🎯 **What Gets Traced**

- **🤖 LLM Interactions**: Prompts, responses, token usage, costs
- **🛠️ Tool Executions**: Search queries, datetime calls, input/output
- **🔄 Agent Reasoning**: Thought processes, action selections, decision flows
- **📊 Performance Metrics**: Execution times, success rates, resource usage
- **🐛 Error Tracking**: Detailed error logs and debugging information

### 🚀 **Setup and Management**

```bash
# Start Langfuse
cd langfuse_setup
./manage_langfuse.sh start

# Check status
./manage_langfuse.sh status

# View logs
./manage_langfuse.sh logs

# Stop services
./manage_langfuse.sh stop
```

### 📈 **Dashboard Features**

Visit http://localhost:3000 to access:

- **🔍 Traces**: Complete execution traces of every query
- **📊 Metrics**: Performance analytics and cost tracking
- **🛠️ Tool Details**: Individual tool execution logs
- **🤖 LLM Logs**: Model interaction details
- **🐛 Debugging**: Error analysis and troubleshooting

### 💡 **Example Trace Structure**

```
Query: "What's the weather today?"
├── Agent: Initial reasoning
├── Tool: GetCurrentDateTime
│   └── Output: "2025-01-27 14:30:45 EST"
├── Tool: Search (DuckDuckGo)
│   └── Query: "weather today"
│   └── Output: "Weather results..."
└── Agent: Final response generation
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python3 test_samples.py
```

Tests cover:
- Basic chat functionality
- Web search with different engines
- Error handling and validation
- Configuration management
- Langfuse integration
- Tool execution

## 🔧 Configuration

### Environment Variables

All configuration is now environment-based for flexibility:

```env
# Ollama Configuration
OLLAMA_MODEL=qwen3:8b                    # Default model
DEFAULT_SEARCH_ENGINE=duckduckgo         # Default search engine

# Google Search (optional)
GOOGLE_API_KEY=your_google_api_key       # Google Custom Search API key
GOOGLE_CSE_ID=your_google_cse_id         # Google Custom Search Engine ID

# Langfuse Observability (optional)
LANGFUSE_PUBLIC_KEY=pk-lf-...            # Langfuse public key
LANGFUSE_SECRET_KEY=sk-lf-...            # Langfuse secret key
LANGFUSE_HOST=http://localhost:3000      # Langfuse dashboard URL
```

### Model Settings

The system automatically reads from environment variables with sensible defaults:

```python
# config.py automatically reads from environment
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
DEFAULT_SEARCH_ENGINE = os.getenv("DEFAULT_SEARCH_ENGINE", "duckduckgo")
```

## 📚 Code Examples

### Web Search Integration

```python
from examples.web_search import run_web_search

# Run with default settings
response = run_web_search("What's the latest news about AI?")

# Run with specific search engine
response = run_web_search("Weather today", "google")
```

### Langfuse Integration

```python
from langfuse_setup import get_langfuse_callback, get_langfuse_status

# Get callback handler for tracing
callback_handler = get_langfuse_callback()

# Check Langfuse status
status = get_langfuse_status()
print(f"Langfuse enabled: {status['enabled']}")
```

### Custom Agent with Observability

```python
from langchain.agents import AgentExecutor, create_react_agent
from langfuse_setup import get_langfuse_callback

# Get Langfuse callback
callback_handler = get_langfuse_callback()

# Create agent with observability
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    callbacks=[callback_handler] if callback_handler else []
)
```

## 🐛 Troubleshooting

### Common Issues

1. **Ollama not running**
   ```bash
   ollama serve
   ```

2. **Model not found**
   ```bash
   ollama pull qwen3:8b
   ```

3. **Google Search errors**
   - Verify API key and CSE ID in `.env`
   - Check Google Cloud Console for quota limits

4. **Langfuse connection issues**
   ```bash
   cd langfuse_setup
   ./manage_langfuse.sh restart
   ```

5. **Import errors**
   - Ensure you're in the project root directory
   - Check that all dependencies are installed

### Logs and Debugging

Application logs are saved to `logs/`:
- `langchain_debug.log`: Human-readable logs with timestamps
- `langchain_json.log`: JSON-formatted logs for programmatic analysis

Langfuse logs:
```bash
cd langfuse_setup
./manage_langfuse.sh logs
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🎯 Next Steps

1. **Explore Examples**: Start with `web_search.py` for the full experience
2. **Monitor Performance**: Use Langfuse dashboard for insights
3. **Customize Models**: Experiment with different Ollama models
4. **Add Tools**: Extend the tool system with custom functions
5. **Scale Up**: Deploy to production with proper monitoring

---

## 🏗️ Langfuse Architecture & Callbacks

This section explains how Langfuse integrates with LangChain and how the callback system works.

### 🔄 **Callback Architecture Overview**

Langfuse integration works through a custom callback handler that intercepts LangChain events and sends them to the Langfuse dashboard. The architecture follows this flow:

```
User Query → LangChain Agent → Callback Handler → Langfuse Dashboard
                ↓
            Tool Execution → Callback Handler → Langfuse Dashboard
                ↓
            LLM Response → Callback Handler → Langfuse Dashboard
```

### 🎯 **Callback Handler Implementation**

The `LangfuseCallbackHandler` class in `langfuse_setup/langfuse_config.py` implements the following key methods:

#### **Chain Callbacks**
```python
def on_chain_start(self, serialized, inputs, **kwargs):
    """Creates a new trace or span when a chain starts"""
    # Creates main trace or child spans for nested chains

def on_chain_end(self, outputs, **kwargs):
    """Ends the current chain and logs outputs"""
    # Updates span with outputs and ends the trace/span
```

#### **LLM Callbacks**
```python
def on_llm_start(self, serialized, prompts, **kwargs):
    """Creates a generation span for LLM calls"""
    # Starts a generation span to track LLM interactions

def on_llm_end(self, response, **kwargs):
    """Ends LLM generation and extracts text content"""
    # Extracts actual text from response.generations[0][0].text
    # Updates span with clean text content (not full object)
```

#### **Tool Callbacks**
```python
def on_tool_start(self, serialized, input_str, **kwargs):
    """Creates a span for tool execution"""
    # Starts a span to track tool usage

def on_tool_end(self, output, **kwargs):
    """Ends tool execution and logs results"""
    # Updates span with tool output and ends span
```

#### **Agent Callbacks**
```python
def on_agent_action(self, action, **kwargs):
    """Handles ReAct agent actions"""
    # Creates spans for agent reasoning and tool selection
    # Supports both object and dictionary action formats

def on_agent_finish(self, finish, **kwargs):
    """Handles agent completion"""
    # Extracts return_values or output from finish object
    # Ends agent action spans
```

### 🔧 **Key Design Decisions**

#### **1. Span Management**
- **Active Spans Tracking**: Uses `self.active_spans` dictionary to track ongoing operations
- **Span ID Generation**: Uses `id(span)` for unique span identification
- **Proper Cleanup**: Automatically removes completed spans from tracking

#### **2. Response Processing**
- **Text Extraction**: Extracts clean text from `response.generations[0][0].text`
- **Fallback Handling**: Falls back to full response if structure differs
- **Object vs Dict Support**: Handles both object and dictionary formats

#### **3. Error Handling**
- **Comprehensive Try-Catch**: Every callback method wrapped in try-catch
- **Graceful Degradation**: Continues operation even if Langfuse fails
- **Detailed Logging**: Logs all errors for debugging

#### **4. AgentExecutor Integration**
- **Single Chain Wrapping**: AgentExecutor wraps entire agent as one chain
- **Internal LLM Calls**: Internal LLM calls not exposed to callbacks
- **Tool Call Capture**: Tool calls captured via `on_agent_action`

### 📊 **Trace Structure in Langfuse**

When you run `web_search.py`, you'll see traces like this in the dashboard:

```
Trace: web_search_chain
├── Input: {"input": "What's the weather today?"}
├── Span: agent_action_Search
│   ├── Input: {"tool": "Search", "tool_input": "weather today"}
│   └── Output: {"output": "Weather results..."}
├── Generation: llm_call
│   ├── Input: {"prompts": ["..."], "model": "qwen3:8b"}
│   └── Output: {"text": "Based on the search results..."}
└── Output: {"output": "Final response..."}
```

### 🔍 **Why Some Callbacks Don't Fire**

With `AgentExecutor`, you might notice:

- **No `on_llm_start`/`on_llm_end`**: AgentExecutor wraps the entire agent as a single chain
- **Internal LLM calls**: Not exposed to callbacks for performance reasons
- **Tool calls via `on_agent_action`**: Tool usage captured through agent actions
- **Chain-level tracing**: Focus on high-level agent reasoning and tool usage

### 🛠️ **Customization Options**

#### **Add LLM Callbacks Directly**
```python
# Add callbacks directly to LLM for detailed tracing
llm = OllamaLLM(
    model="qwen3:8b",
    callbacks=[callback_handler]
)
```

#### **Use Different Agent Patterns**
```python
# Use different agent patterns for more detailed tracing
from langchain.agents import initialize_agent

agent = initialize_agent(
    tools, llm, agent="zero-shot-react-description",
    callbacks=[callback_handler]
)
```

#### **Custom Span Naming**
```python
# Customize span names in callback handler
span = parent.start_span(
    name=f"custom_{tool_name}_execution",
    input={"custom_input": input_str}
)
```

### 📈 **Performance Considerations**

- **Minimal Overhead**: Callbacks designed for minimal performance impact
- **Async Support**: Langfuse supports async operations for high-throughput
- **Batch Processing**: Multiple events can be batched for efficiency
- **Error Isolation**: Langfuse failures don't affect main application

### 🔮 **Future Enhancements**

- **Custom Metrics**: Add custom performance metrics
- **Cost Tracking**: Enhanced cost analysis and budgeting
- **Alerting**: Set up alerts for performance issues
- **Integration**: Connect with other observability tools

This architecture provides comprehensive observability while maintaining performance and flexibility for different use cases.

Happy experimenting! 🚀 