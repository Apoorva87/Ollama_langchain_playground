# Langfuse Setup & Integration

This directory contains all Langfuse-related configuration and management files for the LangChain web search application. Langfuse is an open-source LLM observability platform that helps you trace, monitor, debug, and analyze your LLM applications.

## 📁 Directory Contents

- **`docker-compose.yml`** - Docker Compose configuration for Langfuse and PostgreSQL
- **`langfuse_config.py`** - Python configuration module for Langfuse integration
- **`manage_langfuse.sh`** - Management script for Langfuse operations
- **`setup_langfuse_local.sh`** - Initial setup script for Langfuse
- **`__init__.py`** - Python package initialization

## 🚀 Quick Start

### 1. Start Langfuse

```bash
cd langfuse_setup
./manage_langfuse.sh start
```

### 2. Check Status

```bash
./manage_langfuse.sh status
```

### 3. View Dashboard

Open http://localhost:3000 in your browser

## 🛠️ Management Commands

```bash
# Start Langfuse
./manage_langfuse.sh start

# Stop Langfuse
./manage_langfuse.sh stop

# Restart Langfuse
./manage_langfuse.sh restart

# Check status
./manage_langfuse.sh status

# View logs
./manage_langfuse.sh logs

# Clean up all data (DESTRUCTIVE)
./manage_langfuse.sh cleanup

# Show help
./manage_langfuse.sh help
```

## 🔧 Configuration

### Environment Variables

Add these to your `.env` file:

```env
# Langfuse Configuration
LANGFUSE_PUBLIC_KEY=pk-lf-1234567890abcdef
LANGFUSE_SECRET_KEY=sk-lf-1234567890abcdef
LANGFUSE_HOST=http://localhost:3000
```

### Python Integration

The `langfuse_config.py` module provides:

```python
from langfuse_setup import get_langfuse_callback, get_langfuse_status

# Get callback handler for LangChain
callback = get_langfuse_callback()

# Check Langfuse status
status = get_langfuse_status()
```

## 📊 What Gets Traced

The integration automatically traces:

### 🤖 **LLM Interactions**
- Model prompts and responses
- Token usage and costs
- Response times
- Model performance metrics

### 🛠️ **Tool Executions**
- Search tool calls (DuckDuckGo/Google)
- DateTime tool calls
- Input/output data
- Tool execution times

### 🔄 **Agent Reasoning**
- Thought processes
- Action selections
- Chain of reasoning
- Agent decision flows

### 📊 **Performance Metrics**
- Execution time per step
- Success/failure rates
- Resource usage
- Cost tracking

## 🎯 Dashboard Features

Once Langfuse is running, visit **http://localhost:3000** to access:

### 🔍 **Traces**
- Complete execution traces
- Tool calls and LLM interactions
- Performance analysis
- Error tracking

### 📈 **Metrics**
- Token usage and costs
- Response times
- Success/failure rates
- User interaction patterns

### 🐛 **Debugging**
- Detailed error logs
- Input/output inspection
- Step-by-step execution flow
- Performance bottlenecks

## 📝 Example Trace Structure

```
Query: "What's the weather today?"
├── LLM: Initial prompt processing
├── Tool: GetCurrentDateTime
│   └── Output: "2025-01-27 14:30:45 EST"
├── Tool: Search (DuckDuckGo)
│   └── Output: "Weather results..."
└── LLM: Final response generation
```

## 🔍 Troubleshooting

### Common Issues

1. **Port 3000 already in use**
   ```bash
   # Change port in docker-compose.yml
   ports:
     - "3001:3000"  # Use port 3001 instead
   ```

2. **Database connection issues**
   ```bash
   # Restart PostgreSQL
   docker-compose restart postgres
   ```

3. **API key errors**
   - Ensure keys are correctly set in `.env`
   - Check that Langfuse is running: `docker-compose ps`

4. **Permission errors**
   ```bash
   chmod +x manage_langfuse.sh
   ```

### Logs and Debugging

```bash
# View all logs
./manage_langfuse.sh logs

# View specific service logs
docker-compose logs langfuse
docker-compose logs postgres

# Follow logs in real-time
docker-compose logs -f
```

## 📚 Advanced Configuration

### Custom API Keys

Generate your own keys:

```bash
# Generate a random secret
openssl rand -hex 32

# Update docker-compose.yml with your keys
LANGFUSE_SECRET_KEY: your-generated-secret
LANGFUSE_PUBLIC_KEY: pk-lf-your-public-key
```

### Production Setup

For production, consider:
- Using a managed PostgreSQL database
- Setting up proper SSL certificates
- Configuring authentication
- Setting up monitoring and alerts

## 🎉 Usage Examples

### Basic Integration

```python
from langfuse_setup import get_langfuse_callback
from langchain.agents import AgentExecutor

# Get Langfuse callback handler
callback_handler = get_langfuse_callback()

# Add to your agent
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    callbacks=[callback_handler] if callback_handler else []
)
```

### Status Checking

```python
from langfuse_setup import get_langfuse_status

# Check if Langfuse is properly configured
status = get_langfuse_status()
print(f"Langfuse enabled: {status['enabled']}")
print(f"Host: {status['host']}")
```

## 📖 Documentation

- **Langfuse Docs**: https://langfuse.com/docs
- **LangChain Integration**: https://langfuse.com/docs/integrations/langchain
- **GitHub Repository**: https://github.com/langfuse/langfuse

## 🎯 Next Steps

1. **Start Langfuse**: `./manage_langfuse.sh start`
2. **Configure environment variables** in your `.env` file
3. **Run your application**: `python3 examples/web_search.py`
4. **Explore the dashboard** at http://localhost:3000
5. **Analyze traces and metrics** to optimize performance

Happy tracing! 🚀 