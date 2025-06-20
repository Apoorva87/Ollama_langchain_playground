# Web Frontend for Ollama LangChain Playground

A comprehensive Streamlit-based web interface for the Ollama LangChain Playground, providing an intuitive way to interact with various LLM models and run different examples.

## 🚀 Features

### 🤖 **Multi-Model Support**
- **Ollama Models**: Local models like qwen3:8b, llama3, etc.
- **OpenAI Models**: GPT-3.5-turbo, GPT-4, GPT-4-turbo
- **Anthropic Models**: Claude-3 Sonnet, Opus, Haiku
- **API Key Management**: Secure storage and configuration

### 💬 **Chat Interface**
- **Real-time Chat**: Interactive chat with LLM models
- **Chat History**: Persistent conversation history
- **Message Display**: Clear user/assistant message distinction
- **Clear History**: Easy chat history management

### 🧠 **Thinking Window**
- **Reasoning Display**: Shows content from `<think>` tags
- **Step-by-step Logic**: Visualize model reasoning process
- **Thinking Logs**: Historical thinking patterns

### 📋 **Logging System**
- **Real-time Logs**: Live console log display
- **Log History**: Scrollable log window
- **Error Tracking**: Comprehensive error logging
- **Debug Information**: Detailed debugging output

### 🔍 **Example Integration**
- **Basic Chat**: Simple LLM conversations
- **Web Search**: DuckDuckGo and Google search integration
- **Document QA**: RAG-based document question answering
- **File Upload**: Support for PDF and TXT files

### 📊 **Observability**
- **Langfuse Integration**: Full tracing and monitoring
- **Status Display**: Real-time Langfuse connection status
- **Performance Metrics**: Response times and usage tracking

## 🛠️ Setup

### 1. Install Dependencies

```bash
cd web_frontend
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the web_frontend directory:

```env
# Ollama Configuration
OLLAMA_MODEL=qwen3:8b

# Search Engine Configuration
DEFAULT_SEARCH_ENGINE=duckduckgo

# Google Search (optional)
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_google_cse_id

# Langfuse Observability (optional)
LANGFUSE_PUBLIC_KEY=pk-lf-1234567890abcdef
LANGFUSE_SECRET_KEY=sk-lf-1234567890abcdef
LANGFUSE_HOST=http://localhost:3000
```

### 3. Start Langfuse (Optional)

```bash
cd ../langfuse_setup
./manage_langfuse.sh start
```

### 4. Run the Web Frontend

```bash
cd web_frontend
streamlit run app.py
```

The application will be available at http://localhost:8501

## 🎯 Usage Guide

### **1. Model Configuration**

1. **Select Provider**: Choose from Ollama, OpenAI, or Anthropic
2. **Configure Model**: 
   - **Ollama**: Enter model name (e.g., qwen3:8b)
   - **OpenAI**: Enter API key and select model
   - **Anthropic**: Enter API key and select model
3. **Setup**: Click "Setup" button to initialize the model

### **2. Chat Interface**

1. **Select Example Type**: Choose from Basic Chat, Web Search, or Document QA
2. **Enter Message**: Type your question or prompt
3. **Upload Files**: For Document QA, upload PDF or TXT files
4. **Send**: Click "Send" to get a response

### **3. Observing Results**

- **Chat History**: View conversation in the main area
- **Thinking Window**: See model reasoning (if available)
- **Logs**: Monitor real-time logs and errors
- **Langfuse Dashboard**: Visit http://localhost:3000 for detailed traces

## 📁 File Structure

```
web_frontend/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🔧 Configuration Options

### **LLM Providers**

#### **Ollama (Local)**
- **Models**: qwen3:8b, llama3, mistral, etc.
- **Setup**: Requires Ollama to be running locally
- **Command**: `ollama serve`

#### **OpenAI (Cloud)**
- **Models**: gpt-3.5-turbo, gpt-4, gpt-4-turbo
- **Setup**: Requires OpenAI API key
- **Cost**: Pay-per-token usage

#### **Anthropic (Cloud)**
- **Models**: claude-3-sonnet, claude-3-opus, claude-3-haiku
- **Setup**: Requires Anthropic API key
- **Cost**: Pay-per-token usage

### **Search Engines**

#### **DuckDuckGo**
- **Free**: No API key required
- **Privacy**: Privacy-focused search
- **Limitations**: Rate limiting may apply

#### **Google Search**
- **Setup**: Requires API key and CSE ID
- **Features**: Advanced search capabilities
- **Cost**: Google Cloud pricing

## 🐛 Troubleshooting

### **Common Issues**

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ollama Not Running**
   ```bash
   ollama serve
   ```

3. **Model Not Found**
   ```bash
   ollama pull qwen3:8b
   ```

4. **API Key Errors**
   - Verify API keys are correct
   - Check account quotas and billing

5. **Langfuse Connection Issues**
   ```bash
   cd ../langfuse_setup
   ./manage_langfuse.sh restart
   ```

### **Logs and Debugging**

- **Application Logs**: View in the Logs window
- **Langfuse Traces**: Visit http://localhost:3000
- **Console Output**: Check terminal for additional logs

## 🎨 Customization

### **Adding New Models**

1. **Update LLMManager**: Add new provider method
2. **Update UI**: Add provider option in sidebar
3. **Test Integration**: Verify model works correctly

### **Adding New Examples**

1. **Update ExampleRunner**: Add new example method
2. **Update UI**: Add example type in dropdown
3. **Test Functionality**: Verify example works

### **Styling Changes**

1. **Modify app.py**: Update Streamlit components
2. **Add CSS**: Custom styling if needed
3. **Test Layout**: Verify responsive design

## 📈 Performance Tips

1. **Model Selection**: Choose appropriate model for task
2. **Caching**: Use Streamlit caching for repeated queries
3. **Batch Processing**: Process multiple queries efficiently
4. **Resource Management**: Monitor memory and CPU usage

## 🔒 Security Considerations

1. **API Keys**: Never commit API keys to version control
2. **Environment Variables**: Use .env files for sensitive data
3. **Input Validation**: Validate user inputs
4. **Rate Limiting**: Implement appropriate rate limits

## 🤝 Contributing

1. **Fork Repository**: Create your own fork
2. **Create Branch**: Work on feature branch
3. **Test Changes**: Verify functionality
4. **Submit PR**: Create pull request

## 📄 License

This project is open source and available under the MIT License.

## 🎯 Next Steps

1. **Explore Features**: Try different models and examples
2. **Customize Interface**: Modify UI to your needs
3. **Add Integrations**: Connect with other services
4. **Deploy**: Deploy to production environment

Happy experimenting! 🚀 