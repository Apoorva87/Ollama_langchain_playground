import sys
import os
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Literal, cast
from dotenv import load_dotenv

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import from the project root
from config import Config
from langfuse_setup import get_langfuse_callback, get_langfuse_status

from langchain_ollama import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_community import GoogleSearchAPIWrapper
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.callbacks import FileCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler

# Load environment variables
load_dotenv()

    # Create the prompt template
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}"""



# Custom callback handler for detailed logging
class DetailedCallbackHandler(BaseCallbackHandler):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get('name', 'unknown') if serialized else 'unknown'
        self.logger.info(f"Tool started: {tool_name}")
        self.logger.debug(f"Tool input: {input_str}")
    
    def on_tool_end(self, output, **kwargs):
        self.logger.info("Tool ended")
        self.logger.debug(f"Tool output: {output}")
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.logger.info("LLM started")
        self.logger.debug(f"LLM prompts: {prompts}")
    
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        print(f"DEBUG - Response type: {type(response)}")
        print(f"DEBUG - Response content: {response}")
        self.logger.info("LLM ended")
        #self.logger.debug(f"LLM response: {response}")
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        chain_name = serialized.get('name', 'unknown') if serialized else 'unknown'
        self.logger.info(f"Chain started: {chain_name}")
        self.logger.debug(f"Chain inputs: {inputs}")
    
    def on_chain_end(self, outputs, **kwargs):
        self.logger.info("Chain ended")
        self.logger.debug(f"Chain outputs: {outputs}")

# Configure logging
def setup_logging():
    """Setup comprehensive logging for LangChain debugging."""
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / "langchain_debug.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Enable LangChain specific logging
    #logging.getLogger("langchain").setLevel(logging.DEBUG)
    #logging.getLogger("langchain.agents").setLevel(logging.DEBUG)
    #logging.getLogger("langchain.tools").setLevel(logging.DEBUG)
    #logging.getLogger("langchain.llms").setLevel(logging.DEBUG)
    
    # Create JSON log handler
    json_handler = logging.FileHandler(logs_dir / "langchain_json.log")
    json_handler.setLevel(logging.DEBUG)
    
    # Custom JSON formatter
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }
            
            # Add extra fields if they exist (using getattr for safety)
            for attr in ['tool_name', 'tool_input', 'tool_output', 'llm_input', 'llm_output']:
                value = getattr(record, attr, None)
                if value is not None:
                    log_entry[attr] = value
            
            return json.dumps(log_entry)
    
    json_handler.setFormatter(JSONFormatter())
    logging.getLogger("langchain").addHandler(json_handler)
    
    return logs_dir

# Setup logging
logs_dir = setup_logging()
logger = logging.getLogger(__name__)

SearchEngine = Literal["duckduckgo", "google"]

def get_current_datetime(*args, **kwargs) -> str:
    """Get the current date and time from the system.
    
    Args:
        *args: Ignored arguments (for LangChain compatibility)
        **kwargs: Ignored keyword arguments (for LangChain compatibility)
        
    Returns:
        A string containing the current date and time in a readable format.
    """
    logger.debug(f"get_current_datetime called with args: {args}, kwargs: {kwargs}")
    now = datetime.now()
    result = now.strftime("%Y-%m-%d %H:%M:%S %Z")
    logger.debug(f"get_current_datetime returning: {result}")
    return result

def get_search_tool(engine: SearchEngine = "duckduckgo") -> Tool:
    """Get the appropriate search tool based on the selected engine.
    
    Args:
        engine: The search engine to use ("duckduckgo" or "google")
        
    Returns:
        A Tool instance configured for the selected search engine
        
    Raises:
        ValueError: If an invalid search engine is specified
        RuntimeError: If Google API key or CSE ID is missing when using Google search
    """
    logger.debug(f"Creating search tool for engine: {engine}")
    
    if engine == "duckduckgo":
        search = DuckDuckGoSearchRun()
        tool = Tool(
            name="Search",
            func=search.run,
            description="Useful for searching the web for current information using DuckDuckGo. Input should be a search query."
        )
        logger.debug("Created DuckDuckGo search tool")
        return tool
    elif engine == "google":
        # Check for Google API key and CSE ID
        if not Config.GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY environment variable is required for Google search")
        if not Config.GOOGLE_CSE_ID:
            raise RuntimeError("GOOGLE_CSE_ID environment variable is required for Google search")
        
        # Set environment variables for the wrapper
        os.environ["GOOGLE_API_KEY"] = Config.GOOGLE_API_KEY
        os.environ["GOOGLE_CSE_ID"] = Config.GOOGLE_CSE_ID
        
        search = GoogleSearchAPIWrapper()
        tool = Tool(
            name="Search",
            func=search.run,
            description="Useful for searching the web for current information using Google. Input should be a search query."
        )
        logger.debug("Created Google search tool")
        return tool
    else:
        raise ValueError(f"Unsupported search engine: {engine}")

def run_web_search(query: str, engine: SearchEngine = "duckduckgo") -> str:
    """Run a web search and get an answer based on search results.
    
    Args:
        query: The question or query to search for
        engine: The search engine to use ("duckduckgo" or "google")
        
    Returns:
        The model's answer based on search results
        
    Raises:
        ValueError: If the query is empty or contains only whitespace
        RuntimeError: If Google API key or CSE ID is missing when using Google search
    """
    logger.info(f"Starting web search with query: '{query}', engine: {engine}")
    
    # Validate input
    if not query or not query.strip():
        raise ValueError("Query cannot be empty or contain only whitespace")
    
    # Create callbacks for detailed logging and Langfuse tracing
    callbacks: List[BaseCallbackHandler] = [DetailedCallbackHandler(logger)]
    # Add Langfuse callback if available
    langfuse_handler = get_langfuse_callback()
    if langfuse_handler and isinstance(langfuse_handler, BaseCallbackHandler):
        callbacks.append(langfuse_handler)
        logger.info("Langfuse tracing enabled")
    else:
        logger.info("Langfuse tracing not available")
    
    
    # Initialize Ollama
    logger.debug(f"Initializing Ollama with model: {Config.OLLAMA_MODEL}")
    llm = OllamaLLM(model=Config.OLLAMA_MODEL, callbacks=callbacks)
    
    # Get search tool and datetime tool
    search_tool = get_search_tool(engine)
    datetime_tool = Tool(
        name="GetCurrentDateTime",
        func=get_current_datetime,
        description="Useful for getting the current date and time from the system. Use this when you need to know what day it is, what time it is, or any temporal information. No input is needed. The function does not take any arguments."
    )
    
    tools = [search_tool, datetime_tool]
    logger.debug(f"Created {len(tools)} tools: {[tool.name for tool in tools]}")
    
    global template
    prompt = PromptTemplate.from_template(template)
    
    
    
    # Create the agent
    logger.debug("Creating ReAct agent")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        callbacks=callbacks
    )
    
    # Run the agent
    logger.info("Executing agent with query")
    result = agent_executor.invoke({"input": query})
    
    # Validate response
    if not result or "output" not in result:
        raise ValueError("Invalid response from agent")
    
    response = result["output"]
    if not response or not response.strip():
        raise ValueError("Empty response from agent")
    
    logger.info(f"Agent execution completed. Response length: {len(response)}")
    return response

def main():
    print("Web Search System (type 'quit' to exit)")
    print("Available search engines: duckduckgo, google")
    print("Additional tools: current date/time")
    print(f"Logs will be saved to: {logs_dir}")
    
    # Get Langfuse status
    langfuse_status = get_langfuse_status()
    if langfuse_status["enabled"]:
        print("Langfuse tracing: ENABLED")
        print(f"Dashboard: {langfuse_status['host']}")
    else:
        print("Langfuse tracing: DISABLED (set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable)")
    print("-" * 50)
    
    # Get search engine preference
    if False:
        while True:
            engine_input = input("\nSelect search engine (duckduckgo/google): ").lower()
            if engine_input in ["duckduckgo", "google"]:
                engine = cast(SearchEngine, engine_input)
                break
            print("Invalid choice. Please select 'duckduckgo' or 'google'")
    else:
        engine = "duckduckgo"
    
    print(f"\nUsing {engine} search engine")
    print("-" * 50)
    
    while True:
        try:
            query = input("\nYour question: ")
            if query.lower() == 'quit':
                break
                
            print("\nSearching and analyzing...")
            print("-" * 30)
            response = run_web_search(query, engine)
            print("\nAnswer:", response)
            print("-" * 50)
        except ValueError as e:
            print(f"\nError: {str(e)}")
            print("-" * 50)
        except RuntimeError as e:
            print(f"\nError: {str(e)}")
            if engine == "google":
                print("Please ensure both GOOGLE_API_KEY and GOOGLE_CSE_ID are set in your .env file")
            print("-" * 50)
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            print("-" * 50)

if __name__ == "__main__":
    main() 