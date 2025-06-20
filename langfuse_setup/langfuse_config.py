"""
Langfuse Configuration Module

This module provides Langfuse setup and configuration functions
for the LangChain web search application.
"""

import os
import logging
from typing import Optional, Any, Dict, List, Union
from langfuse import Langfuse
from langchain.callbacks.base import BaseCallbackHandler

logger = logging.getLogger(__name__)

class LangfuseCallbackHandler(BaseCallbackHandler):
    """Custom Langfuse callback handler for LangChain integration."""
    
    def __init__(self, public_key: str, secret_key: str, host: str = "http://localhost:3000"):
        super().__init__()
        self.langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host
        )
        self.current_trace = None
        self.current_span = None
        self.active_spans = {}  # Track active spans by their IDs
        logger.info("Langfuse callback handler initialized")
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        """Log the start of a chain."""
        try:
            if not self.current_trace:
                # Create a new trace
                self.current_trace = self.langfuse.start_span(
                    name="web_search_chain",
                    input=inputs
                )
                logger.info("Started new Langfuse trace")
            else:
                # Create a child span
                self.current_span = self.current_trace.start_span(
                    name=serialized.get('name', 'chain') if serialized else 'chain',
                    input=inputs
                )
                logger.info(f"Started chain span: {serialized.get('name', 'chain') if serialized else 'chain'}")
        except Exception as e:
            logger.error(f"Error in on_chain_start: {e}")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Log the end of a chain."""
        try:
            if self.current_span:
                self.current_span.update(output=outputs)
                self.current_span.end()
                self.current_span = None
                logger.info("Ended chain span")
            elif self.current_trace:
                self.current_trace.update(output=outputs)
                self.current_trace.end()
                self.current_trace = None
                logger.info("Ended main trace")
        except Exception as e:
            logger.error(f"Error in on_chain_end: {e}")
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Log the start of an LLM call."""
        try:
            parent = self.current_span or self.current_trace
            if parent:
                generation = parent.start_generation(
                    name="llm_call",
                    input={"prompts": prompts},
                    model=serialized.get("name", "unknown") if serialized else "unknown"
                )
                # Store the generation span
                span_id = id(generation)
                self.active_spans[span_id] = generation
                logger.info("Started LLM generation span")
        except Exception as e:
            logger.error(f"Error in on_llm_start: {e}")
    
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Log the end of an LLM call."""
        try:
            # Find and end the most recent LLM generation
            for span_id, span in list(self.active_spans.items()):
                if hasattr(span, 'update'):  # It's a generation span
                    # Extract just the text content from the response
                    if hasattr(response, 'generations') and response.generations:
                        # Get the actual text content from the first generation
                        text_content = response.generations[0][0].text if response.generations[0] else ""
                        span.update(output={"text": text_content})
                    else:
                        # Fallback to full response if structure is different
                        span.update(output=dict(response.to_json()))
                    span.end()
                    del self.active_spans[span_id]
                    logger.info("Ended LLM generation span")
                    break
        except Exception as e:
            logger.error(f"Error in on_llm_end: {e}")
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """Log the start of a tool call."""
        try:
            parent = self.current_span or self.current_trace
            if parent:
                tool_name = serialized.get('name', 'unknown') if serialized else 'unknown'
                span = parent.start_span(
                    name=f"tool_{tool_name}",
                    input={"input": input_str}
                )
                # Store the span
                span_id = id(span)
                self.active_spans[span_id] = span
                logger.info(f"Started tool span: {tool_name}")
        except Exception as e:
            logger.error(f"Error in on_tool_start: {e}")
    
    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Log the end of a tool call."""
        try:
            # Find and end the most recent tool span
            for span_id, span in list(self.active_spans.items()):
                if not hasattr(span, 'update'):  # It's a regular span (tool)
                    span.update(output=output)
                    span.end()
                    del self.active_spans[span_id]
                    logger.info("Ended tool span")
                    break
        except Exception as e:
            logger.error(f"Error in on_tool_end: {e}")
    
    def on_agent_action(self, action, **kwargs):
        """Log agent actions (ReAct format)."""
        try:
            parent = self.current_span or self.current_trace
            if parent:
                # Support both dict and object
                if hasattr(action, 'tool'):
                    tool_name = getattr(action, 'tool', 'unknown')
                    tool_input = getattr(action, 'tool_input', '')
                else:
                    tool_name = action.get('tool', 'unknown')
                    tool_input = action.get('tool_input', '')
                
                span = parent.start_span(
                    name=f"agent_action_{tool_name}",
                    input={"tool": tool_name, 
                           "tool_input": tool_input,
                           "action": dict(action.to_json())}
                )
                # Store the span
                span_id = id(span)
                self.active_spans[span_id] = span
                logger.info(f"Started agent action span: {tool_name}")
        except Exception as e:
            logger.error(f"Error in on_agent_action: {e}")
    
    def on_agent_finish(self, finish, **kwargs):
        """Log agent finish."""
        try:
            # Support both dict and object
            if hasattr(finish, 'return_values'):
                output = finish.return_values
            elif hasattr(finish, 'output'):
                output = finish.output
            else:
                output = dict(finish.to_json())
            
            # Find and end the most recent agent action span
            for span_id, span in list(self.active_spans.items()):
                if not hasattr(span, 'update'):  # It's a regular span
                    span.update(output=output)
                    span.end()
                    del self.active_spans[span_id]
                    logger.info("Ended agent action span")
                    break
        except Exception as e:
            logger.error(f"Error in on_agent_finish: {e}")
    
    def on_chain_error(self, error: Union[str, Exception], **kwargs: Any) -> None:
        """Log chain errors."""
        try:
            if self.current_span:
                self.current_span.update(status_message=str(error))
                self.current_span.end()
                self.current_span = None
            elif self.current_trace:
                self.current_trace.update(status_message=str(error))
                self.current_trace.end()
                self.current_trace = None
            logger.error(f"Chain error: {error}")
        except Exception as e:
            logger.error(f"Error in on_chain_error: {e}")
    
    def on_llm_error(self, error: Union[str, Exception], **kwargs: Any) -> None:
        """Log LLM errors."""
        try:
            # End any active spans
            for span_id, span in list(self.active_spans.items()):
                span.update(status_message=str(error))
                span.end()
                del self.active_spans[span_id]
            logger.error(f"LLM error: {error}")
        except Exception as e:
            logger.error(f"Error in on_llm_error: {e}")
    
    def on_tool_error(self, error: Union[str, Exception], **kwargs: Any) -> None:
        """Log tool errors."""
        try:
            # End any active spans
            for span_id, span in list(self.active_spans.items()):
                span.update(status_message=str(error))
                span.end()
                del self.active_spans[span_id]
            logger.error(f"Tool error: {error}")
        except Exception as e:
            logger.error(f"Error in on_tool_error: {e}")


class LangfuseConfig:
    """Configuration class for Langfuse integration."""
    
    def __init__(self):
        self.public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        self.host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
        self.enabled = bool(self.public_key and self.secret_key)
    
    def get_langfuse_instance(self) -> Optional[Langfuse]:
        """Get a Langfuse instance if credentials are available."""
        if not self.enabled:
            logger.warning("Langfuse credentials not found. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY for tracing.")
            return None
        
        try:
            langfuse = Langfuse(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host
            )
            logger.info(f"Langfuse initialized with host: {self.host}")
            return langfuse
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse: {e}")
            return None
    
    def get_callback_handler(self) -> Optional[LangfuseCallbackHandler]:
        """Get a Langfuse callback handler for LangChain integration."""
        if not self.enabled:
            logger.warning("Langfuse credentials not found. Callback handler not available.")
            return None
        
        try:
            # Create custom Langfuse callback handler for LangChain
            if self.public_key and self.secret_key:
                callback_handler = LangfuseCallbackHandler(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.host
                )
                logger.info("Langfuse callback handler created successfully")
                return callback_handler
            else:
                logger.error("Missing Langfuse credentials")
                return None
        except Exception as e:
            logger.error(f"Failed to create Langfuse callback handler: {e}")
            return None
    
    def get_status_info(self) -> dict:
        """Get status information about Langfuse configuration."""
        return {
            "enabled": self.enabled,
            "host": self.host,
            "public_key_set": bool(self.public_key),
            "secret_key_set": bool(self.secret_key)
        }

# Global instance
langfuse_config = LangfuseConfig()

def setup_langfuse() -> Optional[Langfuse]:
    """Setup Langfuse for tracing and observability."""
    return langfuse_config.get_langfuse_instance()

def get_langfuse_callback() -> Optional[LangfuseCallbackHandler]:
    """Get Langfuse callback handler for LangChain integration."""
    return langfuse_config.get_callback_handler()

def get_langfuse_status() -> dict:
    """Get Langfuse configuration status."""
    return langfuse_config.get_status_info() 