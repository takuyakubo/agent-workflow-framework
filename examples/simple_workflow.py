"""
Simple workflow example for agent workflow framework.

This example demonstrates a basic workflow with three nodes:
1. Content extractor
2. Summarizer
3. Formatter
"""

import logging
import sys
from typing import Dict, List, Optional

from agent_workflow_framework import AgentNode, AgentWorkflow, LLM, NodeState
from agent_workflow_framework.tools import JSONSchemaTool, Tool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


# Define a simple mock LLM for testing
class MockLLM(LLM):
    def __init__(self, responses: Dict[str, str] = None):
        self.responses = responses or {}
    
    @property
    def model_name(self) -> str:
        return "mock-llm"
    
    @property
    def provider_name(self) -> str:
        return "mock"
    
    def invoke(self, prompt: str, **kwargs) -> str:
        # For simplicity, just return a hardcoded response based on the prompt
        if "Extract the main topics" in prompt:
            return "The main topics are: AI, Machine Learning, Neural Networks"
        elif "Summarize the following" in prompt:
            return "This is a summary of the topics AI, Machine Learning, and Neural Networks."
        elif "Format the following" in prompt:
            return "# Summary\n\nThis is a formatted summary of the topics AI, Machine Learning, and Neural Networks."
        else:
            return "Default response"
    
    def stream(self, prompt: str, **kwargs):
        yield self.invoke(prompt, **kwargs)
    
    def invoke_with_chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        return self.invoke(messages[-1]["content"], **kwargs)


# Define state
class DocumentState(NodeState):
    """State for the document processing workflow."""
    content: str = ""
    topics: str = ""
    summary: str = ""
    formatted_output: str = ""


# Define a simple search tool
def search_web(query: str) -> str:
    """Mock search tool."""
    return f"Search results for '{query}': AI refers to artificial intelligence, which is..."


search_tool = JSONSchemaTool(
    name="search_web",
    description="Search the web for information",
    function=search_web,
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            }
        },
        "required": ["query"]
    }
)


# Define nodes
class ContentExtractorNode(AgentNode[DocumentState]):
    """Extract main topics from the document."""
    name = "content_extractor"
    instruction = "Extract the main topics from the following content. List them as comma-separated values."
    
    def proc(self, state: DocumentState) -> DocumentState:
        topics = self.run_with_instruction(state.content)
        state.topics = topics
        return state


class SummarizerNode(AgentNode[DocumentState]):
    """Summarize the document based on extracted topics."""
    name = "summarizer"
    instruction = "Summarize the following content focusing on the extracted topics."
    
    def proc(self, state: DocumentState) -> DocumentState:
        # Use both content and topics
        context = {"topics": state.topics}
        summary = self.run_with_instruction(state.content, context)
        state.summary = summary
        return state


class FormatterNode(AgentNode[DocumentState]):
    """Format the summary into a structured output."""
    name = "formatter"
    instruction = "Format the following summary into a Markdown document with headers and bullet points."
    
    def __init__(self, llm: LLM):
        super().__init__(llm)
        self.register_tool(search_tool)
    
    def proc(self, state: DocumentState) -> DocumentState:
        formatted_output = self.run_with_instruction(state.summary)
        state.formatted_output = formatted_output
        return state


def main():
    # Create a mock LLM
    llm = MockLLM()
    
    # Create nodes
    extractor = ContentExtractorNode(llm)
    summarizer = SummarizerNode(llm)
    formatter = FormatterNode(llm)
    
    # Create workflow
    workflow = AgentWorkflow([extractor, summarizer, formatter], DocumentState)
    app = workflow.get_app()
    
    # Run workflow
    result = app.invoke({
        "content": """
        Artificial Intelligence (AI) is the simulation of human intelligence in machines.
        Machine Learning is a subset of AI that involves training models on data.
        Neural Networks are a key component of deep learning systems.
        """
    })
    
    # Print the result
    print("\nWorkflow completed!")
    print(f"Extracted topics: {result['topics']}")
    print(f"Summary: {result['summary']}")
    print("\nFormatted output:")
    print(result['formatted_output'])


if __name__ == "__main__":
    main()
