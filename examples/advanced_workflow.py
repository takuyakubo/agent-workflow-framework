"""
Advanced workflow example for agent workflow framework.

This example demonstrates a more complex workflow with:
1. Multiple nodes with specific instructions
2. Tool usage
3. Guardrails for output validation
4. Context sharing between nodes
"""

import logging
import re
import sys
from typing import Dict, List, Optional

from agent_workflow_framework import AgentNode, AgentWorkflow, LLM, NodeState
from agent_workflow_framework.guardrails import RegexGuardrail, SchemaGuardrail
from agent_workflow_framework.tools import JSONSchemaTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


# Define a simple mock LLM for testing
class MockLLM(LLM):
    """Mock LLM for testing purposes."""
    
    @property
    def model_name(self) -> str:
        return "mock-llm"
    
    @property
    def provider_name(self) -> str:
        return "mock"
    
    def invoke(self, prompt: str, **kwargs) -> str:
        # Return responses based on the content of the prompt
        if "research about" in prompt:
            return f"Research findings about {prompt.split('research about')[1].strip()}..."
        elif "summarize" in prompt:
            return "This is a detailed summary of the research findings..."
        elif "extract key insights" in prompt:
            return "Key insights:\n- Insight 1\n- Insight 2\n- Insight 3"
        elif "create recommendations" in prompt:
            return "Recommendations:\n1. First recommendation\n2. Second recommendation\n3. Third recommendation that contains a bad word"
        elif "format as report" in prompt:
            return """
            # Research Report
            
            ## Summary
            This is a detailed summary of the research findings...
            
            ## Key Insights
            - Insight 1
            - Insight 2
            - Insight 3
            
            ## Recommendations
            1. First recommendation
            2. Second recommendation
            3. Third recommendation
            
            ## Next Steps
            - Step 1
            - Step 2
            """
        else:
            return "Default response for: " + prompt
    
    def stream(self, prompt: str, **kwargs):
        yield self.invoke(prompt, **kwargs)
    
    def invoke_with_chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        return self.invoke(messages[-1]["content"], **kwargs)
    
    def invoke_with_tools(self, prompt: str, tools: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Mock implementation for tool calling."""
        if "search for" in prompt.lower():
            return {
                "response": "I'll use the search tool to find information.",
                "tool_calls": [
                    {
                        "name": "search",
                        "arguments": {
                            "query": prompt.split("search for")[1].strip()
                        }
                    }
                ]
            }
        return {"response": self.invoke(prompt), "tool_calls": []}


# Define state
class ResearchState(NodeState):
    """State for the research workflow."""
    topic: str = ""
    research_data: str = ""
    summary: str = ""
    key_insights: str = ""
    recommendations: str = ""
    final_report: str = ""


# Define tools
def search_tool(query: str) -> str:
    """Mock search tool."""
    return f"Search results for '{query}': Found several articles and papers discussing this topic..."


def retrieve_data_tool(dataset_id: str) -> Dict:
    """Mock data retrieval tool."""
    return {
        "dataset": dataset_id,
        "data": [
            {"id": 1, "value": "Data point 1"},
            {"id": 2, "value": "Data point 2"},
            {"id": 3, "value": "Data point 3"},
        ]
    }


# Define guardrails
content_filter_guardrail = RegexGuardrail(
    name="content_filter",
    description="Filter out inappropriate content",
    patterns=[
        {
            "pattern": r"\b(bad|inappropriate|offensive)\s+word\b",
            "action": "block",
            "message": "Inappropriate content detected"
        }
    ]
)

report_structure_guardrail = SchemaGuardrail(
    name="report_structure",
    description="Ensure the report has the required structure",
    schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "heading": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["heading", "content"]
                }
            }
        },
        "required": ["title", "sections"]
    }
)


# Define nodes
class ResearchNode(AgentNode[ResearchState]):
    """Perform research on a topic."""
    name = "researcher"
    instruction = """
    You are a research assistant. 
    
    Your task is to research about the input topic. 
    Use the search tool to find information if needed.
    
    Provide a comprehensive research summary based on the latest information available.
    """
    
    def __init__(self, llm: LLM):
        super().__init__(llm)
        # Register tools
        self.register_tool(
            JSONSchemaTool(
                name="search",
                description="Search for information on a topic",
                function=search_tool,
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            )
        )
    
    def proc(self, state: ResearchState) -> ResearchState:
        # Run with tools
        result = self.run_with_tools(f"Please research about {state.topic}")
        
        # Extract response
        state.research_data = result.get("response", "")
        
        # Add to context
        state.add_to_context("research_timestamp", "2025-05-02")
        
        return state


class SummaryNode(AgentNode[ResearchState]):
    """Create a summary from research data."""
    name = "summarizer"
    instruction = """
    You are a skilled summarizer.
    
    Your task is to create a concise yet comprehensive summary of the research data.
    Focus on the most important findings and insights.
    """
    
    def proc(self, state: ResearchState) -> ResearchState:
        # Get context
        timestamp = state.get_from_context("research_timestamp", "unknown")
        
        # Create prompt with context
        additional_context = {
            "research_timestamp": timestamp,
            "instruction": "Create a clear and concise summary"
        }
        
        # Run with instruction
        state.summary = self.run_with_instruction(state.research_data, additional_context)
        
        return state


class InsightsNode(AgentNode[ResearchState]):
    """Extract key insights from the summary."""
    name = "insights_extractor"
    instruction = """
    You are an insights specialist.
    
    Your task is to extract key insights from the summary.
    Identify the most important takeaways and implications.
    Format as a bulleted list.
    """
    
    def proc(self, state: ResearchState) -> ResearchState:
        # Run with instruction
        state.key_insights = self.run_with_instruction(state.summary)
        return state


class RecommendationsNode(AgentNode[ResearchState]):
    """Create recommendations based on insights."""
    name = "recommendations_creator"
    instruction = """
    You are a strategic advisor.
    
    Your task is to create actionable recommendations based on the key insights.
    Provide clear, specific, and implementable recommendations.
    Format as a numbered list.
    """
    
    def __init__(self, llm: LLM):
        super().__init__(llm)
        # Register guardrails
        self.register_guardrail(content_filter_guardrail)
    
    def proc(self, state: ResearchState) -> ResearchState:
        combined_input = f"Summary: {state.summary}\n\nKey Insights: {state.key_insights}"
        state.recommendations = self.run_with_instruction(combined_input)
        return state


class ReportFormatterNode(AgentNode[ResearchState]):
    """Format all content into a final report."""
    name = "report_formatter"
    instruction = """
    You are a professional report writer.
    
    Your task is to format all the information into a comprehensive report.
    Include the following sections:
    - Summary
    - Key Insights
    - Recommendations
    - Next Steps
    
    Use Markdown formatting for better readability.
    """
    
    def proc(self, state: ResearchState) -> ResearchState:
        # Combine all information
        combined_input = (
            f"Topic: {state.topic}\n\n"
            f"Summary: {state.summary}\n\n"
            f"Key Insights: {state.key_insights}\n\n"
            f"Recommendations: {state.recommendations}"
        )
        
        state.final_report = self.run_with_instruction(combined_input)
        return state


def main():
    # Create a mock LLM
    llm = MockLLM()
    
    # Create nodes
    researcher = ResearchNode(llm)
    summarizer = SummaryNode(llm)
    insights_extractor = InsightsNode(llm)
    recommendations_creator = RecommendationsNode(llm)
    report_formatter = ReportFormatterNode(llm)
    
    # Create workflow
    workflow = AgentWorkflow(
        [researcher, summarizer, insights_extractor, recommendations_creator, report_formatter],
        ResearchState
    )
    app = workflow.get_app()
    
    # Run workflow
    result = app.invoke({"topic": "Recent advances in quantum computing"})
    
    # Print the result
    print("\nWorkflow completed!")
    print(f"\nFinal report:\n{result['final_report']}")
    
    # Print context values
    print("\nContext values:")
    for key, value in result["context"].items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
