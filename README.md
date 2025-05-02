# Agent Workflow Framework

A framework for agent-based workflow nodes with instruction, tools, and guardrails properties.

## Overview

This framework enhances the traditional workflow node structure by giving nodes agent-like capabilities. Each node can:

- Have specific instructions
- Use defined tools
- Apply guardrails for safety and quality control
- Configure context inheritance between nodes

## Key Features

- **Agent-like Nodes**: Nodes function more like agents with specific instructions and capabilities
- **Tool Integration**: Easily define and use tools within nodes
- **Guardrails**: Apply safety and quality controls to node outputs
- **Context Control**: Flexible configuration for context inheritance between nodes
- **Unified Interface**: Consistent API across different LLM providers

## Installation

```bash
pip install agent-workflow-framework
```

## Basic Usage

```python
from agent_workflow_framework import AgentNode, AgentWorkflow, NodeState

# Define a simple state
class MyState(NodeState):
    data: str = ""
    processed_data: str = ""

# Create an agent node
class DataProcessor(AgentNode[MyState]):
    name = "data_processor"
    instruction = "Process the input data by summarizing it."
    
    def proc(self, state: MyState) -> MyState:
        # Process will use the node's instruction with the LLM
        state.processed_data = self.run_with_instruction(state.data)
        return state

# Create a workflow
workflow = AgentWorkflow([DataProcessor(llm)], MyState)
app = workflow.get_app()

# Run the workflow
result = app.invoke({"data": "This is some text that needs processing."})
```

## License

MIT