"""
Agent Workflow Framework
-----------------------

A framework for creating agent-like workflow nodes with instructions, tools, and guardrails.
"""

from agent_workflow_framework.core.node import AgentNode
from agent_workflow_framework.core.state import NodeState
from agent_workflow_framework.core.workflow import AgentWorkflow
from agent_workflow_framework.llm.base import LLM

__all__ = [
    "AgentNode",
    "NodeState",
    "AgentWorkflow",
    "LLM",
]

__version__ = "0.1.0"
