from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import create_react_agent

from agent_workflow_framework import AgentNode, PromptManager, ProviderType

from ..utils import extract_last_content_without_think, pick
from .states import HikingCheckState as TState
from .states import input_keys, output_key
from .tools import tool_repository

# define promot
instruction = """
あなたはハイキングに天気と温度が適切かを考えるエージェントです。
ユーザーから与えられる情報に応じて答えてください。
チェック対象の都市は{city}です。
"""

prompt_name = "hiking_check_prompt"

prompt_ = PromptManager(prompt_name)
content = [
    HumanMessage(
        content=[
            {
                "type": "text",
                "text": instruction,
            }
        ]
    )
]
prompt_[ProviderType.GOOGLE.value] = content


# define node
class HikingCheck(AgentNode[TState]):
    name = "hiking check"

    def validate(self, state: TState) -> None:
        for k in input_keys:
            if not (hasattr(state, k) and getattr(state, k)):
                raise Exception(f"{k}が入力されていません。")

    def proc(self, state: TState) -> TState:
        """都市名から情報を得て結果を出力"""
        chain = (
            RunnableLambda(lambda x: pick(x, input_keys))
            | prompt_
            | create_react_agent(self.llm, tool_repository.values())
            | extract_last_content_without_think
        )
        setattr(state, output_key, chain.invoke(state))
        return state
