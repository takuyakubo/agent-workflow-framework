import asyncio

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from agent_workflow_framework import AgentNode, PromptManager, ProviderType

from ..utils import extract_last_content_without_think, pick
from .mcp_config import mcp_config
from .states import YoutubeSummarizeState as TState
from .states import input_keys, output_key

# define promot
instruction = """
あなたはyoutubeの動画の要約をするエージェントです。
ユーザーから与えられる情報に応じて**日本語で**答えてください。
ALWAYS respond to users in Japanese. Responding in English is PROHIBITED
チェック対象の動画は {url} です。
"""
prompt_name = "youtube_summarize_prompt"

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
class YoutubeSummarize(AgentNode[TState]):
    name = "hiking check"

    def validate(self, state: TState) -> None:
        for k in input_keys:
            if not (hasattr(state, k) and getattr(state, k)):
                raise Exception(f"{k}が入力されていません。")

    async def aproc(self, state: TState) -> TState:
        async with MultiServerMCPClient(mcp_config) as mcp_client:
            chain = (
                RunnableLambda(lambda x: pick(x, input_keys))
                | prompt_
                | create_react_agent(self.llm, mcp_client.get_tools())
                | extract_last_content_without_think
            )
            setattr(state, output_key, await chain.ainvoke(state))
        return state

    def proc(self, state: TState) -> TState:
        """urlから情報を得て結果を出力"""
        return asyncio.run(self.aproc(state))
