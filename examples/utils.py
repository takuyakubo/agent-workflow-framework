import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.passthrough import RunnablePick


def pick(state, key_list):
    return {k: getattr(state, k) for k in key_list}


extract_last_content_without_think = (
    RunnablePick("messages")
    | RunnableLambda(lambda msgs: msgs[-1])
    | StrOutputParser()
    | RunnableLambda(
        lambda text: re.search(
            r"(?:<think>.*</think>)?(.*)", text, flags=re.MULTILINE | re.DOTALL
        )
        .group(1)
        .strip()
    )
)
