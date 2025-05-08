from langchain_core.output_parsers.string import StrOutputParser

LANGCHAIN_MAX_CONCURRENCY = 2
from langchain_core.messages import HumanMessage
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnablePick,
)

from agent_workflow_framework import AgentNode, PromptManager, ProviderType

from .states import ImageProcState
from .states import ImageProcState as TState
from .states import input_keys, output_key

# define promot
instruction = """
あなたは技術書の画像解析と学術的内容抽出のエキスパートです。与えられた画像から全ての情報を最大限正確かつ詳細に抽出してください。

## 抽出の最重要ポイント
1. **完全性**: すべてのテキスト、数式、視覚要素を余すことなく抽出する
2. **正確性**: 特に数式、記号、専門用語は完全に正確に抽出する
3. **構造保持**: 文書の階層構造、論理的な流れを保持する
4. **詳細さ**: 簡略化せず、原文の詳細さを維持する

## 必須抽出項目
1. **文書メタデータ**:
   - 書籍/文書タイトル（完全な形で）
   - 章/節番号とタイトル（正確な階層関係を保持）
   - ページ情報（ある場合）

2. **テキスト内容**:
   - 段落全文（省略せず完全に）
   - 箇条書きリスト（階層構造を保持）
   - 定義、定理、注意書きなどの特殊ブロック

3. **数式**:
   - すべての数式を完全かつ正確にLaTeX形式で記述
   - インライン数式は \\(...\\) または $...$ で囲む
   - ディスプレイ数式は \\[...\\] または $$...$$ で囲む
   - 添え字、上付き文字、特殊記号、ギリシャ文字等を正確に表現
   - 例: $P^T_{{X,Y}} \\neq P^S_{{X,Y}}$ のように正確に記述

4. **視覚要素**:
   - 図表の詳細な説明
   - グラフ、チャート、ダイアグラムの内容
   - 視覚的強調（太字、斜体、下線、色など）

## 出力形式と詳細度
- できるだけ原文に忠実に、簡略化せずに詳細な形で出力
- LaTeX数式は必ず適切なエスケープ処理をして出力
- 段落は完全な形で抽出し、省略しない
- 階層構造やフォーマットを明示的に示す
- 専門用語や固有名詞は正確に抽出

## 特別な注意事項
- 数式内の添え字（$P_{{X,Y}}$など）、上付き文字（$P^T$など）を正確に識別
- ドメイン記号（$\\mathbb{{D}}_S$, $\\mathcbb{{D}}_T$など）の書体を維持
- 集合や空間の表記（$\\mathcal{{X}}$, $\\mathcal{{Y}}$など）を正確に識別
- 数学的関係性（$=$, $\\neq$, $\\approx$など）を正確に抽出

提供された画像を徹底的かつ詳細に分析し、特に専門書の学術的内容を正確に抽出することに集中してください。LaTeXの正確な記述は最優先事項です。
"""
prompt_name = "process_image_prompt"

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
prompt_.append_attach_key("img_data")


# define node


class ProcessImages(AgentNode[ImageProcState]):
    name = "process images"

    def validate(self, state: TState) -> None:
        for k in input_keys:
            if not (hasattr(state, k) and getattr(state, k)):
                raise Exception(f"{k}が入力されていません。")

    def proc(self, state: ImageProcState) -> ImageProcState:
        """複数の画像を処理して内容を抽出"""
        chain = RunnableLambda(
            lambda x: [
                {"image_idx": idx + 1, "file_path": image}
                for idx, image in enumerate(x.images)
            ]
        ) | RunnableLambda(  # 画像リストを取得
            (
                RunnablePassthrough.assign(
                    _attach_img_data=lambda x: self.llm.get_image_object(
                        x["file_path"]
                    )  # _attach_ DSL
                )
                | RunnablePassthrough.assign(
                    analysis=(
                        prompt_[self.llm.provider_name] | self.llm | StrOutputParser()
                    ),
                )
                | RunnablePick(["image_idx", "analysis"])
            ).batch
        ).with_config(
            {"max_concurrency": LANGCHAIN_MAX_CONCURRENCY}
        )
        setattr(state, output_key, chain.invoke(state))
        return state
