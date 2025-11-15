# colab-ai-bridge

| Pydantic AI | LangChain | DSPy |
|-------------|-----------|------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drillan/colab-ai-bridge/blob/main/examples/pydantic-ai.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drillan/colab-ai-bridge/blob/main/examples/langchain.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drillan/colab-ai-bridge/blob/main/examples/dspy.ipynb) |

Google ColabのAIモデル（Gemini, Gemma）を各種AIフレームワークで簡単に使えるようにするブリッジパッケージです。

## 概要

このパッケージは、Google ColabのAIモデルアクセスを主要なAIフレームワーク（Pydantic AI, LangChain, DSPy）と統合し、型安全なAIアプリケーション開発をColab環境で可能にします。

主な特徴：

- セットアップ不要：`import`するだけで自動的にColab環境を設定
- 3大AIフレームワーク対応（Pydantic AI, LangChain, DSPy）
- Google ColabのGemini/Gemma APIに対応

## 動作環境

- Python 3.12+
- Google Colab環境

## インストール

Google Colabのセルで次のコマンドを実行します：

### 基本インストール（コアのみ）

```bash
!pip install -q --no-warn-conflicts git+https://github.com/drillan/colab-ai-bridge
```

### フレームワークを含むインストール

#### Pydantic AIを使用する場合

```bash
!pip install -q --no-warn-conflicts "colab-ai-bridge[pydantic-ai] @ git+https://github.com/drillan/colab-ai-bridge"
```

#### LangChainを使用する場合

```bash
!pip install -q --no-warn-conflicts "colab-ai-bridge[langchain] @ git+https://github.com/drillan/colab-ai-bridge"
```

#### DSPyを使用する場合

```bash
!pip install -q --no-warn-conflicts "colab-ai-bridge[dspy] @ git+https://github.com/drillan/colab-ai-bridge"
```

#### すべてのフレームワークを含む

```bash
!pip install -q --no-warn-conflicts "colab-ai-bridge[all] @ git+https://github.com/drillan/colab-ai-bridge"
```

## 使い方

### Pydantic AI

#### 基本的な使い方

```python
from colab_ai_bridge.pydantic_ai import ColabPydanticAIModel
from pydantic_ai import Agent

# モデルの作成（セットアップは自動で完了します）
model = ColabPydanticAIModel()
agent = Agent(model)

# Agentの実行
result = agent.run_sync("フランスの首都は？")
print(result.output)
```

#### モデルの選択

利用可能なモデルを確認：

```python
from google.colab import ai

ai.list_models()
```

特定のモデルを使用：

```python
from colab_ai_bridge.pydantic_ai import ColabPydanticAIModel

# Gemini 2.5 Flash Liteを使用
model = ColabPydanticAIModel("google/gemini-2.5-flash-lite")
```

#### 構造化出力

Pydantic AIの型安全な機能を活用できます：

```python
from colab_ai_bridge.pydantic_ai import ColabPydanticAIModel
from pydantic_ai import Agent
from pydantic import BaseModel

class City(BaseModel):
    name: str
    country: str
    population: int

model = ColabPydanticAIModel()
agent = Agent(model, output_type=City)

result = agent.run_sync("東京について教えて")
city = result.output
print(f"{city.name}, {city.country}, 人口: {city.population:,}")
```

### LangChain

#### 基本的な使い方

```python
from colab_ai_bridge.langchain import ColabLangChainModel

# モデルの作成（セットアップは自動で完了します）
model = ColabLangChainModel()

# モデルの実行
response = model.invoke("フランスの首都は？")
print(response.content)
```

#### LCEL（チェーン結合）

LangChain Expression Language (LCEL) を使った宣言的なチェーン構築。複数の処理を `|` 演算子で簡潔に組み合わせることができます：

```python
from colab_ai_bridge.langchain import ColabLangChainModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ColabLangChainModel()

# 複数のチェーンを組み合わせる
summary_prompt = ChatPromptTemplate.from_template(
    "以下のテキストを3文で要約してください：\n\n{text}"
)

translate_prompt = ChatPromptTemplate.from_template(
    "以下の日本語を英語に翻訳してください：\n\n{text}"
)

# チェーン1: 要約
summary_chain = summary_prompt | model | StrOutputParser()

# チェーン2: 要約 → 翻訳
translate_chain = {"text": summary_chain} | translate_prompt | model | StrOutputParser()

# 実行
text = """
LangChainは、言語モデルを使用したアプリケーションを構築するためのフレームワークです。
プロンプトテンプレート、チェーン、エージェント、メモリなどの機能を提供し、
複雑なAIアプリケーションを簡単に構築できます。
"""

result = translate_chain.invoke({"text": text})
print(result)
```

### DSPy

#### 基本的な使い方

```python
from colab_ai_bridge.dspy import ColabDSPyLM
import dspy

# モデルの作成とDSPyの設定（セットアップは自動で完了します）
lm = ColabDSPyLM()
dspy.configure(lm=lm)

# Predictorの実行
predictor = dspy.Predict("question -> answer")
response = predictor(question="フランスの首都は？")
print(response.answer)
```

#### 構造化出力

DSPy の Signature を使用して型安全な出力を定義：

```python
from colab_ai_bridge.dspy import ColabDSPyLM
import dspy
from pydantic import BaseModel, Field

class City(BaseModel):
    name: str = Field(description="都市名")
    country: str = Field(description="国名")
    population: int = Field(description="人口")

class CitySignature(dspy.Signature):
    """都市について情報を取得する"""
    query: str = dspy.InputField(desc="都市に関する質問")
    city_info: City = dspy.OutputField(desc="都市情報")

lm = ColabDSPyLM()
dspy.configure(lm=lm)

predictor = dspy.Predict(CitySignature)
response = predictor(query="東京について教えて")
city = response.city_info
print(f"{city.name}, {city.country}, 人口: {city.population:,}")
```

#### ChainOfThought

```python
from colab_ai_bridge.dspy import ColabDSPyLM
import dspy

lm = ColabDSPyLM()
dspy.configure(lm=lm)

# ChainOfThoughtを使用して段階的に推論
cot = dspy.ChainOfThought("question -> answer")
response = cot(question="日本の四季について説明してください")
print(response.answer)
```

#### プロンプト最適化（BootstrapFewShot）

トレーニングデータから自動的にFew-shot examplesを生成し、プロンプトを最適化：

```python
from colab_ai_bridge.dspy import ColabDSPyLM
import dspy

# トレーニングデータ（質問と回答のペア）
trainset = [
    dspy.Example(question="日本の首都は？", answer="東京").with_inputs("question"),
    dspy.Example(question="フランスの首都は？", answer="パリ").with_inputs("question"),
    dspy.Example(question="イギリスの首都は？", answer="ロンドン").with_inputs("question"),
]

lm = ColabDSPyLM()
dspy.configure(lm=lm)

# メトリクス：回答が正しいかチェック
def validate_answer(example, pred, trace=None):
    return example.answer.lower() in pred.answer.lower()

# BootstrapFewShotで最適化
optimizer = dspy.BootstrapFewShot(metric=validate_answer, max_bootstrapped_demos=2)
optimized_predictor = optimizer.compile(dspy.Predict("question -> answer"), trainset=trainset)

# 最適化されたプロンプトでテスト
result = optimized_predictor(question="ドイツの首都は？")
print(f"回答: {result.answer}")
```

## 技術詳細

### 自動セットアップ

パッケージをインポートすると、以下の処理が自動的に実行されます：

1. `google.colab.ai`のインポート（環境変数`MODEL_PROXY_API_KEY`の設定）
2. `nest_asyncio.apply()`の実行（イベントループのネスト許可）

Colab環境外で使用した場合でも、エラーは発生しません（セットアップがスキップされます）。

### ColabPydanticAIModel

`ColabPydanticAIModel`は、Pydantic AIの`OpenAIChatModel`を継承し、Google ColabのModel Proxyサービス（OpenAI互換API）に対応しています。

- デフォルトモデル: `google/gemini-2.5-flash`
- 利用可能なモデル: `google/gemini-2.5-flash`, `google/gemini-2.5-flash-lite`, `google/gemma-2-9b`
- 構造化出力対応（JSON Schema経由）
- Tool Calling非対応（カスタムツールは使用不可）

## ライセンス

MIT License

## リンク

- GitHub: <https://github.com/drillan/colab-ai-bridge>
- Pydantic AI: <https://ai.pydantic.dev/>
- LangChain: <https://www.langchain.com/>
- DSPy: <https://dspy-docs.vercel.app/>
