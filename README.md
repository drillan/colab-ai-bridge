# colab-pydantic-ai

Google ColabでPydantic AIを簡単に使えるようにする統合パッケージです。

## 概要

このパッケージは、Google ColabのGemini APIアクセスとPydantic AIを統合し、型安全なAIアプリケーション開発をColab環境で可能にします。

主な特徴：

- セットアップ不要：`import`するだけで自動的にColab環境を設定
- Pydantic AIのすべての機能が利用可能
- Google ColabのGemini APIに対応した`ColabGeminiModel`を提供

## 動作環境

- Python 3.12+
- Google Colab環境

## インストール

Google Colabのセルで次のコマンドを実行します：

```bash
!pip install -q --no-warn-conflicts git+https://github.com/drillan/colab-pydantic-ai
```

## 使い方

### 基本的な使い方

```python
from colab_pydantic_ai import ColabGeminiModel
from pydantic_ai import Agent

# モデルの作成（セットアップは自動で完了します）
model = ColabGeminiModel()
agent = Agent(model)

# Agentの実行
result = agent.run_sync("フランスの首都は？")
print(result.output)
```

### モデルの選択

利用可能なモデルを確認：

```python
from google.colab import ai

ai.list_models()
```

特定のモデルを使用：

```python
from colab_pydantic_ai import ColabGeminiModel

# Gemini 2.5 Flash Liteを使用
model = ColabGeminiModel("google/gemini-2.5-flash-lite")
```

### 構造化出力

Pydantic AIの型安全な機能を活用できます：

```python
from colab_pydantic_ai import ColabGeminiModel
from pydantic_ai import Agent
from pydantic import BaseModel

class City(BaseModel):
    name: str
    country: str
    population: int

model = ColabGeminiModel()
agent = Agent(model, output_type=City)

result = agent.run_sync("東京について教えて")
city = result.output
print(f"{city.name}, {city.country}, 人口: {city.population:,}")
```

## 技術詳細

### 自動セットアップ

パッケージをインポートすると、以下の処理が自動的に実行されます：

1. `google.colab.ai`のインポート（環境変数`MODEL_PROXY_API_KEY`の設定）
2. `nest_asyncio.apply()`の実行（イベントループのネスト許可）

Colab環境外で使用した場合でも、エラーは発生しません（セットアップがスキップされます）。

### ColabGeminiModel

`ColabGeminiModel`は、Pydantic AIの`OpenAIChatModel`を継承し、Google ColabのGemini APIプロキシに対応しています。

- デフォルトモデル: `google/gemini-2.5-flash`
- 利用可能なモデル: `google/gemini-2.5-flash`, `google/gemini-2.5-flash-lite`
- 構造化出力対応（JSON Schema経由）
- Tool Calling非対応（カスタムツールは使用不可）

## ライセンス

MIT License

## リンク

- GitHub: https://github.com/drillan/colab-pydantic-ai
- Pydantic AI: https://ai.pydantic.dev/
