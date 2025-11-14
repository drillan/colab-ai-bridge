"""Pydantic AI × Google Colab AI 統合モジュール

Google ColabのGeminiアクセスを Pydantic AI から利用可能にします。

Examples:
    ```python
    from colab_gemini_model import ColabGeminiModel
    from pydantic_ai import Agent

    # 基本的な使用
    model = ColabGeminiModel()
    agent = Agent(model)
    result = agent.run_sync("フランスの首都は？")
    print(result.output)
    ```

Notes:
    - Google Colab環境でのみ動作します
    - MODEL_PROXY_API_KEY が必要（Colab Secretsに設定）
    - OpenAI Chat Completions API互換のため、Pydantic AIの全機能が利用可能
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, cast

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings


def _is_colab_environment() -> bool:
    """Google Colab環境かチェック

    Returns:
        True if running in Google Colab, False otherwise
    """
    try:
        import google.colab  # noqa: F401

        return any(
            [
                "COLAB_GPU" in os.environ,
                "COLAB_TPU_ADDR" in os.environ,
                "COLAB_JUPYTER_IP" in os.environ,
                "COLAB_RELEASE_TAG" in os.environ,
            ]
        )
    except ImportError:
        return False


@dataclass(kw_only=True)
class ColabGeminiModelProfile(OpenAIModelProfile):
    """Google Colab Model Proxy Service 向けのプロファイル

    Model Proxy Service の機能サポート状況:
    - ✅ Structured Outputs (response_format + JSON Schema)
    - ✅ Temperature パラメータ
    - ✅ Streaming
    - ❌ Tool Calling (完全非対応)
    - ❌ JSON Mode (レスポンス形式が異なる)

    このプロファイルは default_structured_output_mode='native' に設定し、
    Tool Calling を回避して Structured Outputs を使用します。
    """

    # 構造化出力: Structured Outputs を使用（Tool Calling を回避）
    default_structured_output_mode: str = "native"  # type: ignore

    # Model Proxy Service は JSON Schema をサポート
    supports_json_schema_output: bool = True

    # Tool Calling は非対応
    supports_tools: bool = False

    # JSON Mode は部分的対応（レスポンス形式が異なるため無効化）
    supports_json_object_output: bool = False

    # JSON Schema トランスフォーマー（OpenAI互換）
    json_schema_transformer: type = OpenAIJsonSchemaTransformer  # type: ignore


def _get_model_proxy_credentials() -> tuple[str, str]:
    """Model Proxy認証情報を取得

    google.colab.ai の内部実装と同じロジックを使用します。

    Returns:
        (base_url, api_key) のタプル

    Raises:
        RuntimeError: Colab環境でない、または認証情報が未設定
    """
    if not _is_colab_environment():
        raise RuntimeError(
            "ColabGeminiModel は Google Colab 環境でのみ動作します。\n"
            "ローカル環境では GoogleModel または OpenAIChatModel を使用してください。\n\n"
            "例:\n"
            "  from pydantic_ai.models.google import GoogleModel\n"
            "  model = GoogleModel('gemini-2.5-flash')"
        )

    # Model Proxy Host (Colab環境で自動設定される)
    model_proxy_host = os.environ.get("MODEL_PROXY_HOST")
    if not model_proxy_host:
        raise RuntimeError(
            "MODEL_PROXY_HOST 環境変数が設定されていません。\n"
            "これは Google Colab 環境で自動的に設定されるはずです。\n"
            "Colab ランタイムを再起動してください。"
        )

    # Model Proxy API Key (google.colab.ai のインポートで自動設定)
    if "MODEL_PROXY_API_KEY" not in os.environ:
        try:
            # google.colab.ai をインポートすると自動的に
            # MODEL_PROXY_API_KEY が環境変数に設定される
            from google.colab import ai  # noqa: F401

            # 念のため再確認
            if "MODEL_PROXY_API_KEY" not in os.environ:
                raise RuntimeError(
                    "MODEL_PROXY_API_KEY が自動設定されませんでした。\n"
                    "google.colab.ai モジュールのバージョンを確認してください。"
                )
        except ImportError as e:
            raise RuntimeError(
                "google.colab.ai モジュールがインポートできません。\n"
                "Colab環境で実行していることを確認してください。"
            ) from e

    base_url = f"{model_proxy_host}/models/openapi"
    api_key = os.environ["MODEL_PROXY_API_KEY"]

    return base_url, api_key


class ColabGeminiModel(OpenAIChatModel):
    """Google Colab Geminiアクセス用のPydantic AI Model

    google.colab.aiの内部実装（OpenAI互換プロキシ）を活用し、
    Pydantic AIのフル機能（Tools, Streaming等）をサポートします。

    このクラスは OpenAIChatModel を継承しているため、以下の機能が利用可能です:
    - Function Calling
    - Streaming responses
    - 構造化出力
    - Vision (画像入力)
    - Web Search Tool (Model Proxyがサポートしている場合)

    Examples:
        基本的な使用:
        ```python
        from colab_gemini_model import ColabGeminiModel
        from pydantic_ai import Agent

        model = ColabGeminiModel()
        agent = Agent(model)
        result = agent.run_sync("フランスの首都は？")
        print(result.output)
        ```

        Web Search Tool付き (要検証):
        ```python
        from colab_gemini_model import ColabGeminiModel
        from pydantic_ai import Agent
        from pydantic_ai.builtin_tools import WebSearchTool

        model = ColabGeminiModel("google/gemini-2.5-flash")
        agent = Agent(model, builtin_tools=[WebSearchTool()])
        result = agent.run_sync("2025年の最新AI技術を調査")
        print(result.output)
        ```

        カスタム設定:
        ```python
        model = ColabGeminiModel(
            "google/gemini-2.5-flash",
            settings={
                'temperature': 0.7,
                'max_tokens': 2000,
            }
        )
        ```

    Supported Models:
        google.colab.ai.list_models() で取得可能なモデル:
        - google/gemini-2.5-flash (デフォルト)
        - google/gemini-2.5-flash-lite

    Requirements:
        - Google Colab環境
        - MODEL_PROXY_API_KEY（Colab Secretsに設定）
        - pydantic-ai パッケージ

    Notes:
        - OpenAI Chat Completions API互換のため、OpenAIChatModelの
          基本機能が利用可能です
        - ✅ **構造化出力対応**: output_type で Pydantic モデルを指定可能
          （Structured Outputs 経由で動作、Tool Calling は使用しない）
        - ✅ Temperature パラメータ完全対応
        - ❌ Tool Calling 非対応（カスタムツール、Web Search Tool等は使用不可）
        - Colab環境外では RuntimeError が発生します
    """

    def __init__(
        self,
        model_name: str = "google/gemini-2.5-flash",
        *,
        settings: ModelSettings | None = None,
        profile: Any | None = None,
    ):
        """初期化

        Args:
            model_name: Geminiモデル名
                例: "google/gemini-2.5-flash", "google/gemini-2.5-flash-lite"
            settings: モデル設定（temperature, max_tokens等）
            profile: モデルプロファイル（高度な設定）
                None の場合、ColabGeminiModelProfile() が使用されます。
                これにより、構造化出力が Structured Outputs 経由で動作します。

        Raises:
            RuntimeError: Colab環境でない、または認証情報が未設定
        """
        base_url, api_key = _get_model_proxy_credentials()

        # Profile が指定されていない場合、Colab 向けのプロファイルを使用
        if profile is None:
            profile = ColabGeminiModelProfile()

        # OpenAIProvider を作成
        provider = OpenAIProvider(
            base_url=base_url,
            api_key=api_key,
        )

        # OpenAIChatModel を初期化（provider を使用）
        super().__init__(
            model_name=model_name,
            provider=provider,
            settings=settings,
            profile=profile,
        )

    @property
    def system(self) -> str:
        """OpenTelemetry用のシステム名

        Returns:
            "google-colab-gemini"
        """
        return "google-colab-gemini"


def get_colab_gemini_model(
    model_name: str = "google/gemini-2.5-flash",
    **kwargs: Any,
) -> ColabGeminiModel:
    """Google Colab Geminiモデルを取得（ヘルパー関数）

    Args:
        model_name: Geminiモデル名
        **kwargs: ColabGeminiModelに渡す追加引数

    Returns:
        Pydantic AI用のColabGeminiModel

    Examples:
        ```python
        model = get_colab_gemini_model()
        agent = Agent(model)
        ```
    """
    return ColabGeminiModel(model_name, **kwargs)


def list_available_models() -> list[str]:
    """利用可能なモデル一覧を取得

    google.colab.ai.list_models() のラッパー関数です。

    Returns:
        利用可能なモデル名のリスト

    Raises:
        RuntimeError: Colab環境でない

    Examples:
        ```python
        models = list_available_models()
        print(models)
        # ['google/gemini-2.5-flash', 'google/gemini-2.0-flash', ...]
        ```
    """
    if not _is_colab_environment():
        raise RuntimeError("この関数は Google Colab 環境でのみ動作します。")

    try:
        from google.colab import ai

        return cast(list[str], ai.list_models())
    except ImportError as e:
        raise RuntimeError(
            "google.colab モジュールがインポートできません。\n"
            "Colab環境で実行していることを確認してください。"
        ) from e
