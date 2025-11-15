"""Bridge Google Colab AI models to popular frameworks.

This package provides seamless integration between Google Colab's AI models
(Gemini, Gemma) and popular AI frameworks (Pydantic AI, LangChain, DSPy).

Example:
    ```python
    from colab_ai_bridge.pydantic_ai import ColabPydanticAIModel
    from pydantic_ai import Agent

    # No setup required! Everything is automatic in Colab
    model = ColabPydanticAIModel()
    agent = Agent(model)

    # Run agent (nest_asyncio is automatically applied)
    result = agent.run_sync("What is the capital of France?")
    print(result.output)
    ```
"""


def _auto_setup_colab_environment() -> bool:
    """Import時に自動的にColab環境をセットアップ

    以下を実行します：
    1. google.colab.ai._get_model_proxy_token() の呼び出し（MODEL_PROXY_API_KEY自動設定）
    2. nest_asyncio.apply()（イベントループのネスト許可）

    Returns:
        True: Colab環境でセットアップ成功
        False: Colab環境外（セットアップ不要）
    """
    try:
        # google.colab.ai._get_model_proxy_token() を呼び出してMODEL_PROXY_API_KEYを設定
        # この関数内部で Colab Secrets から取得した値を環境変数に設定する
        from google.colab import ai  # noqa: F401

        try:
            ai._get_model_proxy_token()  # noqa: SLF001
        except Exception:  # noqa: S110
            # エラーは無視（環境変数設定が目的）
            pass

        # nest_asyncioを適用（イベントループのネスト許可）
        try:
            import nest_asyncio

            nest_asyncio.apply()
        except ImportError:
            # nest_asyncioが未インストールの場合は依存関係で自動インストールされるはず
            pass

        return True
    except ImportError:
        # Colab環境外では何もしない（エラーにしない）
        return False


# Import時に自動セットアップを実行
_auto_setup_colab_environment()


# バージョン情報を importlib.metadata から動的に取得
try:
    from importlib.metadata import version

    __version__ = version("colab-ai-bridge")
except Exception:
    __version__ = "unknown"

# コアモジュール
from colab_ai_bridge.core import (  # noqa: E402, F401
    ColabModel,
    ColabSettings,
    ModelConfig,
    get_colab_settings,
)

__all__ = [
    # Core modules only
    "ColabModel",
    "ModelConfig",
    "ColabSettings",
    "get_colab_settings",
]
