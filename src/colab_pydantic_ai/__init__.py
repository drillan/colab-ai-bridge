"""Pydantic AI integration for Google Colab's Gemini access.

This package provides a seamless integration between Pydantic AI and Google Colab's
Gemini API access, enabling type-safe AI application development in Colab notebooks.

Example:
    ```python
    from colab_pydantic_ai import ColabGeminiModel
    from pydantic_ai import Agent

    # No setup required! Everything is automatic in Colab
    model = ColabGeminiModel()
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


# 公開APIのimport
from colab_pydantic_ai.model import (  # noqa: E402
    ColabGeminiModel,
    ColabGeminiModelProfile,
    get_colab_gemini_model,
    list_available_models,
)

__version__ = "0.1.0"

__all__ = [
    "ColabGeminiModel",
    "ColabGeminiModelProfile",
    "get_colab_gemini_model",
    "list_available_models",
]
