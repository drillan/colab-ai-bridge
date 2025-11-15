"""Pydantic AI integration for Google Colab AI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, cast

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from colab_ai_bridge.core.config import ModelConfig
from colab_ai_bridge.core.model import ColabModel
from colab_ai_bridge.core.settings import _is_colab_environment


@dataclass(kw_only=True)
class ColabModelProfile(OpenAIModelProfile):
    """Model profile for Google Colab Model Proxy Service.

    Model Proxy Service feature support:
    - ✅ Structured Outputs (response_format + JSON Schema)
    - ✅ Temperature parameter
    - ✅ Streaming
    - ❌ Tool Calling (not supported)
    - ❌ JSON Mode (different response format)

    This profile is configured with default_structured_output_mode='native'
    to use Structured Outputs and avoid Tool Calling.
    """

    # Use Structured Outputs (avoid Tool Calling)
    default_structured_output_mode: str = "native"  # type: ignore[assignment]

    # Model Proxy Service supports JSON Schema
    supports_json_schema_output: bool = True

    # Tool Calling not supported
    supports_tools: bool = False

    # JSON Mode partially supported (disabled due to different response format)
    supports_json_object_output: bool = False

    # JSON Schema transformer (OpenAI compatible)
    json_schema_transformer: type = OpenAIJsonSchemaTransformer  # type: ignore[assignment]


class ColabPydanticAIModel(OpenAIChatModel):
    """Pydantic AI model for Google Colab Gemini/Gemma access.

    Uses Google Colab's internal OpenAI-compatible proxy to provide
    full Pydantic AI functionality (Tools, Streaming, etc.).

    This class inherits from OpenAIChatModel, providing:
    - Function Calling
    - Streaming responses
    - Structured outputs
    - Vision (image input)
    - Web Search Tool (if supported by Model Proxy)

    Examples:
        Basic usage:
        ```python
        from colab_ai_bridge.pydantic_ai import ColabPydanticAIModel
        from pydantic_ai import Agent

        model = ColabPydanticAIModel()
        agent = Agent(model)
        result = agent.run_sync("What is the capital of France?")
        print(result.output)
        ```

        Custom settings:
        ```python
        model = ColabPydanticAIModel(
            "google/gemini-2.5-flash",
            settings=ModelSettings(
                temperature=0.7,
                max_tokens=2000,
            )
        )
        ```

    Supported Models:
        Models available via google.colab.ai.list_models():
        - google/gemini-2.5-flash (default)
        - google/gemini-2.5-flash-lite

    Requirements:
        - Google Colab environment
        - MODEL_PROXY_API_KEY (set in Colab Secrets)
        - pydantic-ai package

    Notes:
        - OpenAI Chat Completions API compatible
        - ✅ **Structured output support**: Use output_type with Pydantic models
          (via Structured Outputs, not Tool Calling)
        - ✅ Temperature parameter fully supported
        - ❌ Tool Calling not supported (custom tools, Web Search Tool unavailable)
        - Raises RuntimeError when used outside Colab environment
    """

    def __init__(
        self,
        model_name: str = "google/gemini-2.5-flash",
        *,
        config: Optional[ModelConfig] = None,
        settings: ModelSettings | None = None,
        profile: Any | None = None,
    ) -> None:
        """Initialize Colab Pydantic AI model.

        Args:
            model_name: Gemini model name
                (e.g., "google/gemini-2.5-flash", "google/gemini-2.5-flash-lite")
            config: Model configuration (ModelConfig instance)
            settings: Model settings (temperature, max_tokens, etc.)
            profile: Model profile (advanced settings)
                If None, ColabModelProfile() will be used,
                enabling Structured Outputs.

        Raises:
            RuntimeError: Not in Colab environment or credentials not set
        """
        # Use ColabModel to get base configuration
        colab_model = ColabModel(model_name=model_name, config=config)

        # If profile not specified, use Colab-specific profile
        if profile is None:
            profile = ColabModelProfile()

        # Create OpenAI provider
        provider = OpenAIProvider(
            base_url=colab_model.base_url,
            api_key=colab_model.api_key,
        )

        # Initialize OpenAIChatModel
        super().__init__(
            model_name=colab_model.model_name,
            provider=provider,
            settings=settings,
            profile=profile,
        )

    @property
    def system(self) -> str:
        """System name for OpenTelemetry.

        Returns:
            "colab-ai-bridge"
        """
        return "colab-ai-bridge"


def to_pydantic_ai(
    model: ColabModel,
    *,
    settings: ModelSettings | None = None,
    profile: Any | None = None,
) -> ColabPydanticAIModel:
    """Convert ColabModel to Pydantic AI model.

    Args:
        model: ColabModel instance
        settings: Model settings (optional)
        profile: Model profile (optional)

    Returns:
        ColabPydanticAIModel instance

    Example:
        ```python
        from colab_ai_bridge.core.model import ColabModel
        from colab_ai_bridge.pydantic_ai.adapter import to_pydantic_ai

        base_model = ColabModel("google/gemini-2.5-flash")
        pydantic_model = to_pydantic_ai(base_model)
        ```
    """
    return ColabPydanticAIModel(
        model_name=model.model_name,
        config=model.config,
        settings=settings,
        profile=profile,
    )


def list_available_models() -> list[str]:
    """Get list of available models.

    Wrapper for google.colab.ai.list_models().

    Returns:
        List of available model names

    Raises:
        RuntimeError: Not in Colab environment

    Example:
        ```python
        models = list_available_models()
        print(models)
        # ['google/gemini-2.5-flash', 'google/gemini-2.0-flash', ...]
        ```
    """
    if not _is_colab_environment():
        raise RuntimeError("This function only works in Google Colab environment.")

    try:
        from google.colab import ai

        return cast(list[str], ai.list_models())
    except ImportError as e:
        raise RuntimeError(
            "Cannot import google.colab module.\n"
            "Please ensure you are running in Colab environment."
        ) from e
