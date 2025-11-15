"""Pydantic AI integration for Colab AI Bridge."""

from colab_ai_bridge.pydantic_ai.adapter import (
    ColabGeminiModel,
    ColabModelProfile,
    ColabPydanticAIModel,
    get_colab_gemini_model,
    list_available_models,
    to_pydantic_ai,
)

__all__ = [
    "ColabPydanticAIModel",
    "ColabGeminiModel",  # Backward compatibility
    "ColabModelProfile",
    "to_pydantic_ai",
    "get_colab_gemini_model",
    "list_available_models",
]
