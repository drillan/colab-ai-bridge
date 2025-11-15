"""Model configuration classes."""

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """Configuration for Colab AI models.

    Uses Pydantic for type safety and validation.

    Attributes:
        model_name: Model identifier (e.g., "google/gemini-2.5-flash")
        temperature: Temperature parameter for generation (0.0-2.0)
        max_tokens: Maximum number of tokens to generate
        timeout: Request timeout in seconds (optional)

    Example:
        ```python
        from colab_ai_bridge.core.config import ModelConfig

        config = ModelConfig(
            model_name="google/gemini-2.5-flash",
            temperature=0.7,
            max_tokens=2000
        )
        ```
    """

    model_name: str = Field(
        default="google/gemini-2.5-flash",
        description="Model name (e.g., google/gemini-2.5-flash, google/gemma-2-9b)",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature parameter for generation",
    )
    max_tokens: int = Field(
        default=1000,
        gt=0,
        description="Maximum number of tokens to generate",
    )
    timeout: Optional[int] = Field(
        default=None,
        gt=0,
        description="Request timeout in seconds",
    )

    model_config = {
        "frozen": True,  # Immutable
        "validate_assignment": True,
    }

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate that model_name is not empty."""
        if not v or not v.strip():
            raise ValueError("model_name must not be empty")
        return v.strip()
