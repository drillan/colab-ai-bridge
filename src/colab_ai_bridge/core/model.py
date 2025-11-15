"""Base model class for Colab AI integration."""

from typing import Optional

from colab_ai_bridge.core.config import ModelConfig
from colab_ai_bridge.core.settings import ColabSettings, get_colab_settings


class ColabModel:
    """Base class for Colab AI models.

    Provides access to Gemini/Gemma via Model Proxy.

    Attributes:
        config: Model configuration
        settings: Colab environment settings
        base_url: Model Proxy base URL
        api_key: Model Proxy API key
        model_name: Model identifier

    Example:
        ```python
        from colab_ai_bridge.core.model import ColabModel

        model = ColabModel("google/gemini-2.5-flash")
        print(model.base_url)
        print(model.model_name)
        ```
    """

    def __init__(
        self,
        model_name: str = "google/gemini-2.5-flash",
        config: Optional[ModelConfig] = None,
        settings: Optional[ColabSettings] = None,
    ) -> None:
        """Initialize Colab model.

        Args:
            model_name: Model name (default: google/gemini-2.5-flash)
            config: Model configuration (auto-generated if None)
            settings: Colab settings (auto-retrieved if None)
        """
        self.config = config or ModelConfig(model_name=model_name)
        self.settings = settings or get_colab_settings()

        # Build Model Proxy URL
        self.base_url = f"{self.settings.model_proxy_host}/models/openapi"
        self.api_key = self.settings.model_proxy_api_key

    @property
    def model_name(self) -> str:
        """Get model name.

        Returns:
            Model name from configuration
        """
        return self.config.model_name

    def __repr__(self) -> str:
        """String representation.

        Returns:
            String representation of the model
        """
        return f"ColabModel(model='{self.model_name}')"
