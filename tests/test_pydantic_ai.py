"""Tests for Pydantic AI integration."""

import os
from unittest.mock import patch


from colab_ai_bridge.pydantic_ai.adapter import (
    ColabModelProfile,
    ColabPydanticAIModel,
    to_pydantic_ai,
)
from colab_ai_bridge.core.model import ColabModel
from colab_ai_bridge.core.config import ModelConfig


class TestColabModelProfile:
    """Tests for ColabModelProfile."""

    def test_profile_defaults(self) -> None:
        """Test ColabModelProfile default values."""
        profile = ColabModelProfile()
        assert profile.default_structured_output_mode == "native"
        assert profile.supports_json_schema_output is True
        assert profile.supports_tools is False
        assert profile.supports_json_object_output is False


class TestColabPydanticAIModel:
    """Tests for ColabPydanticAIModel."""

    def test_init_default(self) -> None:
        """Test ColabPydanticAIModel initialization with defaults."""
        with patch.dict(
            os.environ,
            {
                "MODEL_PROXY_HOST": "https://model-proxy.example.com",
                "MODEL_PROXY_API_KEY": "test-api-key",
            },
        ):
            model = ColabPydanticAIModel()
            assert model.system == "colab-ai-bridge"

    def test_init_custom_config(self) -> None:
        """Test ColabPydanticAIModel with custom config."""
        with patch.dict(
            os.environ,
            {
                "MODEL_PROXY_HOST": "https://model-proxy.example.com",
                "MODEL_PROXY_API_KEY": "test-api-key",
            },
        ):
            config = ModelConfig(model_name="google/gemma-2-9b")
            model = ColabPydanticAIModel(config=config)
            assert model.system == "colab-ai-bridge"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_to_pydantic_ai(self) -> None:
        """Test to_pydantic_ai conversion function."""
        with patch.dict(
            os.environ,
            {
                "MODEL_PROXY_HOST": "https://model-proxy.example.com",
                "MODEL_PROXY_API_KEY": "test-api-key",
            },
        ):
            base_model = ColabModel("google/gemini-2.5-flash")
            pydantic_model = to_pydantic_ai(base_model)
            assert isinstance(pydantic_model, ColabPydanticAIModel)
            assert pydantic_model.system == "colab-ai-bridge"
