"""Tests for LangChain integration."""

import os
from unittest.mock import patch


from colab_ai_bridge.core.config import ModelConfig
from colab_ai_bridge.core.model import ColabModel
from colab_ai_bridge.langchain.adapter import ColabLangChainModel, to_langchain


class TestColabLangChainModel:
    """Tests for ColabLangChainModel."""

    def test_init_default(self) -> None:
        """Test ColabLangChainModel initialization with defaults."""
        with patch.dict(
            os.environ,
            {
                "MODEL_PROXY_HOST": "https://model-proxy.example.com",
                "MODEL_PROXY_API_KEY": "test-api-key",
            },
        ):
            model = ColabLangChainModel()
            assert model._llm_type == "colab-ai-bridge"
            assert model.colab_model.model_name == "google/gemini-2.5-flash"

    def test_init_custom_config(self) -> None:
        """Test ColabLangChainModel with custom config."""
        with patch.dict(
            os.environ,
            {
                "MODEL_PROXY_HOST": "https://model-proxy.example.com",
                "MODEL_PROXY_API_KEY": "test-api-key",
            },
        ):
            config = ModelConfig(model_name="google/gemma-2-9b", temperature=0.7)
            model = ColabLangChainModel(config=config)
            assert model.colab_model.model_name == "google/gemma-2-9b"
            assert model.colab_model.config.temperature == 0.7

    def test_llm_type(self) -> None:
        """Test _llm_type property."""
        with patch.dict(
            os.environ,
            {
                "MODEL_PROXY_HOST": "https://model-proxy.example.com",
                "MODEL_PROXY_API_KEY": "test-api-key",
            },
        ):
            model = ColabLangChainModel()
            assert model._llm_type == "colab-ai-bridge"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_to_langchain(self) -> None:
        """Test to_langchain conversion function."""
        with patch.dict(
            os.environ,
            {
                "MODEL_PROXY_HOST": "https://model-proxy.example.com",
                "MODEL_PROXY_API_KEY": "test-api-key",
            },
        ):
            base_model = ColabModel("google/gemini-2.5-flash")
            langchain_model = to_langchain(base_model)
            assert isinstance(langchain_model, ColabLangChainModel)
            assert langchain_model._llm_type == "colab-ai-bridge"
            assert langchain_model.colab_model.model_name == "google/gemini-2.5-flash"
