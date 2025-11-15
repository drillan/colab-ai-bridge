"""Tests for DSPy integration."""

import os
from unittest.mock import patch


from colab_ai_bridge.core.config import ModelConfig
from colab_ai_bridge.core.model import ColabModel
from colab_ai_bridge.dspy.adapter import ColabDSPyLM, to_dspy


class TestColabDSPyLM:
    """Tests for ColabDSPyLM."""

    def test_init_default(self) -> None:
        """Test ColabDSPyLM initialization with defaults."""
        with patch.dict(
            os.environ,
            {
                "MODEL_PROXY_HOST": "https://model-proxy.example.com",
                "MODEL_PROXY_API_KEY": "test-api-key",
            },
        ):
            lm = ColabDSPyLM()
            assert lm.model == "google/gemini-2.5-flash"
            assert lm.model_type == "chat"
            assert lm.colab_model.config.temperature == 0.0
            assert lm.colab_model.config.max_tokens == 1000

    def test_init_custom_config(self) -> None:
        """Test ColabDSPyLM with custom config."""
        with patch.dict(
            os.environ,
            {
                "MODEL_PROXY_HOST": "https://model-proxy.example.com",
                "MODEL_PROXY_API_KEY": "test-api-key",
            },
        ):
            config = ModelConfig(
                model_name="google/gemma-2-9b", temperature=0.7, max_tokens=2000
            )
            lm = ColabDSPyLM(config=config)
            assert lm.model == "google/gemma-2-9b"
            assert lm.colab_model.config.temperature == 0.7
            assert lm.colab_model.config.max_tokens == 2000

    def test_cache_enabled(self) -> None:
        """Test cache is enabled by default."""
        with patch.dict(
            os.environ,
            {
                "MODEL_PROXY_HOST": "https://model-proxy.example.com",
                "MODEL_PROXY_API_KEY": "test-api-key",
            },
        ):
            lm = ColabDSPyLM()
            assert lm.cache is True

    def test_cache_disabled(self) -> None:
        """Test cache can be disabled."""
        with patch.dict(
            os.environ,
            {
                "MODEL_PROXY_HOST": "https://model-proxy.example.com",
                "MODEL_PROXY_API_KEY": "test-api-key",
            },
        ):
            lm = ColabDSPyLM(cache=False)
            assert lm.cache is False


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_to_dspy(self) -> None:
        """Test to_dspy conversion function."""
        with patch.dict(
            os.environ,
            {
                "MODEL_PROXY_HOST": "https://model-proxy.example.com",
                "MODEL_PROXY_API_KEY": "test-api-key",
            },
        ):
            base_model = ColabModel("google/gemini-2.5-flash")
            dspy_lm = to_dspy(base_model)
            assert isinstance(dspy_lm, ColabDSPyLM)
            assert dspy_lm.model == "google/gemini-2.5-flash"
            assert dspy_lm.cache is True

    def test_to_dspy_no_cache(self) -> None:
        """Test to_dspy with cache disabled."""
        with patch.dict(
            os.environ,
            {
                "MODEL_PROXY_HOST": "https://model-proxy.example.com",
                "MODEL_PROXY_API_KEY": "test-api-key",
            },
        ):
            base_model = ColabModel("google/gemini-2.5-flash")
            dspy_lm = to_dspy(base_model, cache=False)
            assert isinstance(dspy_lm, ColabDSPyLM)
            assert dspy_lm.cache is False
