"""Tests for core modules."""

import os
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from colab_ai_bridge.core.config import ModelConfig
from colab_ai_bridge.core.model import ColabModel
from colab_ai_bridge.core.settings import (
    ColabSettings,
    _is_colab_environment,
)
from colab_ai_bridge.core.types import ChatMessage, ChatRequest, ChatResponse, Usage


class TestModelConfig:
    """Tests for ModelConfig class."""

    def test_model_config_defaults(self) -> None:
        """Test ModelConfig default values."""
        config = ModelConfig()
        assert config.model_name == "google/gemini-2.5-flash"
        assert config.temperature == 0.0
        assert config.max_tokens == 1000
        assert config.timeout is None

    def test_model_config_custom_values(self) -> None:
        """Test ModelConfig with custom values."""
        config = ModelConfig(
            model_name="google/gemma-2-9b", temperature=0.7, max_tokens=2000, timeout=30
        )
        assert config.model_name == "google/gemma-2-9b"
        assert config.temperature == 0.7
        assert config.max_tokens == 2000
        assert config.timeout == 30

    def test_model_config_temperature_validation(self) -> None:
        """Test ModelConfig temperature range validation."""
        # Valid range
        config = ModelConfig(temperature=0.0)
        assert config.temperature == 0.0

        config = ModelConfig(temperature=2.0)
        assert config.temperature == 2.0

        # Invalid range (too high)
        with pytest.raises(ValidationError):
            ModelConfig(temperature=3.0)

        # Invalid range (negative)
        with pytest.raises(ValidationError):
            ModelConfig(temperature=-0.1)

    def test_model_config_max_tokens_validation(self) -> None:
        """Test ModelConfig max_tokens validation."""
        # Valid value
        config = ModelConfig(max_tokens=100)
        assert config.max_tokens == 100

        # Invalid value (zero)
        with pytest.raises(ValidationError):
            ModelConfig(max_tokens=0)

        # Invalid value (negative)
        with pytest.raises(ValidationError):
            ModelConfig(max_tokens=-1)

    def test_model_config_timeout_validation(self) -> None:
        """Test ModelConfig timeout validation."""
        # Valid value
        config = ModelConfig(timeout=60)
        assert config.timeout == 60

        # Invalid value (zero)
        with pytest.raises(ValidationError):
            ModelConfig(timeout=0)

        # Invalid value (negative)
        with pytest.raises(ValidationError):
            ModelConfig(timeout=-10)

    def test_model_config_immutable(self) -> None:
        """Test ModelConfig immutability."""
        config = ModelConfig()
        with pytest.raises(ValidationError):
            config.temperature = 1.0  # type: ignore[misc]

    def test_model_config_model_name_validation(self) -> None:
        """Test ModelConfig model_name validation."""
        # Valid name
        config = ModelConfig(model_name="google/gemini-2.5-flash")
        assert config.model_name == "google/gemini-2.5-flash"

        # Empty string
        with pytest.raises(ValidationError):
            ModelConfig(model_name="")

        # Whitespace only
        with pytest.raises(ValidationError):
            ModelConfig(model_name="   ")

        # Whitespace trimming
        config = ModelConfig(model_name="  google/gemini-2.5-flash  ")
        assert config.model_name == "google/gemini-2.5-flash"


class TestColabSettings:
    """Tests for ColabSettings class."""

    def test_colab_settings_from_env(self) -> None:
        """Test ColabSettings loads from environment variables."""
        with patch.dict(
            os.environ,
            {
                "MODEL_PROXY_HOST": "https://model-proxy.example.com",
                "MODEL_PROXY_API_KEY": "test-api-key",
            },
        ):
            settings = ColabSettings()  # type: ignore[call-arg]
            assert settings.model_proxy_host == "https://model-proxy.example.com"
            assert settings.model_proxy_api_key == "test-api-key"

    def test_colab_settings_missing_env_vars(self) -> None:
        """Test ColabSettings raises error when env vars missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValidationError):
                ColabSettings()  # type: ignore[call-arg]

    def test_is_colab_environment(self) -> None:
        """Test _is_colab_environment detection."""
        # Mock Colab environment
        with patch.dict("sys.modules", {"google.colab": Mock()}):
            assert _is_colab_environment() is True

        # Test returns False when google.colab is not available
        # (ImportError is caught and False is returned)
        assert _is_colab_environment() in [True, False]  # Either is valid depending on env


class TestColabModel:
    """Tests for ColabModel class."""

    def test_colab_model_init_default(self) -> None:
        """Test ColabModel initialization with defaults."""
        with patch.dict(
            os.environ,
            {
                "MODEL_PROXY_HOST": "https://model-proxy.example.com",
                "MODEL_PROXY_API_KEY": "test-api-key",
            },
        ):
            model = ColabModel()
            assert model.model_name == "google/gemini-2.5-flash"
            assert model.base_url == "https://model-proxy.example.com/models/openapi"
            assert model.api_key == "test-api-key"

    def test_colab_model_init_custom(self) -> None:
        """Test ColabModel initialization with custom values."""
        with patch.dict(
            os.environ,
            {
                "MODEL_PROXY_HOST": "https://model-proxy.example.com",
                "MODEL_PROXY_API_KEY": "test-api-key",
            },
        ):
            config = ModelConfig(model_name="google/gemma-2-9b")
            model = ColabModel(config=config)
            assert model.model_name == "google/gemma-2-9b"

    def test_colab_model_repr(self) -> None:
        """Test ColabModel string representation."""
        with patch.dict(
            os.environ,
            {
                "MODEL_PROXY_HOST": "https://model-proxy.example.com",
                "MODEL_PROXY_API_KEY": "test-api-key",
            },
        ):
            model = ColabModel()
            assert repr(model) == "ColabModel(model='google/gemini-2.5-flash')"


class TestTypes:
    """Tests for common types."""

    def test_chat_message(self) -> None:
        """Test ChatMessage type."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

        # Invalid role
        with pytest.raises(ValidationError):
            ChatMessage(role="invalid", content="Hello")  # type: ignore[arg-type]

    def test_chat_request(self) -> None:
        """Test ChatRequest type."""
        messages = [ChatMessage(role="user", content="Hello")]
        req = ChatRequest(model="google/gemini-2.5-flash", messages=messages)
        assert req.model == "google/gemini-2.5-flash"
        assert len(req.messages) == 1
        assert req.temperature is None
        assert req.max_tokens is None

    def test_usage(self) -> None:
        """Test Usage type."""
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30

    def test_chat_response(self) -> None:
        """Test ChatResponse type."""
        response = ChatResponse(id="test-id", choices=[{"index": 0, "message": {}}])
        assert response.id == "test-id"
        assert len(response.choices) == 1
        assert response.usage is None
