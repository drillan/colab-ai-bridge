"""Core modules for Colab AI Bridge.

This module provides the foundational components for integrating
Google Colab AI models with various frameworks.
"""

from colab_ai_bridge.core.config import ModelConfig
from colab_ai_bridge.core.model import ColabModel
from colab_ai_bridge.core.settings import ColabSettings, get_colab_settings

__all__ = [
    "ModelConfig",
    "ColabModel",
    "ColabSettings",
    "get_colab_settings",
]
