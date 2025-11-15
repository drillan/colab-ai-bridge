"""DSPy integration for Google Colab AI."""

from typing import Any, Optional

import dspy  # type: ignore[import-untyped]
from openai import OpenAI

from colab_ai_bridge.core.config import ModelConfig
from colab_ai_bridge.core.model import ColabModel


class ColabDSPyLM(dspy.LM):
    """DSPy LM for Google Colab AI.

    This class integrates Google Colab's AI models (Gemini, Gemma) with DSPy,
    enabling use of DSPy's optimization and prompting capabilities.

    Examples:
        Basic usage:
        ```python
        from colab_ai_bridge.dspy import ColabDSPyLM
        import dspy

        lm = ColabDSPyLM()
        dspy.configure(lm=lm)

        # Use DSPy
        predictor = dspy.Predict("question -> answer")
        response = predictor(question="What is the capital of France?")
        print(response.answer)
        ```

        With custom configuration:
        ```python
        from colab_ai_bridge.core.config import ModelConfig
        from colab_ai_bridge.dspy import ColabDSPyLM

        config = ModelConfig(
            model_name="google/gemini-2.5-flash",
            temperature=0.7,
            max_tokens=2000
        )
        lm = ColabDSPyLM(config=config)
        ```

    Supported Models:
        - google/gemini-2.5-flash (default)
        - google/gemini-2.5-flash-lite
        - google/gemma-2-9b

    Requirements:
        - Google Colab environment
        - dspy-ai package
        - openai package
    """

    def __init__(
        self,
        model_name: str = "google/gemini-2.5-flash",
        *,
        config: Optional[ModelConfig] = None,
        cache: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize Colab DSPy LM.

        Args:
            model_name: Model name (e.g., "google/gemini-2.5-flash")
            config: Model configuration (optional)
            cache: Enable caching (default: True)
            **kwargs: Additional arguments for dspy.LM
        """
        colab_model = ColabModel(model_name=model_name, config=config)

        super().__init__(
            model=colab_model.model_name,
            model_type="chat",
            temperature=colab_model.config.temperature,
            max_tokens=colab_model.config.max_tokens,
            cache=cache,
            **kwargs,
        )

        self.colab_model = colab_model

        # Create OpenAI client
        self.client = OpenAI(
            base_url=colab_model.base_url,
            api_key=colab_model.api_key,
        )

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """LM call (DSPy requirement).

        Args:
            prompt: Prompt string (converted to messages if provided)
            messages: List of message dicts (optional)
            **kwargs: Additional arguments

        Returns:
            List of response dicts in DSPy expected format
        """
        # Convert prompt to messages if needed
        if messages is None and prompt is not None:
            messages = [{"role": "user", "content": prompt}]

        if messages is None:
            raise ValueError("Either prompt or messages must be provided")

        # OpenAI API call
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )

        # Return DSPy expected format
        return [
            {
                "role": "assistant",
                "content": response.choices[0].message.content or "",
            }
        ]


def to_dspy(
    model: ColabModel,
    cache: bool = True,
    **kwargs: Any,
) -> ColabDSPyLM:
    """Convert ColabModel to DSPy LM.

    Args:
        model: ColabModel instance
        cache: Enable caching (default: True)
        **kwargs: Additional arguments for ColabDSPyLM

    Returns:
        ColabDSPyLM instance

    Example:
        ```python
        from colab_ai_bridge.core.model import ColabModel
        from colab_ai_bridge.dspy import to_dspy

        base_model = ColabModel("google/gemini-2.5-flash")
        dspy_lm = to_dspy(base_model)
        ```
    """
    return ColabDSPyLM(
        model_name=model.model_name,
        config=model.config,
        cache=cache,
        **kwargs,
    )
