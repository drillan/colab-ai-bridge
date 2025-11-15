"""LangChain integration for Google Colab AI."""

from typing import Any, Iterator, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from openai import OpenAI

from colab_ai_bridge.core.config import ModelConfig
from colab_ai_bridge.core.model import ColabModel


class ColabLangChainModel(BaseChatModel):
    """LangChain model for Google Colab AI.

    This class integrates Google Colab's AI models (Gemini, Gemma) with LangChain,
    enabling use of LangChain's extensive ecosystem.

    Attributes:
        colab_model: Underlying ColabModel instance

    Examples:
        Basic usage:
        ```python
        from colab_ai_bridge.langchain import ColabLangChainModel

        model = ColabLangChainModel()
        response = model.invoke("What is the capital of France?")
        print(response.content)
        ```

        With custom configuration:
        ```python
        from colab_ai_bridge.core.config import ModelConfig
        from colab_ai_bridge.langchain import ColabLangChainModel

        config = ModelConfig(
            model_name="google/gemini-2.5-flash",
            temperature=0.7,
            max_tokens=2000
        )
        model = ColabLangChainModel(config=config)
        ```

    Supported Models:
        - google/gemini-2.5-flash (default)
        - google/gemini-2.5-flash-lite
        - google/gemma-2-9b

    Requirements:
        - Google Colab environment
        - langchain-core package
        - openai package
    """

    colab_model: ColabModel

    def __init__(
        self,
        model_name: str = "google/gemini-2.5-flash",
        *,
        config: Optional[ModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Colab LangChain model.

        Args:
            model_name: Model name (e.g., "google/gemini-2.5-flash")
            config: Model configuration (optional)
            **kwargs: Additional arguments for BaseChatModel
        """
        colab_model = ColabModel(model_name=model_name, config=config)
        super().__init__(colab_model=colab_model, **kwargs)  # type: ignore[call-arg]

    @property
    def _llm_type(self) -> str:
        """Get LLM type identifier.

        Returns:
            "colab-ai-bridge"
        """
        return "colab-ai-bridge"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response from messages.

        Args:
            messages: List of chat messages
            stop: Stop sequences (optional)
            **kwargs: Additional arguments

        Returns:
            ChatResult containing generated response
        """
        # Create OpenAI client
        client = OpenAI(
            base_url=self.colab_model.base_url,
            api_key=self.colab_model.api_key,
        )

        # Convert LangChain messages to OpenAI format
        openai_messages = [
            {"role": msg.type, "content": msg.content} for msg in messages
        ]

        # API call
        response = client.chat.completions.create(
            model=self.colab_model.model_name,
            messages=openai_messages,  # type: ignore[arg-type]
            temperature=self.colab_model.config.temperature,
            max_tokens=self.colab_model.config.max_tokens,
            stop=stop,
            **kwargs,
        )

        # Convert response to LangChain format
        message = AIMessage(content=response.choices[0].message.content or "")
        generation = ChatGeneration(message=message)

        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream response from messages.

        Args:
            messages: List of chat messages
            stop: Stop sequences (optional)
            **kwargs: Additional arguments

        Yields:
            ChatGenerationChunk for each chunk
        """
        # Create OpenAI client
        client = OpenAI(
            base_url=self.colab_model.base_url,
            api_key=self.colab_model.api_key,
        )

        # Convert LangChain messages to OpenAI format
        openai_messages = [
            {"role": msg.type, "content": msg.content} for msg in messages
        ]

        # Stream API call
        stream = client.chat.completions.create(
            model=self.colab_model.model_name,
            messages=openai_messages,  # type: ignore[arg-type]
            temperature=self.colab_model.config.temperature,
            max_tokens=self.colab_model.config.max_tokens,
            stop=stop,
            stream=True,
            **kwargs,
        )

        # Yield chunks
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:  # type: ignore[union-attr]
                content = chunk.choices[0].delta.content  # type: ignore[union-attr]
                message = AIMessageChunk(content=content)
                yield ChatGenerationChunk(message=message)


def to_langchain(
    model: ColabModel,
) -> ColabLangChainModel:
    """Convert ColabModel to LangChain model.

    Args:
        model: ColabModel instance

    Returns:
        ColabLangChainModel instance

    Example:
        ```python
        from colab_ai_bridge.core.model import ColabModel
        from colab_ai_bridge.langchain import to_langchain

        base_model = ColabModel("google/gemini-2.5-flash")
        langchain_model = to_langchain(base_model)
        ```
    """
    return ColabLangChainModel(
        model_name=model.model_name,
        config=model.config,
    )
