"""Common type definitions for Colab AI Bridge."""

from typing import Any, Literal, Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    """Chat message.

    Attributes:
        role: Message role (system, user, or assistant)
        content: Message content
    """

    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Chat request.

    Attributes:
        model: Model name
        messages: List of chat messages
        temperature: Temperature parameter (optional)
        max_tokens: Maximum tokens to generate (optional)
    """

    model: str
    messages: list[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class Usage(BaseModel):
    """Token usage statistics.

    Attributes:
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        total_tokens: Total number of tokens
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    """Chat response.

    Attributes:
        id: Response ID
        choices: List of response choices
        usage: Token usage statistics (optional)
    """

    id: str
    choices: list[dict[str, Any]]
    usage: Optional[Usage] = None
