"""Google Colab environment settings."""


from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ColabSettings(BaseSettings):
    """Google Colab environment configuration.

    Automatically loaded from environment variables.

    Attributes:
        model_proxy_host: Model Proxy host URL
        model_proxy_api_key: Model Proxy API key

    Example:
        ```python
        from colab_ai_bridge.core.settings import get_colab_settings

        settings = get_colab_settings()
        print(settings.model_proxy_host)
        ```
    """

    model_proxy_host: str = Field(
        ...,
        alias="MODEL_PROXY_HOST",
        description="Model Proxy host URL",
    )
    model_proxy_api_key: str = Field(
        ...,
        alias="MODEL_PROXY_API_KEY",
        description="Model Proxy API key",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


def _is_colab_environment() -> bool:
    """Check if running in Google Colab environment.

    Returns:
        True if in Colab, False otherwise
    """
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def _setup_colab_environment() -> None:
    """Setup Google Colab environment.

    Calls google.colab.ai._get_model_proxy_token() to automatically
    set MODEL_PROXY_HOST and MODEL_PROXY_API_KEY environment variables.
    """
    if not _is_colab_environment():
        return

    try:
        from google.colab import ai

        # This sets environment variables automatically
        ai._get_model_proxy_token()  # noqa: SLF001
    except Exception:  # noqa: S110
        # Ignore errors - environment variables might already be set
        pass


def get_colab_settings() -> ColabSettings:
    """Get Colab settings.

    Automatically imports google.colab.ai and sets environment variables
    if running in Colab environment.

    Returns:
        ColabSettings instance

    Raises:
        ValidationError: If required environment variables are not set

    Example:
        ```python
        from colab_ai_bridge.core.settings import get_colab_settings

        settings = get_colab_settings()
        ```
    """
    # Setup environment if in Colab
    _setup_colab_environment()

    return ColabSettings()  # type: ignore[call-arg]
