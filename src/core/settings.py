from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
)
from typing import Tuple, Optional

class Settings(BaseSettings):
    """Class to store all the settings of the application."""

    OPENAI_API_KEY: str
    OPENAI_ORG_ID: Optional[str] = None
    GEMINI_API_KEY: str
    SERP_API_KEY: Optional[str] = None
    SERP_API_URL: Optional[str] = None
    ANTHROPIC_API_KEY: str

    class Config:  # Use Config class for settings
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"

    @classmethod
    def customise_sources(
        cls,
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:  # Use Tuple instead of tuple
        """Customise the settings sources order.

        Order: dotenv, file secrets, environment variables, then initialization.
        """
        return (
            dotenv_settings,
            file_secret_settings,
            env_settings,
            init_settings,
        )


settings = Settings()
