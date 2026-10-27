from typing import Literal

from pydantic import BaseModel, Field, SecretStr, model_validator


class LLMConfig(BaseModel):
    provider: Literal["openai", "anthropic"] = "openai"
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    openai_model: str = "gpt-6-luna"
    anthropic_model: str = "claude-haiku-5-5"

    @model_validator(mode="after")
    def _key(self):
        key = self.openai_api_key if self.provider == "openai" else self.anthropic_api_key
        if key is None or not key.get_secret_value().strip():
            missing = "OPENAI_API_KEY" if self.provider == "openai" else "ANTHROPIC_API_KEY"
            raise ValueError(f"{missing} requerida")
        return self


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ModelResponse(BaseModel):
    content: str = ""
    provider: Literal["openai", "anthropic"]
    error: str | None = None
