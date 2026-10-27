import asyncio
from collections.abc import AsyncIterator

import anthropic
import openai
from pydantic import ValidationError

from schemas import ChatMessage, LLMConfig, ModelResponse

RETRYABLE = (
    openai.APIConnectionError,
    openai.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
)


class AsyncLLMManager:
    def __init__(self, config: LLMConfig):
        self.config = config
        self._delay = 0.5
        if config.provider == "openai":
            assert config.openai_api_key is not None
            self._client = openai.AsyncOpenAI(
                api_key=config.openai_api_key.get_secret_value(), max_retries=0
            )
        else:
            assert config.anthropic_api_key is not None
            self._client = anthropic.AsyncAnthropic(
                api_key=config.anthropic_api_key.get_secret_value(), max_retries=0
            )

    def _messages(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        if not messages:
            raise ValueError("se requiere al menos un mensaje")
        return [ChatMessage.model_validate(m) for m in messages]

    def _openai(self, messages: list[ChatMessage]) -> dict:
        return {
            "model": self.config.openai_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

    def _anthropic(self, messages: list[ChatMessage]) -> dict:
        system = "\n".join(m.content for m in messages if m.role == "system")
        chat = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        if not chat:
            chat = [{"role": "user", "content": system}]
            system = ""
        # ponytail: este SDK no acepta temperature; el rango vive en LLMConfig y solo se envía a OpenAI
        kwargs = {
            "model": self.config.anthropic_model,
            "messages": chat,
            "max_tokens": self.config.max_tokens,
        }
        if system:
            kwargs["system"] = system
        return kwargs

    async def _complete(self, messages: list[ChatMessage]) -> str:
        if self.config.provider == "openai":
            resp = await self._client.chat.completions.create(**self._openai(messages))
            return resp.choices[0].message.content or ""
        resp = await self._client.messages.create(**self._anthropic(messages))
        return "".join(block.text for block in resp.content if getattr(block, "text", None))

    async def _tokens(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        if self.config.provider == "openai":
            stream = await self._client.chat.completions.create(**self._openai(messages), stream=True)
            async for chunk in stream:
                if chunk.choices and (delta := chunk.choices[0].delta.content):
                    yield delta
            return
        async with self._client.messages.stream(**self._anthropic(messages)) as stream:
            async for text in stream.text_stream:
                yield text

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        try:
            messages = self._messages(messages)
        except (ValidationError, ValueError) as e:
            return ModelResponse(provider=self.config.provider, error=str(e))
        last = "error desconocido"
        for attempt in range(3):
            try:
                text = await self._complete(messages)
                return ModelResponse(content=text, provider=self.config.provider)
            except RETRYABLE as e:
                last = f"{type(e).__name__}: {e}"
                if attempt < 2:
                    await asyncio.sleep(self._delay * 2**attempt)
            except (openai.APIError, anthropic.APIError) as e:
                return ModelResponse(provider=self.config.provider, error=f"{type(e).__name__}: {e}")
        return ModelResponse(provider=self.config.provider, error=last)

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        try:
            messages = self._messages(messages)
        except (ValidationError, ValueError) as e:
            yield f"\n[error] {e}\n"
            return
        last = "error desconocido"
        for attempt in range(3):
            try:
                async for token in self._tokens(messages):
                    yield token
                return
            except RETRYABLE as e:
                last = f"{type(e).__name__}: {e}"
                if attempt < 2:
                    await asyncio.sleep(self._delay * 2**attempt)
            except (openai.APIError, anthropic.APIError) as e:
                yield f"\n[error] {type(e).__name__}: {e}\n"
                return
        yield f"\n[error] {last}\n"


def _self_check() -> None:
    try:
        LLMConfig(temperature=3, openai_api_key="k")
        raise SystemExit("temperature fuera de rango pasó")
    except ValidationError:
        pass

    calls = 0

    async def flaky(messages: list[ChatMessage]) -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            import httpx2

            req = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
            raise openai.RateLimitError("slow", response=httpx2.Response(429, request=req), body=None)
        return "ok"

    manager = AsyncLLMManager(LLMConfig(provider="openai", openai_api_key="k", temperature=0, max_tokens=8))
    manager._delay = 0
    manager._complete = flaky

    async def run() -> None:
        out = await manager.generate([ChatMessage(role="user", content="hola")])
        assert out.content == "ok" and out.error is None and calls == 3, out

    asyncio.run(run())


if __name__ == "__main__":
    _self_check()
    print("ok")
