import asyncio
import os

from dotenv import load_dotenv
from pydantic import ValidationError

from chain import AsyncLLMManager
from schemas import ChatMessage, LLMConfig

load_dotenv()

PREGUNTA = [ChatMessage(role="user", content="¿Qué es la entropía?")]


async def main():
    try:
        config = LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "openai"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-6-luna"),
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"),
        )
    except ValidationError as e:
        print(f"Configuración inválida: {e}")
        return

    manager = AsyncLLMManager(config)
    print("— modo normal —")
    respuesta = await manager.generate(PREGUNTA)
    print(respuesta.error if respuesta.error else respuesta.content)
    print("\n— streaming —")
    async for token in manager.stream(PREGUNTA):
        print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
