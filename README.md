# Unified Async LLM Client

`AsyncLLMManager` llama a OpenAI o Anthropic según `LLM_PROVIDER`. `generate` devuelve la respuesta completa. `stream` hace `yield` de cada token.

## Setup

Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

| Variable | Uso |
|---|---|
| `LLM_PROVIDER` | `openai` o `anthropic` |
| `OPENAI_API_KEY` | Obligatoria si el proveedor es OpenAI |
| `ANTHROPIC_API_KEY` | Obligatoria si el proveedor es Anthropic |
| `OPENAI_MODEL` | Opcional. Default `gpt-6-luna` |
| `ANTHROPIC_MODEL` | Opcional. Default `claude-haiku-5-5` |

`LLMConfig` rechaza temperatura fuera de 0–2 y `max_tokens` fuera de 1–8192.

## Ejecutar

```bash
python main.py
```

Pregunta «¿Qué es la entropía?» en modo normal y en streaming. Red o rate limit: 3 reintentos y, si sigue, imprime el error sin cortar el script.
