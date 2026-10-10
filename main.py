import asyncio
from chain import process_text

async def main():
    texto = "ERROR: FastAPI devuelve 504 Gateway Timeout; el pool de conexiones a PostgreSQL está agotado y Redis no responde."
    resultado = await process_text(texto)
    print(resultado)

asyncio.run(main())