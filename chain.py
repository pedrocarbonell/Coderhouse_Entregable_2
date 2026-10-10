import asyncio
import os
import langchain_core
import openai
import pydantic
from pydantic import BaseModel, ValidationError, SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from schemas import EntityExtraction

load_dotenv()

async def process_text(text:str):
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-6-luna",
        reasoning_effort="medium",
    )

    structured_llm = llm.with_structured_output(EntityExtraction)

    resilient_llm = structured_llm.with_retry(
        wait_exponential_jitter=True,
        stop_after_attempt=3,
        retry_if_exception_type=(openai.APIConnectionError,
                                 openai.RateLimitError,
                                 pydantic.ValidationError,
                                 langchain_core.exceptions.OutputParserException))

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "Eres un arquitecto de software y recibes un parrafo que describe una arquitectura de software o "
             "log de error y devuelve un objeto validado"),
            ("human", "Parrafo: {parrafo}"),
        ]

    )

    try:
        chain = prompt | resilient_llm
        response = await chain.ainvoke({"parrafo": text})
        return response
    except ValidationError as e:
        print(f"La respuesta del LLM no cumple el esquema: {e.error_count()} errores")
        for err in e.errors():
            print(f" - {err['loc'][0]}: {err['msg']} (recibido: {err['input']!r})")
    except Exception as e:
        print(f"Error: {e}")





