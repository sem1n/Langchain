import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

load_dotenv(override=True)


def _get_env(name: str, required: bool = False):
    value = os.getenv(name)

    if isinstance(value, str):
        value = value.strip()

    if required and not value:
        raise ValueError(f".env 파일에 {name} 값이 없음.")

    return value


def create_llm():
    model_name = _get_env("MODEL_NAME", required=True)
    token = _get_env("TOKEN", required=True)
    base_url = _get_env("BASE_URL", required=True)

    return ChatOpenAI(
        model=model_name,
        api_key=token,
        base_url=base_url,
        temperature=0,
    )


def create_embeddings():
    return FastEmbedEmbeddings()