import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)


def _get_env(name: str, default=None, required: bool = False):
    value = os.getenv(name, default)

    if isinstance(value, str):
        value = value.strip()

    if required and not value:
        raise ValueError(f".env 파일에 {name} 값이 없음.")

    return value


def _get_int(name: str, default: int) -> int:
    value = _get_env(name, str(default))
    try:
        return int(value)
    except ValueError:
        raise ValueError(f".env의 {name} 값은 정수, 현재 값: {value}")


def _get_float(name: str, default: float) -> float:
    value = _get_env(name, str(default))
    try:
        return float(value)
    except ValueError:
        raise ValueError(f".env의 {name} 값은 실수, 현재 값: {value}")


def create_llm():
    model_name = _get_env("MODEL_NAME", required=True)
    token = _get_env("TOKEN", required=True)
    base_url = _get_env("BASE_URL", "")

    kwargs = {
        "model": model_name,
        "api_key": token,
        "temperature": _get_float("TEMPERATURE", 0.0),
    }

    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)


def create_embeddings():
    mode = _get_env("EMBEDDING_MODE", "fastembed").lower()

    if mode == "fastembed":
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

        return FastEmbedEmbeddings(
            model_name=_get_env("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
        )

    elif mode == "huggingface":
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings

        return HuggingFaceBgeEmbeddings(
            model_name=_get_env("HF_EMBEDDING_MODEL", "BAAI/bge-m3"),
            model_kwargs={"device": _get_env("HF_DEVICE", "cpu")},
            encode_kwargs={"normalize_embeddings": True},
        )

    elif mode == "openai":
        from langchain_openai import OpenAIEmbeddings

        model_name = _get_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        token = _get_env("TOKEN", required=True)
        base_url = _get_env("BASE_URL", "")

        kwargs = {
            "model": model_name,
            "api_key": token,
        }

        if base_url:
            kwargs["base_url"] = base_url

        return OpenAIEmbeddings(**kwargs)

    else:
        raise ValueError(
            f"지원하지 않는 EMBEDDING_MODE: {mode} "
            f"(fastembed / huggingface / openai 중 하나 사용)"
        )


def get_retriever_config():
    return {
        "mode": _get_env("RETRIEVER_MODE", "similarity").lower(),
        "k": _get_int("TOP_K", 3),
        "score_threshold": _get_float("SCORE_THRESHOLD", 0.5),
        "fetch_k": _get_int("FETCH_K", 20),
        "lambda_mult": _get_float("LAMBDA_MULT", 0.5),
        "bm25_weight": _get_float("BM25_WEIGHT", 0.5),
        "faiss_weight": _get_float("FAISS_WEIGHT", 0.5),
    }


def get_app_config():
    return {
        "chunk_size": _get_int("CHUNK_SIZE", 1000),
        "chunk_overlap": _get_int("CHUNK_OVERLAP", 50),
        "splitter_mode": _get_env("SPLITTER_MODE", "recursive").lower(),
        "vectorstore_mode": _get_env("VECTORSTORE_MODE", "faiss").lower(),
        "persist_directory": _get_env("PERSIST_DIRECTORY", "./chroma_db"),
        "prompt_mode": _get_env("PROMPT_MODE", "local").lower(),
    }