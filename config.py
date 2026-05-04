import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

load_dotenv()

def create_llm():
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME"),
        api_key=os.getenv("TOKEN"),
        base_url=os.getenv("BASE_URL"),
        temperature=0,
    )

def create_embeddings():
    mode = os.getenv("EMBEDDING_MODE", "fastembed")
    
    if mode == "fastembed":
        return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    else:
        return HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

def get_retriever_config():
    return {
        "mode": os.getenv("RETRIEVER_MODE", "similarity"),
        "k": 3
    }