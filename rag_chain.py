from typing import List

import bs4
from langchain_classic import hub
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS


def load_documents(url: str) -> List[Document]:
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs={
            "parse_only": bs4.SoupStrainer(
                "div",
                attrs={
                    "class": [
                        "newsct_article _article_body",
                        "media_end_head_title",
                    ]
                },
            )
        },
    )

    return loader.load()


def split_documents(docs: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
    )

    return text_splitter.split_documents(docs)


def create_vectorstore(splits: List[Document], embeddings):
    return FAISS.from_documents(
        documents=splits,
        embedding=embeddings,
    )


def create_retriever(vectorstore):
    return vectorstore.as_retriever()


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever, llm):
    prompt = hub.pull("rlm/rag-prompt")

    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )