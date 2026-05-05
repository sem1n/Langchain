import logging
import re
from pathlib import Path
from typing import List

import bs4
from langchain_classic import hub
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    CSVLoader,
    DirectoryLoader,
    PythonLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import EnsembleRetriever


def _web_loader(url: str, soup_strainer=None):
    kwargs = {
        "web_paths": (url,),
        "header_template": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        },
    }

    if soup_strainer is not None:
        kwargs["bs_kwargs"] = {"parse_only": soup_strainer}

    return WebBaseLoader(**kwargs)


def load_naver_news(url: str) -> List[Document]:
    attempts = [
        bs4.SoupStrainer(
            "div",
            attrs={
                "class": [
                    "newsct_article _article_body",
                    "media_end_head_title",
                ]
            },
        ),
        bs4.SoupStrainer(
            ["h2", "article"],
            attrs={
                "id": [
                    "title_area",
                    "dic_area",
                ]
            },
        ),
    ]

    for soup in attempts:
        docs = _web_loader(url, soup).load()
        docs = clean_documents(docs)
        if docs and len(docs[0].page_content.strip()) > 80:
            return docs

    docs = _web_loader(url).load()
    return clean_documents(docs)


def load_web_page(url: str) -> List[Document]:
    docs = _web_loader(url).load()
    return clean_documents(docs)


def load_pdf(file_path: str) -> List[Document]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없음: {file_path}")

    loader = PyPDFLoader(str(path))
    return clean_documents(loader.load())


def load_csv(file_path: str) -> List[Document]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없음: {file_path}")

    loader = CSVLoader(file_path=str(path), encoding="utf-8")
    return clean_documents(loader.load())


def load_python_files(path: str) -> List[Document]:
    target = Path(path)

    if not target.exists():
        raise FileNotFoundError(f"경로를 찾을 수 없음: {path}")

    if target.is_file():
        loader = PythonLoader(str(target))
        return clean_documents(loader.load())

    loader = DirectoryLoader(
        str(target),
        glob="**/*.py",
        loader_cls=PythonLoader,
        show_progress=True,
    )
    return clean_documents(loader.load())


def load_documents(source: str, source_type: str = "naver") -> List[Document]:
    source_type = source_type.lower()

    if source_type == "naver":
        return load_naver_news(source)

    if source_type == "web":
        return load_web_page(source)

    if source_type == "pdf":
        return load_pdf(source)

    if source_type == "csv":
        return load_csv(source)

    if source_type == "python":
        return load_python_files(source)

    raise ValueError(
        f"지원하지 않는 source_type: {source_type} "
        f"(naver / web / pdf / csv / python 중 하나 사용)"
    )


def clean_documents(docs: List[Document]) -> List[Document]:
    cleaned_docs = []

    for doc in docs:
        text = doc.page_content or ""
        text = re.sub(r"\s+", " ", text).strip()

        if text:
            cleaned_docs.append(
                Document(
                    page_content=text,
                    metadata=doc.metadata,
                )
            )

    return cleaned_docs


def split_documents(
    docs: List[Document],
    splitter_mode: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 50,
) -> List[Document]:
    splitter_mode = splitter_mode.lower()

    if splitter_mode == "character":
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    elif splitter_mode == "recursive":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    else:
        raise ValueError(
            f"지원하지 않는 SPLITTER_MODE: {splitter_mode} "
            f"(recursive / character 중 하나 사용)"
        )

    return text_splitter.split_documents(docs)


def create_vectorstore(
    splits: List[Document],
    embeddings,
    vectorstore_mode: str = "faiss",
    persist_directory: str = "./chroma_db",
):
    vectorstore_mode = vectorstore_mode.lower()

    if vectorstore_mode == "faiss":
        return FAISS.from_documents(
            documents=splits,
            embedding=embeddings,
        )

    elif vectorstore_mode == "chroma":
        from langchain_community.vectorstores import Chroma

        return Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_directory,
        )

    else:
        raise ValueError(
            f"지원하지 않는 VECTORSTORE_MODE: {vectorstore_mode} "
            f"(faiss / chroma 중 하나 사용)"
        )


def create_retriever(
    vectorstore,
    splits: List[Document],
    llm,
    mode: str = "similarity",
    k: int = 3,
    score_threshold: float = 0.5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    bm25_weight: float = 0.5,
    faiss_weight: float = 0.5,
):
    mode = mode.lower()

    if mode == "similarity":
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    elif mode == "threshold":
        return vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": score_threshold,
                "k": k,
            },
        )

    elif mode == "mmr":
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k,
                "lambda_mult": lambda_mult,
            },
        )

    elif mode == "multi_query":
        base_retriever = vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

        logging.basicConfig()
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
        logging.getLogger("langchain_classic.retrievers.multi_query").setLevel(logging.INFO)

        return MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm,
        )

    elif mode == "ensemble":
        bm25_retriever = BM25Retriever.from_documents(splits)
        bm25_retriever.k = k

        faiss_retriever = vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

        return EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[bm25_weight, faiss_weight],
        )

    else:
        raise ValueError(
            f"지원하지 않는 RETRIEVER_MODE: {mode} "
            f"(similarity / threshold / mmr / multi_query / ensemble 중 하나 사용)"
        )


def create_prompt(prompt_mode: str = "local"):
    prompt_mode = prompt_mode.lower()

    if prompt_mode == "hub":
        try:
            return hub.pull("rlm/rag-prompt")
        except Exception as e:
            print(f"[!] hub.pull 실패. local prompt로 대체. 원인: {e}")

    return ChatPromptTemplate.from_template(
        """
        다음 문서를 참고해 질문에 답하시오.

        [문서 내용]
        {context}

        [질문]
        {question}

        [답변]
        """.strip()
    )


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever, llm, prompt_mode: str = "local"):
    prompt = create_prompt(prompt_mode)

    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def preview_docs(docs: List[Document], max_chars: int = 300):
    for i, doc in enumerate(docs, start=1):
        print(f"\n[{i}] source={doc.metadata.get('source', 'unknown')}")
        print(doc.page_content[:max_chars])