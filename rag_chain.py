import bs4
import logging
from langchain_classic import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

def load_naver_news(url):
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                "div",
                attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},
            )
        ),
    )
    return loader.load()

def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    return text_splitter.split_documents(docs)

def create_vectorstore(splits, embeddings):
    return FAISS.from_documents(documents=splits, embedding=embeddings)

def create_retriever(vectorstore, splits, llm, mode="similarity", k=3):
    if mode == "threshold":
        return vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={
            "score_threshold": 0.5, 
            "k": k
            })
    
    elif mode == "mmr":
        return vectorstore.as_retriever(search_type="mmr", search_kwargs={
            "k": k,
            "fetch_k": 20,
            "lambda_mult": 0.5
            })
    
    elif mode == "multi_query":
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        logging.basicConfig()
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
        return MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
    
    elif mode == "ensemble":
        bm25_retriever = BM25Retriever.from_documents(splits)
        bm25_retriever.k = k
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        return EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5], c=2)
    
    else:
        return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

def build_rag_chain(retriever, llm):
    prompt = hub.pull("rlm/rag-prompt")
    format_docs = lambda docs: "\n\n".join(doc.page_content for doc in docs)
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt 
        | llm 
        | StrOutputParser()
    )