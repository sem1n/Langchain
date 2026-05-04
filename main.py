import os
import argparse
from config import create_llm, create_embeddings, get_retriever_config
from rag_chain import (
    load_naver_news, 
    split_documents, 
    create_vectorstore, 
    create_retriever, 
    build_rag_chain
)

def main():
    parser = argparse.ArgumentParser(description="Naver News RAG System")
    parser.add_argument(
        "--url", 
        type=str, 
        default="https://n.news.naver.com/article/437/0000378416",
        help="분석할 네이버 뉴스 URL"
    )
    parser.add_argument(
        "--question", 
        type=str, 
        default="이 뉴스 기사의 핵심 내용을 요약해줘.",
        help="뉴스에 대해 궁금한 점"
    )
    
    cfg = get_retriever_config()
    
    parser.add_argument(
        "--retriever",
        type=str,
        default=cfg["mode"],
        choices=["similarity", "threshold", "mmr", "multi_query", "ensemble"],
        help="리트리버 검색 모드"
    )
    parser.add_argument(
        "--k", 
        type=int, 
        default=cfg["k"],
        help="검색할 문서 청크 개수"
    )

    args = parser.parse_args()

    print(f"[*] 뉴스 로드 중: {args.url}")
    docs = load_naver_news(args.url)
    
    if not docs:
        print("[!] 문서를 불러오지 못했습니다. URL을 확인하세요.")
        return

    splits = split_documents(docs)
    print(f"[*] 문서 분할 완료 ({len(splits)} 청크)")

    embeddings = create_embeddings()
    vectorstore = create_vectorstore(splits, embeddings)
    print(f"[*] 임베딩 완료 (모드: {os.getenv('EMBEDDING_MODE', 'fastembed')})")

    llm = create_llm()
    retriever = create_retriever(
        vectorstore=vectorstore, 
        splits=splits, 
        llm=llm, 
        mode=args.retriever, 
        k=args.k
    )

    rag_chain = build_rag_chain(retriever, llm)
    
    print(f"\n[질문]: {args.question}")
    print("-" * 50)
    
    try:
        response = rag_chain.invoke(args.question)
        print(f"[답변]:\n{response}")
    except Exception as e:
        print(f"[!] 오류 발생: {e}")

if __name__ == "__main__":
    main()