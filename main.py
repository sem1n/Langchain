import argparse
import os

from config import (
    create_llm,
    create_embeddings,
    get_retriever_config,
    get_app_config,
)
from rag_chain import (
    load_documents,
    split_documents,
    create_vectorstore,
    create_retriever,
    build_rag_chain,
    preview_docs,
)


def main():
    retriever_cfg = get_retriever_config()
    app_cfg = get_app_config()

    parser = argparse.ArgumentParser(description="Naver News / Document RAG System")

    parser.add_argument(
        "--source",
        type=str,
        default="https://n.news.naver.com/article/437/0000378416",
        help="분석할 URL 또는 파일/폴더 경로",
    )

    parser.add_argument(
        "--source-type",
        type=str,
        default="naver",
        choices=["naver", "web", "pdf", "csv", "python"],
        help="문서 타입",
    )

    parser.add_argument(
        "--question",
        type=str,
        default="이 뉴스 기사의 핵심 내용을 요약해줘.",
        help="문서에 대해 궁금한 점",
    )

    parser.add_argument(
        "--retriever",
        type=str,
        default=retriever_cfg["mode"],
        choices=["similarity", "threshold", "mmr", "multi_query", "ensemble"],
        help="리트리버 검색 모드",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=retriever_cfg["k"],
        help="검색할 문서 청크 개수",
    )

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=retriever_cfg["score_threshold"],
        help="threshold 검색에서 사용할 최소 유사도 점수",
    )

    parser.add_argument(
        "--fetch-k",
        type=int,
        default=retriever_cfg["fetch_k"],
        help="MMR 검색에서 후보로 가져올 문서 수",
    )

    parser.add_argument(
        "--lambda-mult",
        type=float,
        default=retriever_cfg["lambda_mult"],
        help="MMR 다양성 조절값",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=app_cfg["chunk_size"],
        help="청크 크기",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=app_cfg["chunk_overlap"],
        help="청크 겹침 크기",
    )

    parser.add_argument(
        "--splitter",
        type=str,
        default=app_cfg["splitter_mode"],
        choices=["recursive", "character"],
        help="문서 분할 방식",
    )

    parser.add_argument(
        "--vectorstore",
        type=str,
        default=app_cfg["vectorstore_mode"],
        choices=["faiss", "chroma"],
        help="벡터스토어 종류",
    )

    parser.add_argument(
        "--persist-dir",
        type=str,
        default=app_cfg["persist_directory"],
        help="Chroma 사용 시 저장 폴더",
    )

    parser.add_argument(
        "--prompt-mode",
        type=str,
        default=app_cfg["prompt_mode"],
        choices=["local", "hub"],
        help="프롬프트 모드",
    )

    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="답변 전에 검색된 청크를 출력",
    )

    args = parser.parse_args()

    print(f"[*] 문서 로드 중")
    print(f"    - source_type: {args.source_type}")
    print(f"    - source: {args.source}")

    docs = load_documents(
        source=args.source,
        source_type=args.source_type,
    )

    if not docs:
        print("[!] 문서를 불러오지 못했습니다. URL 또는 파일 경로를 확인하세요.")
        return

    print(f"[*] 문서 로드 완료: {len(docs)}개")

    splits = split_documents(
        docs,
        splitter_mode=args.splitter,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    if not splits:
        print("[!] 문서 분할 결과가 없습니다.")
        return

    print(f"[*] 문서 분할 완료: {len(splits)} 청크")
    print(f"    - splitter: {args.splitter}")
    print(f"    - chunk_size: {args.chunk_size}")
    print(f"    - chunk_overlap: {args.chunk_overlap}")

    embeddings = create_embeddings()
    print(f"[*] 임베딩 모델 생성 완료: {os.getenv('EMBEDDING_MODE', 'fastembed')}")

    vectorstore = create_vectorstore(
        splits=splits,
        embeddings=embeddings,
        vectorstore_mode=args.vectorstore,
        persist_directory=args.persist_dir,
    )

    print(f"[*] 벡터스토어 생성 완료: {args.vectorstore}")

    llm = create_llm()
    print(f"[*] LLM 생성 완료: {os.getenv('MODEL_NAME')}")

    retriever = create_retriever(
        vectorstore=vectorstore,
        splits=splits,
        llm=llm,
        mode=args.retriever,
        k=args.k,
        score_threshold=args.score_threshold,
        fetch_k=args.fetch_k,
        lambda_mult=args.lambda_mult,
        bm25_weight=retriever_cfg["bm25_weight"],
        faiss_weight=retriever_cfg["faiss_weight"],
    )

    print(f"[*] Retriever 생성 완료: {args.retriever}")

    if args.show_sources:
        print("\n[검색된 문서 청크 미리보기]")
        print("-" * 50)
        retrieved_docs = retriever.invoke(args.question)
        preview_docs(retrieved_docs)

    rag_chain = build_rag_chain(
        retriever=retriever,
        llm=llm,
        prompt_mode=args.prompt_mode,
    )

    print(f"\n[질문]: {args.question}")
    print("-" * 50)

    try:
        response = rag_chain.invoke(args.question)
        print(f"[답변]\n{response}")

    except Exception as e:
        print(f"[!] 오류 발생: {e}")

if __name__ == "__main__":
    main()