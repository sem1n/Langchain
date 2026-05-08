from config import create_llm, create_embeddings
from rag_chain import (
    load_documents,
    split_documents,
    create_vectorstore,
    create_retriever,
    build_rag_chain,
)


NAVER_NEWS_URL = "https://n.news.naver.com/article/437/0000378416"
QUESTION = "이 뉴스 기사의 핵심 내용을 요약해줘."


def main():
    print("문서 로드 중")
    print(f"  - source: {NAVER_NEWS_URL}")

    docs = load_documents(NAVER_NEWS_URL)

    if not docs:
        print("문서를 불러오지 못했습니다. URL을 확인하세요.")
        return

    print(f"문서 로드 완료: {len(docs)}개")

    splits = split_documents(docs)

    if not splits:
        print("문서 분할 결과가 없습니다.")
        return

    print(f"문서 분할 완료: {len(splits)} 청크")

    embeddings = create_embeddings()
    print("임베딩 모델 생성 완료")

    vectorstore = create_vectorstore(
        splits=splits,
        embeddings=embeddings,
    )

    print("벡터스토어 생성 완료")

    retriever = create_retriever(vectorstore)
    print("Retriever 생성 완료")

    llm = create_llm()
    print("LLM 생성 완료")

    rag_chain = build_rag_chain(
        retriever=retriever,
        llm=llm,
    )

    print(f"\n질문: {QUESTION}")
    print("-" * 50)

    try:
        response = rag_chain.invoke(QUESTION)
        print(f"답변\n{response}")

    except Exception as e:
        print(f"오류 발생: {e}")


if __name__ == "__main__":
    main()