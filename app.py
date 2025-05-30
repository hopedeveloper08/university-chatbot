from src.rag_system import build_rag_chain


def main():
    print('Loading...')
    rag_chain = build_rag_chain(
        llm_config_path='config/generator.yaml',
        retriever_config_path='config/retriever.yaml',
        prompt_path='config/prompt_template.txt',
        vector_store_path='data/collection/',
        embedding_model_config_path='config/embedding.yaml',
    )

    user_input = input('ask your question: ')
    response = rag_chain.invoke(user_input)['result']

    print()
    print(response)
    with open('output.txt', 'w', encoding='utf-8') as f:
        f.write(response)


if __name__ == '__main__':
    main()
