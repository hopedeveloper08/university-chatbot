from langchain_community.vectorstores import FAISS


def create_vector_store(docs, embedding_model, save_path):
    vector_store = FAISS.from_documents(
        documents=docs,
        embedding=embedding_model,
    )
    vector_store.save_local(save_path)
    return vector_store
