from langchain_community.vectorstores import FAISS


def create_vector_store(docs, embedding_model, save_path):
    vector_store = FAISS.from_documents(
        documents=docs,
        embedding=embedding_model,
    )
    vector_store.save_local(save_path)
    return vector_store


def load_vector_store(path, embedding_model):
    return FAISS.load_local(
        folder_path=path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
