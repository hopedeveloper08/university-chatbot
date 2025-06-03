from langchain_community.vectorstores import FAISS


def load_vector_store(path, embedding_model):
    return FAISS.load_local(
        folder_path=path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
