from langchain.chains import RetrievalQA

from src.rag_system import load_embedding_model, load_vector_store
from src.rag_system import get_retriver, get_prompt_template, get_llm


def build_rag_chain(llm_config_path, retriever_config_path, prompt_path, vector_store_path, embedding_model_config_path):
    embedding_model = load_embedding_model(embedding_model_config_path)
    vector_store = load_vector_store(vector_store_path, embedding_model)
    retriever = get_retriver(retriever_config_path, vector_store)
    prompt = get_prompt_template(prompt_path)
    llm = get_llm(llm_config_path)

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    return rag_chain
