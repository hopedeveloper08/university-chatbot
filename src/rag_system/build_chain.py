from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from .embedding import load_embedding_model 
from .vectorstore import load_vector_store
from .retriever import get_retriver
from .generation import get_prompt_template, get_llm


def build_rag_chain(llm_config_path, retriever_config_path, prompt_path, vector_store_path, embedding_model_config_path):
    embedding_model = load_embedding_model(embedding_model_config_path)
    vector_store = load_vector_store(vector_store_path, embedding_model)
    retriever = get_retriver(retriever_config_path, vector_store)
    prompt = get_prompt_template(prompt_path)
    llm = get_llm(llm_config_path)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return rag_chain
