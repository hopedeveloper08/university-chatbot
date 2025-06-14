from .chunking import load_dataset, chunking_text
from .embedding import load_embedding_model
from .vectorstore import create_vector_store


def prepare_data(data_path, chunking_config, embedding_config, save_path):
    docs = chunking_text(chunking_config, dataset=load_dataset(data_path))
    embedding_model = load_embedding_model(embedding_config)
    return create_vector_store(docs=docs, embedding_model=embedding_model, save_path=save_path)
