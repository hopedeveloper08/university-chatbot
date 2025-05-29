from src.utils import load_yaml

def get_retriver(config_path, vector_store):
    config = load_yaml(config_path)
    return vector_store.as_retriever(
        search_type=config['search_type'],
        search_kwargs=config['search_kwargs']
    )
