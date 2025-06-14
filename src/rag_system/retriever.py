from src.utils import load_yaml

def get_retriever(config_path, vector_store, metadata_filter):
    metadata_filter = {'$or': metadata_filter}
    config = load_yaml(config_path)
    search_kwargs = config['search_kwargs']
    search_kwargs["filter"] = metadata_filter
    return vector_store.as_retriever(
        search_type=config['search_type'],
        search_kwargs=search_kwargs
    )
