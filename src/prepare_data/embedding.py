from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.load_yaml import load_yaml


def load_embedding_model(config_path):
    config = load_yaml(path=config_path)
    return HuggingFaceEmbeddings(
        model_name=config['model_name'],
        model_kwargs={'device': config['device']}
    )
