from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.load_yaml import load_yaml


def loading_dataset_texts(path):
    text_loader = TextLoader(path, encoding='utf-8')
    dataset = text_loader.load()
    return dataset


def chunking_text(config_path, data_path):
    config = load_yaml(config_path) 
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap'],
        separators=config['separators']
    )
    dataset = loading_dataset_texts(data_path)
    return recursive_splitter.split_documents(dataset)
