import os 

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.load_yaml import load_yaml


def load_text(path):
    return TextLoader(path, encoding='utf-8').load()[0]


def chunking_text(config_path, dataset):
    config = load_yaml(config_path) 
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap'],
        separators=config['separators']
    )
    return recursive_splitter.split_documents(dataset)


def load_dataset(base_data_dir):
    file_names = os.listdir(base_data_dir)
    docs = list()
    for file_name in file_names:
        metadatas = file_name[:-4].split('-')
        metadata = {
            'category': metadatas[0],
            'audience': metadatas[1],
            'specific_audience': metadatas[2],
        }
        doc = load_text(os.path.join(base_data_dir, file_name))
        doc.metadata = metadata
        docs.append(doc)
    return docs
