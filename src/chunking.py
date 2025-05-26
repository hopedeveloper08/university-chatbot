from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def loading_dataset_texts(path):
    text_loader = TextLoader(path, encoding='utf-8')

    dataset = text_loader.load()
    return dataset


def chunking_text(path, chunk_size, chunk_overlap, separators):
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    dataset = loading_dataset_texts(path)
    return recursive_splitter.split_documents(dataset)
