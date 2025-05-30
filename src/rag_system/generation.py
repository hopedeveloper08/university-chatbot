from langchain_core.prompts import ChatPromptTemplate

from langchain_ollama import ChatOllama

from src.utils import load_yaml


def get_prompt_template(prompt_path):
    with open(prompt_path, 'r', encoding='utf-8') as f:
        system_message = f.read()

    return ChatPromptTemplate.from_messages([
        ('system', system_message),
    ])


def get_llm(config_path):
    config = load_yaml(config_path)
    return ChatOllama(
        model=config['model'],
        temperature=config['temperature']
    )
