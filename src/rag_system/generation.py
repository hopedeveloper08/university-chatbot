from langchain_core.prompts import ChatPromptTemplate


def get_prompt_template(prompt_path):
    with open(prompt_path, 'r', encoding='utf-8') as f:
        system_message = f.read()

    return ChatPromptTemplate.from_messages([
        ('system', system_message),
    ])


def generator():
    pass
