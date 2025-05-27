import yaml


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        x = yaml.full_load(f)
    return x
