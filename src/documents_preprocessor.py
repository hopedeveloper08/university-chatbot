import re


def clean_noise(text):
    text = re.sub(r'[^\S\r]{2,}', ' ', text)
    return text


def normalize_chars(text):
    text = text.replace('ي', 'ی').replace('ك', 'ک').replace('ة', 'ه').replace('ۀ', 'ه')
    text = re.sub(r'[ـ]', '', text)  
    return text


def normalize_punctuation(text):
    text = re.sub(r'[“”«»]', '"', text)
    text = re.sub(r"[’‘]", "'", text)
    text = re.sub(r'[\.\!]{2,}', '.', text)
    return text


def preprocessor(raw_text):
    text = raw_text
    text = clean_noise(text)
    text = normalize_chars(text)
    text = normalize_punctuation(text)
    return text
