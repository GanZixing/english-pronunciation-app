import re


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_words(text: str) -> list:
    t = normalize_text(text)
    return t.split() if t else []