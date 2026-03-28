from difflib import SequenceMatcher
from utils.text_normalize import split_words


def align_target_and_recognized(target_text: str, recognized_text: str) -> dict:
    target_words = split_words(target_text)
    recognized_words = split_words(recognized_text)

    matcher = SequenceMatcher(None, target_words, recognized_words)
    ops = matcher.get_opcodes()

    alignment = []
    for tag, i1, i2, j1, j2 in ops:
        alignment.append({
            "tag": tag,
            "target_words": target_words[i1:i2],
            "recognized_words": recognized_words[j1:j2],
            "target_range": [i1, i2],
            "recognized_range": [j1, j2]
        })

    return {
        "target_words": target_words,
        "recognized_words": recognized_words,
        "alignment": alignment
    }