from g2p_en import G2p
from utils.text_normalize import split_words

g2p = G2p()

ARPABET_TO_IPA = {
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ʌ",
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "EH": "e",
    "ER": "ɝ",
    "EY": "eɪ",
    "F": "f",
    "G": "g",
    "HH": "h",
    "IH": "ɪ",
    "IY": "iː",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "P": "p",
    "R": "r",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "UH": "ʊ",
    "UW": "uː",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}


def arpabet_to_ipa(phones):
    ipa = []

    for p in phones:
        base = ''.join([c for c in p if not c.isdigit()])
        ipa.append(ARPABET_TO_IPA.get(base, base.lower()))

    return "".join(ipa)


def get_sentence_phonetics(text: str):

    words = split_words(text)

    result_words = []
    ipa_sentence = []

    for w in words:

        phones = g2p(w)

        arpabet = " ".join(phones)
        ipa = arpabet_to_ipa(phones)

        result_words.append({
            "word": w,
            "arpabet": arpabet,
            "ipa": ipa
        })

        ipa_sentence.append(ipa)

    return {
        "words": result_words,
        "ipa_sentence": " ".join(ipa_sentence)
    }