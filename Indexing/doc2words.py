import re

SPLIT_RGX = re.compile(r'\w+', re.U)


def extract_words(text):
    words = re.findall(SPLIT_RGX, text)
    return map(lambda s: s.lower(), words)
