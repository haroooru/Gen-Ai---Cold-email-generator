import re

def clean_text(text: str) -> str:
    if text is None:
        return ""
    # simple cleaning: remove extra whitespace, weird control chars
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
