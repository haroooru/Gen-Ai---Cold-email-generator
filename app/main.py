# ...existing code...
import os
import streamlit as st

from utils import clean_text

try:
    from langchain_community.document_loaders import WebBaseLoader
except Exception:
    WebBaseLoader = None

from chains import Chain
from portfolio import Portfolio

def load_text_from_url(url: str) -> str:
    if not WebBaseLoader:
        raise RuntimeError("WebBaseLoader not available. Paste text instead or install langchain-community.")
    loader = WebBaseLoader(url)
    try:
        docs = loader.load()
    except ModuleNotFoundError as e:
        # surface helpful message for missing bs4/lxml
        raise ModuleNotFoundError(
            f"{e}. Install missing HTML parser packages: `beautifulsoup4` and `lxml`."
        ) from e
    return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)
# ...existing code...


