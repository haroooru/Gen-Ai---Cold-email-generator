import os
import streamlit as st
from utils import clean_text

try:
    from langchain_community.document_loaders import WebBaseLoader
except Exception:
    WebBaseLoader = None

from chains import Chain
from portfolio import Portfolio

st.set_page_config(page_title="Gen-AI Cold Email Generator", layout="wide")

PROJECT_DESCRIPTION = "Generative AI tool to help software & AI services companies send cold emails to potential clients."

def load_text_from_url(url: str) -> str:
    if not WebBaseLoader:
        raise RuntimeError("URL loader not available. Install langchain-community.")
    loader = WebBaseLoader(url)
    try:
        docs = loader.load()
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"{e}. Install `beautifulsoup4` and `lxml`.") from e
    return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

def main():
    st.title("Gen-AI Cold Email Generator")
    st.markdown(PROJECT_DESCRIPTION)
    col1, col2 = st.columns([2,1])
    with col1:
        url = st.text_input("Careers page URL (optional)")
        pasted = st.text_area("Or paste scraped text here (optional)", height=300)
        generate = st.button("Generate")
    with col2:
        st.write("Settings")
        sender_name = st.text_input("Sender name", value="Hari")
        use_llm = st.checkbox("Use Groq LLM if available", value=True)
        show_raw = st.checkbox("Show extracted jobs", value=False)

    if generate:
        page_text = (pasted or "").strip()
        if not page_text and url:
            if not WebBaseLoader:
                st.error("URL loading not available in this environment. Paste text instead.")
                return
            with st.spinner("Loading page..."):
                try:
                    page_text = load_text_from_url(url)
                except Exception as e:
                    st.error(f"Failed to load URL: {e}")
                    return

        if not page_text:
            st.error("Provide a URL or paste text.")
            return

        cleaned = clean_text(page_text)
        chain = Chain()
        if not use_llm:
            chain.llm = None

        with st.spinner("Extracting jobs..."):
            jobs = chain.extract_jobs(cleaned)

        if show_raw:
            st.subheader("Raw extracted jobs")
            st.json(jobs)

        port = Portfolio()
        st.write(f"Found {len(jobs)} job(s).")
        for i, job in enumerate(jobs, start=1):
            st.divider()
            st.subheader(f"Job #{i}: {job.get('role','Unknown')}")
            st.write("Experience:", job.get("experience","Not specified"))
            st.write("Skills:", job.get("skills", []))
            st.write("Description:")
            st.write(job.get("description","")[:400] + ("..." if len(job.get("description",""))>400 else ""))

            try:
                links = port.query_links(job.get("skills", []))
            except Exception:
                links = []

            email = chain.write_mail(job, links, sender_name=sender_name)
            st.markdown("**Generated email**")
            st.code(email)

if __name__ == "__main__":
    main()



