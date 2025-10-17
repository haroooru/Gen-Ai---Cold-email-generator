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
        raise ModuleNotFoundError(
            f"{e}. Install missing HTML parser packages: `beautifulsoup4` and `lxml`."
        ) from e
    return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

def main():
    st.title("Cold Email Generator")
    st.markdown("Provide a careers page URL or paste the scraped text. Click Generate to extract jobs and create emails.")

    url = st.text_input("Careers page URL (optional)")
    pasted = st.text_area("Or paste scraped text here (optional)", height=200)
    generate = st.button("Generate")

    if generate:
        page_text = pasted.strip()
        if not page_text and url:
            if not WebBaseLoader:
                st.error("URL loading not available because langchain-community is not installed. Paste text instead.")
                return
            with st.spinner("Loading URL..."):
                try:
                    page_text = load_text_from_url(url)
                except ModuleNotFoundError as e:
                    st.error(str(e))
                    return
                except Exception as e:
                    st.error(f"Failed to load URL: {e}")
                    return

        if not page_text:
            st.error("No input provided. Paste text or supply a URL.")
            return

        cleaned = clean_text(page_text)
        chain = Chain()
        try:
            jobs = chain.extract_jobs(cleaned)
        except Exception as e:
            st.error(f"Job extraction failed: {e}")
            return

        port = Portfolio()
        results = []
        for job in jobs:
            links = port.query_links(job.get("skills", ""))
            email = chain.write_mail(job, links)
            results.append((job, links, email))

        for i, (job, links, email) in enumerate(results, start=1):
            st.divider()
            st.subheader(f"Job #{i}: {job.get('role', 'Unknown')}")
            st.write("Experience:", job.get("experience", "Not specified"))
            st.write("Skills:", job.get("skills", []))
            st.write("Description:")
            st.write(job.get("description", ""))
            st.write("Portfolio links:")
            for l in links:
                st.write(l)
            st.markdown("**Generated cold email:**")
            st.code(email)

if __name__ == "__main__":
    main()


