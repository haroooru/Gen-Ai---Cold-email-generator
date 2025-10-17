# ...existing code...
import os
import re
import json
from dotenv import load_dotenv

load_dotenv()

# optional HTML parser
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except Exception:
    BeautifulSoup = None
    BS4_AVAILABLE = False

# optional LLM (keeps fallback)
try:
    from langchain_groq import ChatGroq
    LLM_AVAILABLE = True
except Exception:
    ChatGroq = None
    LLM_AVAILABLE = False

ROLE_KEYWORDS = r"(engineer|developer|manager|analyst|designer|intern|scientist|specialist|consultant|associate|partner|sales|support|engineer|marketing|product)"
SPLIT_MARKERS = [r"View Job", r"Apply", r"Job", r"Position", r"Openings", r"Role", r"Careers"]

def _clean_role_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(?i)\b(view job|apply|learn more|see details)\b.*", "", s)
    s = re.sub(r"[-—|·\|]+\s*$", "", s).strip()
    s = re.sub(r"\s*\([^)]+\)\s*$", "", s).strip()
    return s

def _extract_skill_block(text: str):
    # look for "Requirements/Qualifications/Skills" and return following bullet lines
    m = re.search(r"(?is)(requirements|qualifications|skills|you should have|what we're looking for)[\s:.-]*(.*)", text)
    skills = []
    if m:
        tail = text[m.end(1):]
        # capture bullets near header
        bullets = re.findall(r"(?m)^[\-\u2022\*\•]\s*(.+)$", tail)
        if bullets:
            skills = [b.strip() for b in bullets if len(b.strip()) > 1]
        else:
            # fallback: take short comma separated phrases from the next 2 lines
            lines = tail.splitlines()
            if lines:
                candidate = " ".join(lines[:3])
                skills = [s.strip() for s in re.split(r"[;,/]| and | or ", candidate) if len(s.strip())>1]
    # final cleanup & lowercase
    return [s.lower() for s in skills][:20]

class Chain:
    def __init__(self):
        self.llm = None
        if LLM_AVAILABLE:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                try:
                    self.llm = ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama-3.1-70b-versatile")
                except Exception:
                    self.llm = None

    def extract_jobs(self, cleaned_text: str):
        """
        Attempt structured extraction:
        1) If LLM available, try LLM first (best-effort).
        2) If bs4 available, parse HTML-like content and extract job-card blocks.
        3) Otherwise use improved text heuristics: split on markers, extract title, skills and description.
        Returns list of dicts: {role, experience, skills, description}
        """
        # 1) LLM attempt (non-fatal)
        if self.llm:
            try:
                prompt = (
                    "Extract job postings from the text below as a JSON list of objects with keys: "
                    "role, experience, skills (list) and description. Only output JSON.\n\n"
                    f"{cleaned_text}"
                )
                if hasattr(self.llm, "invoke"):
                    resp = self.llm.invoke(prompt)
                    text = getattr(resp, "text", str(resp))
                else:
                    text = str(self.llm(prompt))
                start = text.find("[")
                end = text.rfind("]") + 1
                if start != -1 and end != -1:
                    parsed = json.loads(text[start:end])
                    if isinstance(parsed, list) and parsed:
                        return parsed
            except Exception:
                pass  # fall through to heuristics

        blocks = []
        # 2) If bs4 available, try to isolate job cards
        if BS4_AVAILABLE:
            try:
                soup = BeautifulSoup(cleaned_text, "html.parser")
                # common patterns: articles, list items, divs with job-like text
                candidates = []
                for tag in soup.find_all(['article', 'li', 'div', 'section']):
                    txt = tag.get_text(separator="\n").strip()
                    if not txt or len(txt) < 20:
                        continue
                    low = txt.lower()
                    if any(marker.lower() in low for marker in ["view job", "apply", "job", "position", "careers"]):
                        candidates.append(txt)
                # if none found, fallback to heading-based discovery
                if not candidates:
                    for h in soup.find_all(['h1','h2','h3','h4']):
                        parent = h.find_parent()
                        if parent:
                            txt = parent.get_text(separator="\n").strip()
                            if len(txt) > 30:
                                candidates.append(txt)
                blocks = candidates
            except Exception:
                blocks = []

        # 3) Text heuristics if no bs4 result
        if not blocks:
            # Normalize newlines and split by strong markers
            join_markers = "|".join([re.escape(m) for m in SPLIT_MARKERS])
            parts = re.split(rf"(?i)(?:{join_markers})", cleaned_text)
            # If splitting too coarse, also split by multiple newlines
            if len(parts) < 2:
                parts = re.split(r"\n{2,}", cleaned_text)
            for p in parts:
                p = p.strip()
                if not p or len(p) < 30:
                    continue
                # filter out big navigation blocks that look like filters
                if re.search(r"filter results|go to first page|go to next page|page \d+", p, flags=re.I):
                    continue
                blocks.append(p)

        jobs = []
        seen_titles = set()
        for b in blocks:
            # get first candidate title line
            lines = [l.strip() for l in b.splitlines() if l.strip()]
            title = ""
            # prefer short lines with role keywords
            for ln in lines[:6]:
                if re.search(ROLE_KEYWORDS, ln, flags=re.I) and 3 < len(ln) < 120:
                    title = ln
                    break
            if not title:
                # fallback: first non-empty line that's not too long
                title = lines[0] if lines else ""
            title = _clean_role_text(title)
            norm = re.sub(r"\W+", " ", title).strip().lower()
            if not title or norm in seen_titles:
                continue
            seen_titles.add(norm)

            # experience detection
            exp = "Not specified"
            mexp = re.search(r"(\d+\+?\s+years|\d+\s+years|mid[\- ]level|senior|junior|entry)", b, flags=re.I)
            if mexp:
                exp = mexp.group(0)

            # skills
            skills = _extract_skill_block(b)
            # final fallback: search for known skill keywords inline
            if not skills:
                skill_keywords = ["python","java","javascript","react","aws","sql","docker","kubernetes","excel","machine learning","nlp","data","sales","communication","leadership"]
                found = []
                lower_b = b.lower()
                for sk in skill_keywords:
                    if re.search(r"\b" + re.escape(sk) + r"\b", lower_b):
                        found.append(sk)
                skills = found

            desc = b
            # trim description length
            desc = re.sub(r"\s{2,}", " ", desc).strip()
            if len(desc) > 2000:
                desc = desc[:2000] + "..."

            jobs.append({
                "role": title,
                "experience": exp,
                "skills": skills,
                "description": desc
            })

        # if nothing found, return a single generic block
        if not jobs:
            jobs.append({
                "role": "Unknown role",
                "experience": "Not specified",
                "skills": [],
                "description": cleaned_text[:1500]
            })
        return jobs

    def write_mail(self, job: dict, links: list):
        role = job.get("role", "the role")
        skills = job.get("skills", []) or []
        top_skills = skills[:3]
        skill_str = ", ".join(top_skills) if top_skills else "relevant experience"
        intro = f"Hi,\n\nI came across the {role} opening on your careers page and wanted to introduce myself."
        body_lines = [
            f"I have {skill_str} and experience that aligns with this role.",
        ]
        if links:
            body_lines.append("Here are a few relevant work samples:")
            for u in links[:5]:
                body_lines.append(f"- {u}")
        body_lines.append("\nWould love to discuss how I can contribute to your team.")
        closing = "\n\nBest regards,\n[Your Name]\n[Your Email]"
        return "\n\n".join([intro, "\n".join(body_lines), closing])
# ...existing code...

