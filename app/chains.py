# ...existing code...
import os
import re
import json
from dotenv import load_dotenv

load_dotenv()

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except Exception:
    BeautifulSoup = None
    BS4_AVAILABLE = False

try:
    from langchain_groq import ChatGroq
    LLM_AVAILABLE = True
except Exception:
    ChatGroq = None
    LLM_AVAILABLE = False

ROLE_KEYWORDS = r"(engineer|developer|manager|analyst|designer|intern|scientist|specialist|consultant|associate|partner|sales|support|marketing|product|retail|store)"
NAV_NOISE_PATTERNS = [
    r"skip to main content", r"filter results", r"select a language", r"go to first page",
    r"go to next page", r"page \d+", r"clear filter", r"menu", r"language", r"brand"
]
SPLIT_MARKERS = [r"View Job", r"Apply", r"Job", r"Position", r"Openings", r"Role", r"Careers"]

def _clean_role_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(?i)\b(view job|apply|learn more|see details|skip to main content)\b.*", "", s)
    s = re.sub(r"[-—|·\|]+\s*$", "", s).strip()
    s = re.sub(r"\s*\([^)]+\)\s*$", "", s).strip()
    return s

def _is_noise_block(text: str) -> bool:
    low = text.lower()
    # too short, or contains many navigation/filter tokens => noise
    if len(re.findall(r"\w+", text)) < 3:
        return True
    for p in NAV_NOISE_PATTERNS:
        if re.search(p, low):
            return True
    # contains many numbers/controls typical of filters
    if len(re.findall(r"\d{2,}", text)) > 6 and len(low) > 200:
        return True
    return False

def _extract_skill_block(text: str):
    # look for requirements/qualifications/skills sections and bullets
    m = re.search(r"(?is)(requirements|qualifications|skills|you should have|what we're looking for|what we look for)[\s:.-]*(.*)", text)
    skills = []
    if m:
        tail = text[m.end(1):]
        bullets = re.findall(r"(?m)^[\-\u2022\*\•]\s*(.+)$", tail)
        if bullets:
            skills = [re.sub(r"[^\w\s\+\#\.\-]", "", b).strip() for b in bullets if len(b.strip()) > 1]
        else:
            # fallback: take short comma separated phrases from the next 4 lines
            lines = [l.strip() for l in tail.splitlines() if l.strip()]
            candidate = " ".join(lines[:4])
            skills = [s.strip() for s in re.split(r"[;,/]| and | or ", candidate) if len(s.strip())>1]
    # attempt to find inline "skills:" patterns
    if not skills:
        inline = re.findall(r"(?i)(?:skills?|requirements?)[:\s]*([A-Za-z0-9 ,/\+\-#]+)", text)
        if inline:
            skills = []
            for part in inline[:2]:
                skills += [s.strip() for s in re.split(r"[;,/]| and | or ", part) if len(s.strip())>1]
    # cleanup and lowercase, remove generic tokens
    stop = {"apply","job","openings","internships","filters","menu","learn more"}
    cleaned = []
    for s in skills:
        s2 = re.sub(r"\s{2,}", " ", s).strip().lower()
        s2 = re.sub(r"[^\w\s\+\#\.\-]","", s2)
        if len(s2) < 2 or s2 in stop:
            continue
        cleaned.append(s2)
    # dedupe preserve order
    uniq = []
    for it in cleaned:
        if it not in uniq:
            uniq.append(it)
    return uniq[:30]

def _extract_from_title_for_skills(title: str):
    # titles like "Sales Partner Experience Specialist" -> extract 'sales', 'partner', 'experience'
    parts = re.split(r"[\-|/,\:]", title)
    tokens = []
    for p in parts:
        for w in p.split():
            w = re.sub(r"[^\w\+#\-]", "", w).lower()
            if len(w) > 2 and not re.match(r"^\d+$", w):
                tokens.append(w)
    # common skill-like tokens
    common_skills = {"sales","retail","python","java","sql","aws","design","marketing","manager","engineer","developer","data","nlp","ai","ml"}
    found = [t for t in tokens if t in common_skills]
    return found

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
        # 1) try LLM (non-fatal)
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
                pass

        blocks = []
        # 2) HTML parsing with bs4 when available
        if BS4_AVAILABLE:
            try:
                soup = BeautifulSoup(cleaned_text, "html.parser")
                candidates = []
                # prefer elements that contain "view job" / "apply" etc
                for tag in soup.find_all(['article','li','div','section']):
                    txt = tag.get_text(separator="\n").strip()
                    if not txt or len(txt) < 20:
                        continue
                    if any(marker.lower() in txt.lower() for marker in ["view job","apply","position","role","careers"]):
                        candidates.append(txt)
                # heading-based fallback
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

        # 3) text heuristics fallback
        if not blocks:
            join_markers = "|".join([re.escape(m) for m in SPLIT_MARKERS])
            parts = re.split(rf"(?i)(?:{join_markers})", cleaned_text)
            if len(parts) < 2:
                parts = re.split(r"\n{2,}", cleaned_text)
            for p in parts:
                p = p.strip()
                if not p or len(p) < 30:
                    continue
                if _is_noise_block(p):
                    continue
                blocks.append(p)

        jobs = []
        seen = set()
        for b in blocks:
            if _is_noise_block(b):
                continue
            lines = [l.strip() for l in b.splitlines() if l.strip()]
            if not lines:
                continue
            # find candidate title line
            title = ""
            for ln in lines[:6]:
                if re.search(ROLE_KEYWORDS, ln, flags=re.I) and 3 < len(ln) < 120:
                    title = ln
                    break
            if not title:
                title = lines[0]
            title = _clean_role_text(title)
            norm = re.sub(r"\W+"," ", title).strip().lower()
            if not title or norm in seen:
                continue
            seen.add(norm)

            # description: prefer the block minus the title line
            desc_lines = lines.copy()
            if desc_lines and desc_lines[0].strip() == title.strip():
                desc_lines = desc_lines[1:]
            desc = " ".join(desc_lines).strip()
            if len(desc) > 2000:
                desc = desc[:2000] + "..."

            # experience detection
            exp = "Not specified"
            mexp = re.search(r"(\d+\+?\s+years|\d+\s+years|mid[\- ]level|senior|junior|entry|intern)", b, flags=re.I)
            if mexp:
                exp = mexp.group(0)

            # skills extraction
            skills = _extract_skill_block(b)
            if not skills:
                # try title-derived skills
                skills = _extract_from_title_for_skills(title)
            if not skills:
                # keyword scan
                skill_keywords = ["python","java","javascript","react","aws","sql","docker","kubernetes","excel","machine learning","nlp","data","sales","communication","leadership","design","retail"]
                found = []
                lowb = b.lower()
                for sk in skill_keywords:
                    if re.search(r"\b" + re.escape(sk) + r"\b", lowb):
                        found.append(sk)
                skills = found

            jobs.append({
                "role": title,
                "experience": exp,
                "skills": skills,
                "description": desc
            })

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
        # if still empty, try to extract a short focus phrase from description
        focus = ""
        if not top_skills:
            desc = job.get("description", "")
            m = re.search(r"(?i)(data|ai|ml|machine learning|nlp|sales|retail|design|cloud|backend|frontend|devops|python|java|javascript)", desc)
            if m:
                focus = m.group(0)
        skill_str = ", ".join(top_skills) if top_skills else (focus or "relevant experience")
        intro = f"Hi,\n\nI came across the {role} opening on your careers page and wanted to introduce myself."
        body = f"I have {skill_str} and experience that aligns with this role."
        if links:
            body += "\n\nRelevant work samples:\n" + "\n".join(f"- {u}" for u in links[:5])
        closing = "\n\nWould love to discuss how I can contribute to your team.\n\nBest regards,\n[Your Name]\n[Your Email]"
        return "\n\n".join([intro, body, closing])
# ...existing code...
