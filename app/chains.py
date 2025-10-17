import os
import re
import json
from dotenv import load_dotenv

load_dotenv()

# try LLM / prompt libraries
try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None

try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
except Exception:
    PromptTemplate = None
    JsonOutputParser = None

# bs4 optional for HTML parsing fallback
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except Exception:
    BeautifulSoup = None
    BS4_AVAILABLE = False

ROLE_KEYWORDS = r"(engineer|developer|manager|analyst|designer|intern|scientist|specialist|consultant|associate|sales|product|security|research|backend|frontend)"
NAV_NOISE = [r"skip to main content", r"filter results", r"select a language", r"clear filter", r"menu", r"brand"]

def _is_noise(text: str) -> bool:
    low = text.lower()
    if len(re.findall(r"\w+", text)) < 4:
        return True
    for p in NAV_NOISE:
        if re.search(p, low):
            return True
    return False

def _extract_skills_block(text: str):
    m = re.search(r"(?is)(requirements|qualifications|skills|you should have|what we're looking for)[\s:.-]*(.*)", text)
    skills = []
    if m:
        tail = text[m.end(1):]
        bullets = re.findall(r"(?m)^[\-\u2022\*\•]\s*(.+)$", tail)
        if bullets:
            skills = [re.sub(r"[^\w\s\+\#\.\-]","",b).strip().lower() for b in bullets if b.strip()]
        else:
            lines = [l.strip() for l in tail.splitlines() if l.strip()]
            candidate = " ".join(lines[:4])
            skills = [s.strip().lower() for s in re.split(r"[;,/]| and | or ", candidate) if s.strip()]
    if not skills:
        inline = re.findall(r"(?i)(?:skills?|requirements?)[:\s]*([A-Za-z0-9 ,/\+\-#]+)", text)
        for part in inline[:2]:
            skills += [s.strip().lower() for s in re.split(r"[;,/]| and | or ", part) if s.strip()]
    # dedupe keep order
    uniq = []
    for s in skills:
        if s not in uniq:
            uniq.append(s)
    return uniq[:30]

class Chain:
    def __init__(self):
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.llm = None
        if ChatGroq and self.groq_key:
            try:
                self.llm = ChatGroq(temperature=0, groq_api_key=self.groq_key, model_name="llama-3.1-70b-versatile")
            except Exception:
                self.llm = None

        # prepare prompts if langchain_core available
        if PromptTemplate:
            self.prompt_extract = PromptTemplate.from_template(
                """
                ### SCRAPED TEXT FROM WEBSITE:
                {page_data}
                ### INSTRUCTION:
                Extract job postings from the input. Return a VALID JSON array of objects with keys:
                - role (string)
                - experience (string or empty)
                - skills (list of strings)
                - description (string)
                ONLY RETURN JSON (no commentary).
                """
            )
            self.prompt_email = PromptTemplate.from_template(
                """
                ### JOB JSON:
                {job_json}
                ### PORTFOLIO LINKS:
                {link_list}
                ### INSTRUCTION:
                You are Hari, Business Development Executive at AtliQ (AI & Software consulting).
                Write a concise professional cold email (subject line then body). 
                - Highlight 2–3 most relevant skills/experience from the job JSON.
                - Include the most relevant portfolio links from the link list.
                - Keep the email short, specific and polite.
                Output ONLY the email text (subject + body).
                """
            )

    def extract_jobs(self, page_text: str):
        page_text = (page_text or "").strip()
        # 1) LLM-first extraction
        if self.llm and PromptTemplate:
            try:
                chain = self.prompt_extract | self.llm
                res = chain.invoke({"page_data": page_text})
                content = getattr(res, "content", str(res)).strip()
                # parse JSON safely
                start = content.find("[")
                end = content.rfind("]") + 1
                if start != -1 and end != -1:
                    parsed = json.loads(content[start:end])
                    if isinstance(parsed, list) and parsed:
                        norm = []
                        for o in parsed:
                            norm.append({
                                "role": o.get("role","").strip(),
                                "experience": o.get("experience","").strip(),
                                "skills": [s.strip() for s in o.get("skills",[])],
                                "description": o.get("description","").strip()
                            })
                        return norm
            except Exception:
                pass

        # 2) HTML parse fallback to get candidate blocks
        blocks = []
        if BS4_AVAILABLE:
            try:
                soup = BeautifulSoup(page_text, "html.parser")
                cand = []
                for tag in soup.find_all(['article','li','div','section']):
                    txt = tag.get_text(separator="\n").strip()
                    if not txt or len(txt) < 30:
                        continue
                    if any(m in txt.lower() for m in ["view job","apply","position","role","careers","openings"]):
                        cand.append(txt)
                if not cand:
                    for h in soup.find_all(['h1','h2','h3','h4']):
                        parent = h.find_parent()
                        if parent:
                            txt = parent.get_text(separator="\n").strip()
                            if len(txt) > 40:
                                cand.append(txt)
                blocks = cand
            except Exception:
                blocks = []

        # 3) text heuristics fallback
        if not blocks:
            parts = re.split(r"\n{2,}", page_text)
            for p in parts:
                p = p.strip()
                if not p or len(p) < 40:
                    continue
                if _is_noise(p):
                    continue
                blocks.append(p)

        jobs = []
        seen = set()
        for b in blocks:
            if _is_noise(b):
                continue
            lines = [l.strip() for l in b.splitlines() if l.strip()]
            if not lines:
                continue
            title = ""
            for ln in lines[:6]:
                if re.search(ROLE_KEYWORDS, ln, flags=re.I) and 3 < len(ln) < 140:
                    title = ln
                    break
            if not title:
                title = lines[0]
            title = re.sub(r"\s+", " ", title).strip()
            key = re.sub(r"\W+"," ", title).strip().lower()
            if not title or key in seen:
                continue
            seen.add(key)

            desc_lines = lines.copy()
            if desc_lines and desc_lines[0].strip() == title.strip():
                desc_lines = desc_lines[1:]
            desc = " ".join(desc_lines).strip()
            if len(desc) > 2000:
                desc = desc[:2000] + "..."

            mexp = re.search(r"(\d+\+?\s+years|\d+\s+years|mid[\- ]level|senior|junior|entry|intern)", b, flags=re.I)
            exp = mexp.group(0) if mexp else "Not specified"

            skills = _extract_skills_block(b)
            if not skills:
                # try by keywords in title / block
                common = {"python","java","aws","sql","react","docker","kubernetes","ml","ai","devops","sales","design","javascript"}
                found = [t.lower() for t in re.findall(r"[A-Za-z\+#\-]{3,}", title) if t.lower() in common]
                if not found:
                    lowb = b.lower()
                    found = [sk for sk in common if re.search(r"\b" + re.escape(sk) + r"\b", lowb)]
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
                "description": page_text[:1500]
            })
        return jobs

    def write_mail(self, job: dict, links: list, sender_name: str = "Hari"):
        # LLM-generated email if available
        job_json = json.dumps(job, ensure_ascii=False)
        links_text = "\n".join(links) if links else ""
        if self.llm and PromptTemplate:
            try:
                chain = self.prompt_email | self.llm
                res = chain.invoke({"job_json": job_json, "link_list": links_text})
                content = getattr(res, "content", str(res)).strip()
                return content
            except Exception:
                pass

        # fallback simple generator (varied)
        import random, textwrap
        role = job.get("role","the role")
        skills = job.get("skills") or []
        top = skills[:3]
        skill_str = ", ".join(top) if top else "relevant experience"
        desc = job.get("description","")
        snippet = re.sub(r"\s+"," ", desc)[:140].strip() or "short summary available on request"
        subj_templates = ["Quick intro — {role}", "Interest in {role}", "Portfolio for {role}"]
        subject = random.choice(subj_templates).format(role=role)
        templates = [
            "Hi,\n\nI saw the {role} opening. I specialize in {skills}. A quick example: \"{snippet}\". Would you be open to a short call?",
            "Hello,\n\nI'm reaching out about the {role} role. My background includes {skills}. I can share relevant demos on request.",
            "Hi,\n\nNoticed the {role} role. I have hands-on experience with {skills}. Briefly: {snippet}. Happy to connect for 15 mins."
        ]
        body = random.choice(templates).format(role=role, skills=(skill_str), snippet=snippet)
        links_block = ("\n\nRelevant work samples:\n" + "\n".join(f"- {u}" for u in links[:5])) if links else "\n\nI can share work samples on request."
        closing = f"\n\nBest regards,\n{sender_name}\n[your-email@example.com]"
        email_text = f"Subject: {subject}\n\n{body}{links_block}{closing}"
        return "\n".join(textwrap.fill(line, width=100) if len(line) > 120 else line for line in email_text.splitlines())
