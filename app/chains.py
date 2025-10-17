import os
import re
from dotenv import load_dotenv

load_dotenv()

# Lightweight Chain: will try to use ChatGroq if installed, otherwise fallback to heuristic parser
try:
    from langchain_groq import ChatGroq
    LLM_AVAILABLE = True
except Exception:
    ChatGroq = None
    LLM_AVAILABLE = False

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
        Return a list of job dicts with keys: role, experience, skills (list), description.
        If an LLM is available it will be used, otherwise a heuristic extractor runs.
        """
        if self.llm:
            # If an LLM object exists, try a simple prompt invoke if supported.
            try:
                prompt = (
                    "You are given scraped text from a careers page. Extract job postings as JSON list "
                    "with keys role, experience, skills (list) and description. Only return valid JSON.\n\n"
                    f"{cleaned_text}"
                )
                # try common LLM call methods
                if hasattr(self.llm, "invoke"):
                    resp = self.llm.invoke(prompt)
                    text = getattr(resp, "text", str(resp))
                elif hasattr(self.llm, "chat"):
                    resp = self.llm.chat([{"role": "user", "content": prompt}])
                    text = resp[0]["content"]
                else:
                    text = str(self.llm(prompt))
                # attempt simple JSON extraction from text
                import json
                start = text.find("[")
                end = text.rfind("]") + 1
                if start != -1 and end != -1:
                    return json.loads(text[start:end])
            except Exception:
                # fall back to heuristic
                pass

        # Heuristic extractor (non-LLM): simple regex and keyword matching
        lines = [l.strip() for l in cleaned_text.splitlines() if l.strip()]
        text = " ".join(lines)
        # split into candidate blocks by common separators
        blocks = re.split(r"\n{2,}|-{3,}|={3,}", cleaned_text)
        candidates = []
        skill_keywords = [
            "python", "javascript", "react", "aws", "docker", "sql", "java", "c#", "node", "golang",
            "machine learning", "nlp", "data", "excel", "communication", "leadership"
        ]
        role_patterns = r"(engineer|developer|manager|analyst|designer|intern|scientist|specialist|consultant)"
        for b in blocks:
            if re.search(role_patterns, b, flags=re.I):
                # role: first occurrence of pattern
                m = re.search(r"([A-Z][A-Za-z0-9 &/-]{1,60}?(?:engineer|developer|manager|analyst|designer|intern|scientist|specialist|consultant))", b, flags=re.I)
                role = m.group(1).strip() if m else re.search(role_patterns, b, flags=re.I).group(0).title()
                # skills found
                found_skills = []
                for sk in skill_keywords:
                    if re.search(r"\b" + re.escape(sk) + r"\b", b, flags=re.I):
                        found_skills.append(sk)
                candidates.append({
                    "role": role,
                    "experience": "Not specified",
                    "skills": found_skills,
                    "description": b.strip()
                })
        # if nothing found, create one generic job
        if not candidates:
            candidates.append({
                "role": "Unknown role",
                "experience": "Not specified",
                "skills": [],
                "description": text[:1000]
            })
        return candidates

    def write_mail(self, job: dict, links: list):
        """
        Produce a short cold email string using job info and portfolio links.
        """
        role = job.get("role", "the role")
        skills = job.get("skills", [])
        skill_str = ", ".join(skills) if skills else "relevant skills"
        intro = f"Hi,\n\nI saw the {role} opening on your careers page and wanted to introduce myself."
        body = f"I have experience with {skill_str} and believe I can contribute to your team."
        if links:
            body += "\n\nRelevant work samples:\n" + "\n".join(f"- {u}" for u in links)
        closing = "\n\nWould love to discuss how I can help. Best regards,\n[Your Name]\n[Your Email]"
        return "\n\n".join([intro, body, closing])
