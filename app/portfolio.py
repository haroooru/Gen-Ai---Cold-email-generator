import os
import pandas as pd

class Portfolio:
    def __init__(self, file_path=None):
        # default path in repo
        default = os.path.join(os.path.dirname(__file__), "resources", "my_portfolio.csv")
        self.file_path = file_path or default
        if os.path.exists(self.file_path):
            try:
                self.data = pd.read_csv(self.file_path, dtype=str).fillna("")
            except Exception:
                self.data = pd.DataFrame(columns=["title", "url", "skills"])
        else:
            self.data = pd.DataFrame(columns=["title", "url", "skills"])

    def query_links(self, skills):
        """
        skills: string or list; returns list of urls from the CSV whose skills overlap.
        CSV expected columns: title,url,skills (skills as comma-separated)
        """
        if not isinstance(skills, (list, tuple)):
            skills = [s.strip().lower() for s in str(skills).split(",") if s.strip()]
        results = []
        for _, row in self.data.iterrows():
            row_skills = [s.strip().lower() for s in str(row.get("skills", "")).split(",") if s.strip()]
            if not skills:
                # return everything if no skills specified
                if row.get("url"):
                    results.append(row.get("url"))
                continue
            if any(any(sk in rs for rs in row_skills) or any(rs in sk for rs in row_skills) for sk in skills):
                if row.get("url"):
                    results.append(row.get("url"))
        # deduplicate and limit
        uniq = []
        for u in results:
            if u not in uniq:
                uniq.append(u)
        return uniq[:10]
