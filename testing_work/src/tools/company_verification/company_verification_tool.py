"""
company_verification_tool.py
────────────────────────────
Verifies whether a company posting a job is legitimate or fraudulent.

Scoring architecture
────────────────────
The final fraud score is built from three independent signal groups:

  [A] Web signals    — what search results say about the company
  [B] Domain signals — how the job posting's linked URL behaves
  [C] Job-text flags — hard red-flags found inside the job description

Each group produces a score in [-1.0, +1.0]:
  negative = legitimacy evidence
  positive  = fraud evidence

Final score:
  blended = normalise(0.65*A + 0.35*B)   — then soft-centred to avoid
                                            false extremes from the blend alone
  final   = blended + C_hard_bonuses      — job-text signals are ground truth
                                            and bypass the blend

Labels:
  < 0.38  → legit
  0.38–0.60 → suspicious
  > 0.60  → fraud

Confidence reflects BOTH the evidence count AND score extremity,
so it won't be falsely high when signals cancel each other out.
"""

import re
import whois
import os
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
from datetime import datetime

# ── env setup ──────────────────────────────────────────────────────────────────
env_path = Path(__file__).resolve().parent.parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
SERP_API_KEY = os.getenv("SERP_API_KEY")

from serpapi import GoogleSearch


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# TLDs overwhelmingly used in fraud job postings
SUSPICIOUS_TLDS = {
    ".xyz", ".top", ".click", ".tk", ".ml", ".ga", ".cf",
    ".gq", ".pw", ".icu", ".work", ".link", ".online", ".site",
    ".live", ".shop", ".digital", ".services",
}

# Domains whose presence for a company is a positive credibility signal
CREDIBILITY_DOMAINS = {
    "linkedin.com", "crunchbase.com", "bloomberg.com", "reuters.com",
    "forbes.com", "inc.com", "indeed.com", "glassdoor.com",
    "moneycontrol.com", "economictimes.indiatimes.com", "livemint.com",
    "ambitionbox.com", "naukri.com",
}

# Domains that carry opinion only — reduce weight of signals from these
OPINION_DOMAINS = {
    "reddit.com", "quora.com", "twitter.com",
    "facebook.com", "youtube.com", "x.com",
}

# Generic scam-awareness content — does NOT target a specific company
GENERIC_ADVICE_PHRASES = [
    "how to identify", "avoid scams", "how to spot",
    "tips to avoid", "protect yourself", "common scam",
    "beware of scams", "red flags", "what is a scam",
    "employment scam red flags", "job scam red flags",
    "recruitment fraud alert",
    "how to avoid fake", "scam awareness",
]

# Keywords that strongly indicate THIS company is being accused
STRONG_FRAUD_KEYWORDS = [
    "job scam", "fake job", "scam alert", "fraud case",
    "scam warning", "fraudulent company", "reported scam",
    "victim of", "cheated by", "fake company", "ponzi",
    "pyramid scheme", "employment fraud",
]

# Milder negative signals
WEAK_FRAUD_KEYWORDS = [
    "complaint", "negative review", "bad experience",
    "poor service", "unprofessional", "horrible company",
]

# (keywords_list, fraud_bonus, evidence_label)
HARD_JOB_FLAGS = [
    (
        ["pay fee", "registration fee", "processing fee",
         "security deposit", "training fee", "advance fee",
         "joining fee", "membership fee"],
        0.40,
        "Job posting demands payment from applicant",
    ),
    (
        ["wire transfer", "western union", "moneygram",
         "bitcoin", "gift card", "money order", "crypto"],
        0.40,
        "Payment via untraceable method requested",
    ),
    (
        ["gmail.com", "yahoo.com", "hotmail.com",
         "outlook.com", "rediffmail.com", "ymail.com"],
        0.25,
        "Contact address is a free personal email (not a company domain)",
    ),
    (
        ["guaranteed job", "no interview", "immediate hiring",
         "work from home earn", "make money fast",
         "earn from home", "daily payment", "paid training"],
        0.15,
        "Job posting contains unrealistic or suspicious promises",
    ),
    (
        ["apply immediately", "limited slots", "only today",
         "urgent hiring", "act now", "don't miss this"],
        0.10,
        "Job posting uses high-pressure urgency language",
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. WEB RETRIEVER
# ══════════════════════════════════════════════════════════════════════════════

class WebRetriever:
    """Fetches Google search results for a company via SerpAPI."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, company: str) -> list:
        query = f'"{company}" scam OR fraud OR review OR legitimacy'
        params = {
            "engine":  "google",
            "q":       query,
            "api_key": self.api_key,
            "num":     7,
        }
        try:
            raw = GoogleSearch(params).get_dict()
            return self._parse(raw)
        except Exception as exc:
            print(f"[WebRetriever] search error: {exc}")
            return []

    @staticmethod
    def _parse(raw: dict) -> list:
        return [
            {
                "title":   r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "link":    r.get("link", ""),
            }
            for r in raw.get("organic_results", [])
        ]


# ══════════════════════════════════════════════════════════════════════════════
# 2. DOMAIN ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

class DomainAnalyzer:
    """
    Analyses the domain found inside the job posting.

    Returns {"score": float in [-1, +1], "evidence": list[str]}
    Negative score = legitimacy; positive = fraud.
    """

    def analyze(self, job_text: str) -> dict:
        domain = self._extract_domain(job_text)

        if not domain:
            return {
                "score":    0.10,
                "evidence": ["No company website URL included in job posting"],
            }

        score = 0.0
        evidence = []

        # ── TLD check ──────────────────────────────────────────────────────
        tld = self._get_tld(domain)
        if tld in SUSPICIOUS_TLDS:
            score += 0.50
            evidence.append(f"Website uses a high-risk TLD: '{tld}'")
        elif tld in {".com", ".in", ".co.in", ".org", ".net", ".io", ".co"}:
            score -= 0.05
        else:
            score += 0.10

        # ── WHOIS age check ────────────────────────────────────────────────
        try:
            w = whois.whois(domain)
            creation_date = w.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            if creation_date is None:
                raise ValueError("no creation_date")

            age_days = (datetime.now() - creation_date).days

            if age_days < 90:
                score += 0.55
                evidence.append(
                    f"Domain registered very recently ({age_days} days ago) — very high risk"
                )
            elif age_days < 180:
                score += 0.40
                evidence.append(f"Domain registered recently ({age_days} days ago)")
            elif age_days < 365:
                score += 0.15
                evidence.append(f"Domain is less than 1 year old ({age_days} days)")
            else:
                score -= 0.15
                evidence.append(
                    f"Domain is established ({age_days // 365}+ years old)"
                )

        except Exception:
            score += 0.30
            evidence.append("Domain WHOIS is blocked or unavailable")

        return {"score": round(max(-1.0, min(score, 1.0)), 3), "evidence": evidence}

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_domain(text: str):
        m = re.search(r"https?://([^/\s]+)", text)
        return m.group(1).lower() if m else None

    @staticmethod
    def _get_tld(domain: str) -> str:
        parts = domain.rstrip(".").split(".")
        if len(parts) >= 3 and parts[-2] in {"co", "com", "org", "net", "gov", "ac"}:
            return "." + ".".join(parts[-2:])
        return "." + parts[-1] if parts else ""


# ══════════════════════════════════════════════════════════════════════════════
# 3. WEB SCORER
# ══════════════════════════════════════════════════════════════════════════════

class WebScorer:
    """
    Analyses search results for a given company.

    Returns (score, evidence):
      score in [-1, +1]  — negative = legit, positive = fraud
    """

    def score(self, results: list, company_name: str) -> tuple:
        if not results:
            return 0.20, ["No web search results found for this company"]

        name_tokens = self._name_tokens(company_name)
        sc = 0.0
        evidence = []
        strong_hits = 0
        official_hits = 0
        credibility_hits = 0

        for r in results:
            text        = self._text(r)
            link        = r["link"].lower()
            is_official = self._is_official_site(r, company_name)
            is_generic  = self._is_generic_advice(text)
            mentions    = self._mentions_company(text, link, name_tokens)
            credible    = any(cd in link for cd in CREDIBILITY_DOMAINS)
            opinion     = any(od in link for od in OPINION_DOMAINS)

            # ── Official company website ────────────────────────────────
            if is_official:
                official_hits += 1
                sc -= 0.40
                evidence.append("Official company website found in results")

                if any(p in text for p in
                       ["fraud alert", "recruitment fraud", "beware of fake", "impostor"]):
                    sc -= 0.20
                    evidence.append(
                        "Company publishes official anti-fraud warning "
                        "(characteristic of a real organisation)"
                    )
                continue

            # ── Skip generic scam-advice articles not about this company ─
            if is_generic and not mentions:
                continue

            # ── Strong fraud accusation about this company ───────────────
            if mentions and any(kw in text for kw in STRONG_FRAUD_KEYWORDS):
                weight = 0.28 if opinion else 0.45
                sc += weight
                strong_hits += 1
                evidence.append(
                    "Fraud or scam accusation found directly about this company"
                )

            # ── Weak negative signal ────────────────────────────────────
            elif mentions and any(kw in text for kw in WEAK_FRAUD_KEYWORDS):
                sc += 0.08
                evidence.append("Negative review or complaint found about this company")

            # ── Credibility signal ─────────────────────────────────────
            if credible and mentions:
                credibility_hits += 1
                sc -= 0.18
                host = urlparse(r["link"]).netloc.replace("www.", "")
                evidence.append(f"Company has presence on credible platform ({host})")

        # ── Conflicting signals ─────────────────────────────────────────
        if strong_hits > 0 and official_hits > 0:
            sc -= 0.20
            evidence.append(
                "Conflicting signals: official site exists alongside fraud reports"
            )

        # ── No credible presence found for this company ─────────────────
        if official_hits == 0 and credibility_hits == 0:
            sc += 0.22
            evidence.append(
                "No official website or credible platform presence found for this company"
            )

        return round(max(-1.0, min(sc, 1.0)), 3), list(dict.fromkeys(evidence))

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _text(r: dict) -> str:
        return (r["title"] + " " + r["snippet"]).lower()

    @staticmethod
    def _name_tokens(company_name: str) -> list:
        """
        Breaks company name into meaningful tokens.
        'Fake Job Corp' → ['fake job corp', 'fake', 'job']
        This lets 'fake' appear in a result title trigger a match.
        """
        stop = {
            "ltd", "limited", "pvt", "private", "inc", "corp",
            "corporation", "company", "co", "llc", "and", "the", "of",
        }
        tokens = [t.lower() for t in re.split(r"\s+", company_name.strip())]
        meaningful = [t for t in tokens if t not in stop and len(t) > 2]
        return [company_name.lower()] + meaningful

    @staticmethod
    def _mentions_company(text: str, link: str, tokens: list) -> bool:
        return any(tok in text or tok in link for tok in tokens)

    @staticmethod
    def _is_official_site(r: dict, company_name: str) -> bool:
        try:
            domain = urlparse(r["link"]).netloc.lower().replace("www.", "")
            name_slug  = re.sub(r"[^a-z0-9]", "", company_name.lower())
            first_word = company_name.strip().split()[0].lower()
            return name_slug in domain or (len(first_word) > 3 and first_word in domain)
        except Exception:
            return False

    @staticmethod
    def _is_generic_advice(text: str) -> bool:
        return any(phrase in text for phrase in GENERIC_ADVICE_PHRASES)


# ══════════════════════════════════════════════════════════════════════════════
# 4. JOB TEXT ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

class JobTextAnalyzer:
    """
    Checks the raw job posting for hard fraud signals.
    Returns {"bonus": float, "evidence": list[str]}
    bonus is added DIRECTLY to final_score, bypassing the blend.
    """

    def analyze(self, job_text: str) -> dict:
        text        = job_text.lower()
        total_bonus = 0.0
        evidence    = []

        for keywords, weight, label in HARD_JOB_FLAGS:
            if any(kw in text for kw in keywords):
                total_bonus += weight
                evidence.append(label)

        return {
            "bonus":    round(min(total_bonus, 0.80), 2),
            "evidence": evidence,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 5. CONFIDENCE ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_confidence(final_score: float, evidence_count: int) -> float:
    """
    Confidence is jointly determined by:
      1. How extreme the score is (very low or very high → more certain)
      2. How much evidence was collected (more pieces → more certain)

    Without sufficient evidence, confidence stays moderate even if
    the score looks extreme (e.g. signals that cancel each other out).
    """
    ev_factor  = min(evidence_count / 5.0, 1.0)
    extremity  = abs(final_score - 0.5) * 2          # 0 at 0.5, 1 at 0 or 1
    raw = 0.40 + 0.35 * ev_factor + 0.25 * extremity
    return round(min(raw, 0.99), 2)


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN TOOL
# ══════════════════════════════════════════════════════════════════════════════

class CompanyVerificationTool:

    def __init__(self, api_key: str):
        self.web        = WebRetriever(api_key)
        self.web_scorer = WebScorer()
        self.domain     = DomainAnalyzer()
        self.job_text   = JobTextAnalyzer()

    def verify(self, company_name: str, job_text: str = "") -> dict:
        """
        Returns:
          score      : float [0, 1]  — 0 = definitely legit, 1 = definitely fraud
          label      : "legit" | "suspicious" | "fraud"
          confidence : float [0, 1]
          evidence   : list[str]
          sources    : list[dict]  — top web results
        """

        # ── A: Web signals ─────────────────────────────────────────────────
        web_results             = self.web.search(company_name)
        web_score, web_evidence = self.web_scorer.score(web_results, company_name)

        # ── B: Domain signals ──────────────────────────────────────────────
        domain_result   = self.domain.analyze(job_text)
        domain_score    = domain_result["score"]
        domain_evidence = domain_result["evidence"]

        # ── C: Job-text hard flags ─────────────────────────────────────────
        jt          = self.job_text.analyze(job_text)
        jt_bonus    = jt["bonus"]
        jt_evidence = jt["evidence"]

        # ── Blend A + B (both normalised from [-1,+1] to [0,1]) ───────────
        web_norm    = (web_score    + 1.0) / 2.0
        domain_norm = (domain_score + 1.0) / 2.0
        blended     = 0.65 * web_norm + 0.35 * domain_norm

        # Soft-centre: pull the blend toward 0.5 so only hard evidence
        # (job-text bonuses) pushes the score into the extreme ranges.
        blended = 0.5 + (blended - 0.5) * 0.75

        # ── Add job-text bonuses (ground truth, bypass blend) ──────────────
        final_score = round(max(0.0, min(blended + jt_bonus, 1.0)), 2)

        # ── Label ──────────────────────────────────────────────────────────
        if final_score >= 0.60:
            label = "fraud"
        elif final_score >= 0.38:
            label = "suspicious"
        else:
            label = "legit"

        # ── Merge evidence (preserve insertion order, deduplicate) ─────────
        all_evidence = list(dict.fromkeys(
            web_evidence + domain_evidence + jt_evidence
        ))

        # ── Confidence ─────────────────────────────────────────────────────
        confidence = _estimate_confidence(final_score, len(all_evidence))

        return {
            "score":      final_score,
            "label":      label,
            "confidence": confidence,
            "evidence":   all_evidence,
            "sources":    web_results[:3],
        }
