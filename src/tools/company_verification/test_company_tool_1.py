"""
test_company_tool.py
────────────────────
Comprehensive test suite for CompanyVerificationTool.

Tests are grouped into 5 categories:
  1. Clearly Legit     — well-known real companies
  2. Clearly Fraudulent — obvious scam postings
  3. Suspicious / Edge  — borderline cases that should not be legit
  4. Borderline Legit   — small/unknown companies that are real
  5. Stress / Adversarial — tricky inputs designed to fool the scorer

Each test case defines:
  company    : str
  job_text   : str   — the raw job posting text
  expected   : str   — "legit" | "suspicious" | "fraud"
  note       : str   — why this case is interesting / what it tests

Run:
  python test_company_tool.py
  python test_company_tool.py --live      # actually calls SerpAPI + WHOIS
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Literal

# ── parse args ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--live", action="store_true",
    help="Run against real SerpAPI + WHOIS (uses API credits)"
)
parser.add_argument(
    "--category", type=str, default=None,
    help="Run only tests in a specific category (e.g. 'Clearly Legit')"
)
parser.add_argument(
    "--verbose", "-v", action="store_true",
    help="Print full result dict for each test"
)
args = parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# TEST CASE DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestCase:
    company:  str
    job_text: str
    expected: Literal["legit", "suspicious", "fraud"]
    note:     str
    category: str = ""


TEST_CASES: list[TestCase] = [

    # ══════════════════════════════════════════════════════════════════════════
    # 1. CLEARLY LEGIT — well-known companies with strong web presence
    # ══════════════════════════════════════════════════════════════════════════

    TestCase(
        company  = "Infosys",
        job_text = "Apply at https://www.infosys.com/careers and submit your resume.",
        expected = "legit",
        note     = "Large Indian IT company; official domain, established WHOIS, LinkedIn/Glassdoor presence.",
        category = "Clearly Legit",
    ),
    TestCase(
        company  = "Tata Consultancy Services",
        job_text = "Visit https://www.tcs.com/careers to apply. Contact: recruiter@tcs.com",
        expected = "legit",
        note     = "TCS — largest Indian IT employer; official domain, company email.",
        category = "Clearly Legit",
    ),
    TestCase(
        company  = "Google",
        job_text = "Apply via https://careers.google.com. Role: Software Engineer, Bangalore.",
        expected = "legit",
        note     = "Global tech giant; massive credible web presence, official domain.",
        category = "Clearly Legit",
    ),
    TestCase(
        company  = "Wipro",
        job_text = "Send your CV to careers@wipro.com or visit https://www.wipro.com/careers/",
        expected = "legit",
        note     = "Established IT firm; official email domain, credible web presence.",
        category = "Clearly Legit",
    ),
    TestCase(
        company  = "Deloitte",
        job_text = "Apply at https://www2.deloitte.com/in/en/careers.html for the Analyst role.",
        expected = "legit",
        note     = "Big Four firm; widely indexed, official domain, LinkedIn presence.",
        category = "Clearly Legit",
    ),
    TestCase(
        company  = "Amazon",
        job_text = "Role: Operations Manager. Apply at https://www.amazon.jobs/",
        expected = "legit",
        note     = "Global company; Amazon has a dedicated jobs subdomain.",
        category = "Clearly Legit",
    ),
    TestCase(
        company  = "HDFC Bank",
        job_text = "Openings for Relationship Manager. Visit https://www.hdfcbank.com/personal/careers",
        expected = "legit",
        note     = "Major Indian bank; regulated, official domain.",
        category = "Clearly Legit",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # 2. CLEARLY FRAUDULENT — obvious scam job postings
    # ══════════════════════════════════════════════════════════════════════════

    TestCase(
        company  = "Fake Job Corp",
        job_text = "Send money to hr.fakejob@gmail.com",
        expected = "fraud",
        note     = "Personal Gmail contact + unknown company = clear scam.",
        category = "Clearly Fraudulent",
    ),
    TestCase(
        company  = "Global Earn Fast Solutions",
        job_text = (
            "Work from home and earn ₹50,000/month! No experience needed. "
            "Guaranteed job. Pay registration fee of ₹999 at globalfastjobs.xyz "
            "Contact: hire@globalfastjobs.xyz"
        ),
        expected = "fraud",
        note     = "Suspicious TLD + registration fee + Gmail + unrealistic promises = textbook fraud.",
        category = "Clearly Fraudulent",
    ),
    TestCase(
        company  = "QuickHire India Pvt Ltd",
        job_text = (
            "Immediate hiring! No interview required. Earn ₹25,000 weekly. "
            "Pay a ₹1500 processing fee via Western Union. "
            "Apply: quickhireindia.top/apply"
        ),
        expected = "fraud",
        note     = "Suspicious TLD + Western Union payment + no-interview promise = strong fraud.",
        category = "Clearly Fraudulent",
    ),
    TestCase(
        company  = "Dream Career Consultancy",
        job_text = (
            "Limited slots available — act now! Pay ₹2000 security deposit "
            "to confirm your offer. Send to hr.dreamcareer@yahoo.com. "
            "Site: http://dreamcareerjobs.online"
        ),
        expected = "fraud",
        note     = "Free email + suspicious TLD + payment request + urgency language.",
        category = "Clearly Fraudulent",
    ),
    TestCase(
        company  = "EasyMoney Staffing Solutions",
        job_text = (
            "Make money fast from home! Send your Aadhaar and bank details "
            "to easymoney.staff@hotmail.com. Pay ₹500 membership fee via gift card. "
            "http://easymoneystaffing.click"
        ),
        expected = "fraud",
        note     = "Gift card payment + personal email + suspicious TLD + PII request.",
        category = "Clearly Fraudulent",
    ),
    TestCase(
        company  = "Royal Jobs Network",
        job_text = (
            "We are hiring! Registration fee of ₹1200 required. "
            "Contact royaljobsnetwork@gmail.com. "
            "Guaranteed placement in MNC companies."
        ),
        expected = "fraud",
        note     = "No URL, Gmail contact, registration fee, guaranteed placement promise.",
        category = "Clearly Fraudulent",
    ),
    TestCase(
        company  = "NextGen Opportunity Hub",
        job_text = (
            "Apply via http://nextgen-opphub-hiring.tk — pay advance fee of ₹800. "
            "No experience required. Daily payment assured."
        ),
        expected = "fraud",
        note     = ".tk domain (free, high-fraud TLD) + advance fee + unrealistic daily payment.",
        category = "Clearly Fraudulent",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # 3. SUSPICIOUS / EDGE CASES — should not pass as legit
    # ══════════════════════════════════════════════════════════════════════════

    TestCase(
        company  = "XYZ Hiring Ltd",
        job_text = "Apply now at http://xyz-careers-123.com and pay registration fee.",
        expected = "fraud",
        note     = "Unknown company + payment request + WHOIS-unavailable domain.",
        category = "Suspicious / Edge",
    ),
    TestCase(
        company  = "Nexus Talent Group",
        job_text = (
            "Data Entry Work from Home. Contact us at nexusjobs123@gmail.com. "
            "Visit http://nexustalentgroup.online"
        ),
        expected = "suspicious",
        note     = "Suspicious TLD + Gmail + WFH data-entry (common scam category). "
                   "No payment request so not certain fraud.",
        category = "Suspicious / Edge",
    ),
    TestCase(
        company  = "Bright Future Careers",
        job_text = (
            "Call centre opening. No fee charged. "
            "Apply at brightfuturecareers@rediffmail.com. "
            "Office: Sector 18, Noida."
        ),
        expected = "suspicious",
        note     = "Free personal email (Rediffmail) + unknown company, but no payment request.",
        category = "Suspicious / Edge",
    ),
    TestCase(
        company  = "PrimeWork Solutions",
        job_text = (
            "Part-time opening. Earn ₹15,000/month. "
            "Freshers welcome. Apply: primeworksolutions@yahoo.com"
        ),
        expected = "suspicious",
        note     = "Yahoo email + unknown company + unrealistic freshers salary.",
        category = "Suspicious / Edge",
    ),
    TestCase(
        company  = "SwiftRecruit Agency",
        job_text = (
            "Urgent hiring for US-based client. Send CV to swiftrecruit.agency@gmail.com. "
            "Apply at http://swiftrecruitagency.work"
        ),
        expected = "suspicious",
        note     = "Suspicious .work TLD + Gmail contact. Agency legitimacy unclear.",
        category = "Suspicious / Edge",
    ),
    TestCase(
        company  = "AlphaStaff India",
        job_text = (
            "Back-office opening. Visit https://alphastaff-india.com. "
            "No fee. Interview required. Send CV to hr@alphastaff-india.com"
        ),
        expected = "suspicious",
        note     = "Company email (positive) but completely unknown company with no web presence.",
        category = "Suspicious / Edge",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # 4. BORDERLINE LEGIT — real but small / less-known companies
    # ══════════════════════════════════════════════════════════════════════════

    TestCase(
        company  = "Zoho Corporation",
        job_text = "Apply at https://careers.zoho.com for a Software Developer role.",
        expected = "legit",
        note     = "Real Indian SaaS company; dedicated careers subdomain, LinkedIn, Glassdoor.",
        category = "Borderline Legit",
    ),
    TestCase(
        company  = "Freshworks",
        job_text = "Visit https://www.freshworks.com/company/careers/ to apply.",
        expected = "legit",
        note     = "Listed Indian SaaS company; well-indexed, official domain.",
        category = "Borderline Legit",
    ),
    TestCase(
        company  = "Urban Company",
        job_text = "Join our team. Apply at https://www.urbancompany.com/careers",
        expected = "legit",
        note     = "Real Indian startup with credible web presence and official domain.",
        category = "Borderline Legit",
    ),
    TestCase(
        company  = "Razorpay",
        job_text = "Fintech role open. Apply at https://razorpay.com/jobs/",
        expected = "legit",
        note     = "Indian fintech unicorn; well-known, Glassdoor + LinkedIn presence.",
        category = "Borderline Legit",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # 5. STRESS / ADVERSARIAL — designed to expose scorer weaknesses
    # ══════════════════════════════════════════════════════════════════════════

    TestCase(
        company  = "Infosys Recruitment Team",
        job_text = (
            "Official Infosys hiring. Pay ₹500 background verification fee. "
            "Contact: infosys.hr.recruit@gmail.com"
        ),
        expected = "fraud",
        note     = "ADVERSARIAL: impersonates Infosys but uses Gmail + payment fee. "
                   "Scorer must not give credit for 'Infosys' in company name alone.",
        category = "Stress / Adversarial",
    ),
    TestCase(
        company  = "TCS HR Department",
        job_text = (
            "Selected for TCS BPS role. Attend orientation. "
            "Pay ₹1500 processing fee via Western Union to confirm. "
            "Contact: tcs.hr.department@gmail.com"
        ),
        expected = "fraud",
        note     = "ADVERSARIAL: TCS impersonation + Western Union + processing fee.",
        category = "Stress / Adversarial",
    ),
    TestCase(
        company  = "Google India Hiring",
        job_text = (
            "Congratulations! You have been selected for Google India. "
            "Pay ₹2000 registration fee to careers@google-india-official.xyz. "
            "Apply: http://google-india-hiring.xyz"
        ),
        expected = "fraud",
        note     = "ADVERSARIAL: Google name + .xyz domain + registration fee. "
                   "Suspicious TLD and payment override the brand name.",
        category = "Stress / Adversarial",
    ),
    TestCase(
        company  = "Deloitte Consulting Pvt Ltd",
        job_text = (
            "Analyst role at Deloitte. "
            "Apply at https://www2.deloitte.com/in/en/careers.html. "
            "No fee. Contact: talent@deloitte.com"
        ),
        expected = "legit",
        note     = "ADVERSARIAL: real Deloitte URL + company email, despite slightly long company name.",
        category = "Stress / Adversarial",
    ),
    TestCase(
        company  = "Amazon Delivery Partner",
        job_text = (
            "Amazon delivery partner hiring. No office visit needed. "
            "Send ₹800 activation fee via Paytm to confirm your kit. "
            "Contact: amazon.delivery.partner@gmail.com"
        ),
        expected = "fraud",
        note     = "ADVERSARIAL: Amazon brand + activation fee + Gmail = scam impersonation.",
        category = "Stress / Adversarial",
    ),
    TestCase(
        company  = "Microsoft India",
        job_text = (
            "Microsoft India is hiring! Visit https://careers.microsoft.com/en/in to apply. "
            "No fee. Interview process: HR round, technical round, offer."
        ),
        expected = "legit",
        note     = "Real Microsoft India careers URL. Should score legit despite large brand.",
        category = "Stress / Adversarial",
    ),
    TestCase(
        company  = "Scam Buster Technologies",
        job_text = (
            "We help companies fight job fraud. "
            "Apply at https://www.scambuster.io/careers. "
            "Contact: hr@scambuster.io"
        ),
        expected = "suspicious",
        note     = "ADVERSARIAL: company name contains 'scam' but is described as legit. "
                   "Web results will be polluted by generic scam content. "
                   "Should be suspicious (unknown company) not fraud (no hard signals).",
        category = "Stress / Adversarial",
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# MOCK RUNNER (no API calls)
# ══════════════════════════════════════════════════════════════════════════════

def run_mock(tc: TestCase) -> dict:
    """
    Simulates the tool using only the job_text signals (domain + job-text flags).
    The web_score is set to 0 (neutral) since we cannot call SerpAPI.
    This validates the domain + job-text layers in isolation.
    """
    import re
    from urllib.parse import urlparse

    SUSPICIOUS_TLDS = {
        ".xyz", ".top", ".click", ".tk", ".ml", ".ga", ".cf",
        ".gq", ".pw", ".icu", ".work", ".link", ".online", ".site",
        ".live", ".shop", ".digital", ".services",
    }
    HARD_JOB_FLAGS = [
        (["pay fee","registration fee","processing fee","security deposit",
          "training fee","advance fee","joining fee","membership fee",
          "activation fee"], 0.40, "Payment demanded from applicant"),
        (["wire transfer","western union","moneygram","bitcoin",
          "gift card","money order","crypto","paytm","upi transfer"],
         0.40, "Untraceable payment method"),
        (["gmail.com","yahoo.com","hotmail.com","outlook.com",
          "rediffmail.com","ymail.com"],
         0.25, "Free personal email contact"),
        (["guaranteed job","no interview","immediate hiring",
          "work from home earn","make money fast","earn from home",
          "daily payment","paid training","guaranteed placement"],
         0.15, "Unrealistic promises"),
        (["apply immediately","limited slots","only today",
          "urgent hiring","act now","don't miss"],
         0.10, "Urgency pressure language"),
    ]

    def extract_domain(text):
        m = re.search(r"https?://([^/\s]+)", text)
        return m.group(1).lower() if m else None

    def get_tld(domain):
        parts = domain.rstrip(".").split(".")
        if len(parts) >= 3 and parts[-2] in {"co","com","org","net","gov","ac"}:
            return "." + ".".join(parts[-2:])
        return "." + parts[-1] if parts else ""

    text   = tc.job_text.lower()
    domain = extract_domain(tc.job_text)

    # Domain score
    d_score = 0.0
    d_ev    = []
    if domain:
        tld = get_tld(domain)
        if tld in SUSPICIOUS_TLDS:
            d_score += 0.50
            d_ev.append(f"Suspicious TLD: {tld}")
        elif tld in {".com",".in",".co.in",".org",".net",".io",".co"}:
            d_score -= 0.05
        # Skip WHOIS in mock
    else:
        d_score = 0.10
        d_ev.append("No URL provided")

    # Job-text bonus
    jt_bonus = 0.0
    jt_ev    = []
    for kws, w, label in HARD_JOB_FLAGS:
        if any(kw in text for kw in kws):
            jt_bonus += w
            jt_ev.append(label)
    jt_bonus = min(jt_bonus, 0.80)

    # Blend with neutral web_score=0 → web_norm=0.5
    d_norm  = (d_score + 1.0) / 2.0
    blended = 0.65 * 0.5 + 0.35 * d_norm
    blended = 0.5 + (blended - 0.5) * 0.75
    final   = round(max(0.0, min(blended + jt_bonus, 1.0)), 2)

    label = "fraud" if final >= 0.60 else "suspicious" if final >= 0.38 else "legit"
    ev    = list(dict.fromkeys(d_ev + jt_ev))

    return {"score": final, "label": label, "confidence": None, "evidence": ev}


# ══════════════════════════════════════════════════════════════════════════════
# LIVE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_live(tc: TestCase, tool) -> dict:
    return tool.verify(company_name=tc.company, job_text=tc.job_text)


# ══════════════════════════════════════════════════════════════════════════════
# RESULT PRINTER
# ══════════════════════════════════════════════════════════════════════════════

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
WARN = "\033[93m~ WARN\033[0m"   # adjacent label (legit↔suspicious or suspicious↔fraud)

LABEL_ORDER = ["legit", "suspicious", "fraud"]

def is_adjacent(got: str, expected: str) -> bool:
    gi = LABEL_ORDER.index(got)
    ei = LABEL_ORDER.index(expected)
    return abs(gi - ei) == 1

def print_result(tc: TestCase, result: dict, idx: int):
    got      = result["label"]
    passed   = got == tc.expected
    adjacent = not passed and is_adjacent(got, tc.expected)
    status   = PASS if passed else (WARN if adjacent else FAIL)

    score = result["score"]
    conf  = result.get("confidence")
    conf_str = f"  conf={conf}" if conf is not None else ""

    print(f"\n  [{idx:02d}] {status}  {tc.company!r}")
    print(f"        expected={tc.expected:<11} got={got:<11} score={score}{conf_str}")
    print(f"        note: {tc.note}")
    if result["evidence"]:
        for e in result["evidence"]:
            print(f"          · {e}")
    if args.verbose:
        print(f"        full: {result}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    tool = None
    if args.live:
        try:
            from company_verification_tool import CompanyVerificationTool, SERP_API_KEY
            if not SERP_API_KEY:
                print("[ERROR] SERP_API_KEY not set. Run without --live for mock mode.")
                sys.exit(1)
            tool = CompanyVerificationTool(api_key=SERP_API_KEY)
            print("[INFO] Running in LIVE mode (SerpAPI + WHOIS)")
        except ImportError as e:
            print(f"[ERROR] Could not import tool: {e}")
            sys.exit(1)
    else:
        print("[INFO] Running in MOCK mode (domain + job-text only, no API calls)")
        print("[INFO] Use --live to run full pipeline with SerpAPI\n")

    # ── filter by category ─────────────────────────────────────────────────
    cases = TEST_CASES
    if args.category:
        cases = [tc for tc in TEST_CASES if tc.category == args.category]
        if not cases:
            cats = sorted({tc.category for tc in TEST_CASES})
            print(f"[ERROR] Unknown category '{args.category}'. Valid: {cats}")
            sys.exit(1)

    # ── group by category ──────────────────────────────────────────────────
    categories = {}
    for tc in cases:
        categories.setdefault(tc.category, []).append(tc)

    total = passed = failed = warned = 0
    results_by_cat = {}

    for cat, tcs in categories.items():
        print(f"\n{'═'*60}")
        print(f"  {cat}  ({len(tcs)} tests)")
        print(f"{'═'*60}")
        cat_pass = cat_fail = cat_warn = 0

        for i, tc in enumerate(tcs, 1):
            if args.live:
                result = run_live(tc, tool)
                time.sleep(1.5)          # avoid rate limiting
            else:
                result = run_mock(tc)

            got      = result["label"]
            _passed  = got == tc.expected
            _adj     = not _passed and is_adjacent(got, tc.expected)

            if _passed:
                passed += 1; cat_pass += 1
            elif _adj:
                warned += 1; cat_warn += 1
            else:
                failed += 1; cat_fail += 1
            total += 1

            print_result(tc, result, i)

        results_by_cat[cat] = (cat_pass, cat_warn, cat_fail)
        print(f"\n  Category result: {cat_pass} pass, {cat_warn} adjacent, {cat_fail} fail")

    # ── summary ────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  FINAL SUMMARY  ({total} tests)")
    print(f"{'═'*60}")
    for cat, (cp, cw, cf) in results_by_cat.items():
        bar = "✓" * cp + "~" * cw + "✗" * cf
        print(f"  {cat:<28} {bar}")

    print()
    print(f"  Total passed   : {passed}/{total}")
    print(f"  Adjacent (~1 label off): {warned}/{total}")
    print(f"  Hard failures  : {failed}/{total}")
    pct = round(100 * passed / total) if total else 0
    print(f"  Pass rate      : {pct}%")

    if failed == 0:
        print("\n  \033[92mAll tests passed (or within 1 label).\033[0m")
    else:
        print(f"\n  \033[91m{failed} test(s) failed.\033[0m")
    print()


if __name__ == "__main__":
    main()
