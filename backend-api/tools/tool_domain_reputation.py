"""
tools/tool_domain_reputation.py
Domain WHOIS reputation check — age, registrar, expiry, liveness.
100% free — python-whois + requests.

Risk thresholds:
  < 180 days old  → HIGH   (freshly registered, common scam pattern)
  180-730 days    → MEDIUM
  > 730 days      → LOW    (established domain)
"""
import concurrent.futures
import requests
import whois
from datetime import datetime, timezone

from tools.tools_config import REQUEST_HEADERS, REQUEST_TIMEOUT

# Free email / major consumer domains — WHOIS often rate-limits or blocks.
# Their age is always "low risk" (decades old), so skip the expensive lookup.
_KNOWN_FREE_EMAIL_DOMAINS = frozenset({
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "live.com",
    "icloud.com", "me.com", "mac.com", "aol.com", "protonmail.com",
    "proton.me", "zoho.com", "yandex.com", "mail.com", "gmx.com",
    "tutanota.com", "fastmail.com", "rediffmail.com",
})

# Domains that block HTTP probes (Cloudflare 403, etc.) — skip liveness check
_SKIP_LIVENESS = frozenset({
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "live.com",
    "microsoft.com", "google.com", "apple.com",
})

_WHOIS_TIMEOUT = 15   # seconds for the blocking whois call
_HTTP_TIMEOUT  = 8    # seconds for the liveness HTTP check


def _pick(val):
    return val[0] if isinstance(val, list) else val


def _days_since(dt) -> int | None:
    if not dt:
        return None
    try:
        now = datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max((now - dt).days, 0)
    except Exception:
        return None


def _iso(dt) -> str | None:
    return dt.isoformat() if dt else None


def _bare_domain(value: str) -> str:
    """Strip protocol/path, return bare hostname."""
    v = value.strip().lower()
    for p in ("https://", "http://"):
        if v.startswith(p):
            v = v[len(p):]
    v = v.split("/")[0]
    return v[4:] if v.startswith("www.") else v


def _whois_with_timeout(domain: str, timeout: int):
    """Run whois.whois() in a thread so we can enforce a hard timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(whois.whois, domain)
        return future.result(timeout=timeout)


def check_domain_reputation(domain_or_email: str) -> dict:
    """
    WHOIS lookup for a domain, email address, or URL.

    INPUT : domain (infosys.com) OR email (hr@infosys.com) OR URL
    OUTPUT: {ok, data: {domain, registrar, creation_date, domain_age_days,
                        expiration_date, is_live, risk_level}}
    """
    if not domain_or_email:
        return {"ok": False, "error": "domain or email required"}

    raw = domain_or_email.strip()
    if "@" in raw and not raw.startswith("http"):
        domain = raw.split("@")[-1].lower()
    else:
        domain = _bare_domain(raw)

    # Fast-path: well-known free email providers are always old and trusted
    if domain in _KNOWN_FREE_EMAIL_DOMAINS:
        return {
            "ok": True,
            "data": {
                "domain":          domain,
                "registrar":       "Major provider (well-known)",
                "creation_date":   None,
                "expiration_date": None,
                "updated_date":    None,
                "domain_age_days": None,
                "is_live":         True,
                "live_url":        f"https://{domain}",
                "risk_level":      "low",
                "note":            (
                    "Free/major email provider. Using a free email for hiring "
                    "is a weak fraud signal — check other indicators."
                ),
            },
        }

    try:
        w = _whois_with_timeout(domain, _WHOIS_TIMEOUT)
        created = _pick(getattr(w, "creation_date", None))
        expires = _pick(getattr(w, "expiration_date", None))
        updated = _pick(getattr(w, "updated_date", None))
        age     = _days_since(created)

        is_live  = False
        live_url = None
        if domain not in _SKIP_LIVENESS:
            try:
                r = requests.get(
                    f"https://{domain}",
                    headers=REQUEST_HEADERS,
                    timeout=_HTTP_TIMEOUT,
                    allow_redirects=True,
                )
                is_live  = r.status_code < 500
                live_url = r.url
            except Exception:
                pass

        risk = (
            "high"   if age is not None and age < 180  else
            "medium" if age is not None and age < 730  else
            "low"
        )

        return {
            "ok": True,
            "data": {
                "domain":          domain,
                "registrar":       getattr(w, "registrar", None),
                "creation_date":   _iso(created),
                "expiration_date": _iso(expires),
                "updated_date":    _iso(updated),
                "domain_age_days": age,
                "is_live":         is_live,
                "live_url":        live_url,
                "risk_level":      risk,
            },
        }
    except concurrent.futures.TimeoutError:
        return {
            "ok": False,
            "error": f"WHOIS lookup timed out after {_WHOIS_TIMEOUT}s for '{domain}'. "
                     "The domain may be blocking WHOIS queries or the registrar is slow.",
            "data": {"domain": domain},
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "data": {"domain": domain}}
