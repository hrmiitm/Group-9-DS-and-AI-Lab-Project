"""
tools/tool_email_verify.py
Free email verification — syntax check + DNS MX lookup.
No API key. Uses email-validator library.
"""
from email_validator import validate_email, EmailNotValidError

DISPOSABLE_DOMAINS = {
    "mailinator.com", "tempmail.com", "10minutemail.com", "guerrillamail.com",
    "sharklasers.com", "yopmail.com", "trashmail.com", "throwam.com", "getnada.com",
}

ROLE_PREFIXES = {
    "admin", "hr", "jobs", "careers", "info", "support", "hello",
    "contact", "billing", "noreply", "no-reply", "recruitment",
}


def verify_email(email: str) -> dict:
    """
    Two-stage check:
      Stage 1: syntax validation (offline, instant)
      Stage 2: DNS MX lookup   (network, ~1-2s)

    INPUT : email string
    OUTPUT: {ok, data: {email, domain, is_syntax_valid, is_deliverable,
                        mx_host, is_disposable, is_role_account, overall_status}}
    """
    if not email:
        return {"ok": False, "error": "email is required"}

    try:
        info = validate_email(email.strip(), check_deliverability=False)
    except EmailNotValidError as e:
        return {
            "ok": False,
            "error": f"Syntax invalid: {e}",
            "data": {"email": email, "is_syntax_valid": False},
        }

    domain     = info.domain
    local_part = info.local_part

    is_deliverable = False
    mx_host        = None
    deliv_err      = None

    try:
        full = validate_email(email.strip(), check_deliverability=True)
        is_deliverable = True
        if getattr(full, "mx", None):
            mx_host = str(full.mx[0][1])
    except EmailNotValidError as e:
        deliv_err = str(e)

    return {
        "ok": True,
        "data": {
            "email":               info.normalized,
            "local_part":          local_part,
            "domain":              domain,
            "is_syntax_valid":     True,
            "is_deliverable":      is_deliverable,
            "mx_host":             mx_host,
            "is_disposable":       domain in DISPOSABLE_DOMAINS,
            "is_role_account":     local_part.lower() in ROLE_PREFIXES,
            "deliverability_error": deliv_err,
            "overall_status":      (
                "deliverable" if is_deliverable
                else "undeliverable" if deliv_err
                else "unknown"
            ),
        },
    }
