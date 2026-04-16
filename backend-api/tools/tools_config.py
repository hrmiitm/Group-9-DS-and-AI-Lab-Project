"""
tools/tools_config.py
Shared configuration for all investigative tools.
No API keys — all tools are free.
"""

REQUEST_TIMEOUT = 20       # seconds per HTTP request
DEFAULT_PHONE_REGION = "IN"  # 2-letter country hint for phonenumbers library

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0 Safari/537.36"
    )
}
