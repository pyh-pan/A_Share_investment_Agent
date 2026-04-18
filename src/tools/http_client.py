"""Smart HTTP client with anti-scraping resilience.

Features:
- Per-domain rate limiting (thread-safe)
- TLS fingerprint impersonation via curl_cffi
- Exponential backoff retry for SSL/connection errors
- Separate smart_get / smart_post with identical resilience logic
"""
from __future__ import annotations

import threading
import time
from typing import Any

import requests

from src.utils.logging_config import setup_logger

logger = setup_logger("http_client")

try:
    from curl_cffi import requests as curl_requests
    USE_CURL_CFFI = True
except Exception:
    curl_requests = None
    USE_CURL_CFFI = False

_domain_last_request: dict[str, float] = {}
_rate_limit_lock = threading.Lock()

RATE_LIMITS: dict[str, float] = {
    "eastmoney.com": 0.5,   # 500ms between requests
    "10jqka.com.cn": 0.3,   # 300ms
    "sina.com.cn": 0.2,     # 200ms
    "default": 0.1,         # 100ms
}


def _enforce_rate_limit(url: str) -> None:
    """Block until the per-domain minimum interval has elapsed."""
    domain = next((d for d in RATE_LIMITS if d != "default" and d in url), "default")
    min_interval = RATE_LIMITS.get(domain, RATE_LIMITS["default"])

    with _rate_limit_lock:
        last_time = _domain_last_request.get(domain, 0.0)
        elapsed = time.time() - last_time
        if elapsed < min_interval:
            wait = min_interval - elapsed
            logger.debug(f"Rate limit: waiting {wait:.3f}s for domain {domain}")
            time.sleep(wait)
        _domain_last_request[domain] = time.time()


_DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
_DEFAULT_HEADERS = {
    "User-Agent": _DEFAULT_UA,
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://www.eastmoney.com/",
}


def _merge_headers(extra: dict[str, str] | None) -> dict[str, str]:
    headers = dict(_DEFAULT_HEADERS)
    if extra:
        headers.update(extra)
    return headers


def smart_get(
    url: str,
    timeout: int = 10,
    max_retries: int = 3,
    **kwargs: Any,
):
    """Smart GET with rate limiting, TLS impersonation, and retry.

    For anti-bot endpoints (Eastmoney, 10jqka), prefer curl_cffi with
    Chrome impersonation.  Falls back to stdlib requests on failure.
    Retries up to *max_retries* times with exponential backoff on
    SSL / connection / timeout errors.
    """
    _enforce_rate_limit(url)
    headers = _merge_headers(kwargs.pop("headers", None))
    is_anti_bot_target = "eastmoney" in url.lower() or "10jqka" in url.lower()

    last_error: Exception | None = None
    for attempt in range(max_retries):
        if USE_CURL_CFFI and is_anti_bot_target and curl_requests is not None:
            try:
                return curl_requests.get(
                    url, timeout=timeout, headers=headers,
                    impersonate="chrome120", **kwargs,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(f"curl_cffi attempt {attempt + 1}/{max_retries} failed for {url}: {exc}")
        else:
            try:
                return requests.get(url, timeout=timeout, headers=headers, **kwargs)
            except (requests.exceptions.SSLError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as exc:
                last_error = exc
                logger.warning(f"requests attempt {attempt + 1}/{max_retries} failed for {url}: {exc}")

        # Exponential backoff: 0.5s, 1s, 2s …
        if attempt < max_retries - 1:
            wait = 0.5 * (2 ** attempt)
            logger.debug(f"Retrying in {wait:.1f}s …")
            time.sleep(wait)

    logger.error(f"All {max_retries} attempts failed for GET {url}: {last_error}")
    raise last_error if last_error else Exception(f"Failed to fetch {url}")


def smart_post(
    url: str,
    data: Any = None,
    json: Any = None,
    timeout: int = 10,
    max_retries: int = 3,
    **kwargs: Any,
):
    """Smart POST with rate limiting, TLS impersonation, and retry.

    Mirrors smart_get semantics for POST requests.
    Eastmoney's search API is one notable consumer that requires POST.
    """
    _enforce_rate_limit(url)
    headers = _merge_headers(kwargs.pop("headers", None))
    headers.setdefault("Content-Type", "application/json")
    is_anti_bot_target = "eastmoney" in url.lower()

    last_error: Exception | None = None
    for attempt in range(max_retries):
        if USE_CURL_CFFI and is_anti_bot_target and curl_requests is not None:
            try:
                return curl_requests.post(
                    url, data=data, json=json, timeout=timeout,
                    headers=headers, impersonate="chrome120", **kwargs,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(f"curl_cffi POST attempt {attempt + 1}/{max_retries} failed for {url}: {exc}")
        else:
            try:
                return requests.post(
                    url, data=data, json=json, timeout=timeout,
                    headers=headers, **kwargs,
                )
            except (requests.exceptions.SSLError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as exc:
                last_error = exc
                logger.warning(f"requests POST attempt {attempt + 1}/{max_retries} failed for {url}: {exc}")

        if attempt < max_retries - 1:
            wait = 0.5 * (2 ** attempt)
            logger.debug(f"Retrying POST in {wait:.1f}s …")
            time.sleep(wait)

    logger.error(f"All {max_retries} attempts failed for POST {url}: {last_error}")
    raise last_error if last_error else Exception(f"Failed POST {url}")
