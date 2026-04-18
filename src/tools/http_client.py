from __future__ import annotations

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


def smart_get(url: str, timeout: int = 10, **kwargs: Any):
    """Smart GET with curl_cffi TLS impersonation fallback.

    For anti-bot endpoints (such as Eastmoney), prefer curl_cffi with
    Chrome impersonation. If unavailable or failed, fallback to requests.
    """
    headers = kwargs.pop("headers", {}) or {}
    headers.setdefault(
        "User-Agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36",
    )
    headers.setdefault("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8")
    headers.setdefault("Referer", "https://www.eastmoney.com/")

    is_anti_bot_target = "eastmoney" in url.lower() or "10jqka" in url.lower()
    if USE_CURL_CFFI and is_anti_bot_target and curl_requests is not None:
        try:
            return curl_requests.get(
                url,
                timeout=timeout,
                headers=headers,
                impersonate="chrome120",
                **kwargs,
            )
        except Exception as exc:
            logger.warning(f"curl_cffi 请求失败，回退 requests: {exc}")

    return requests.get(url, timeout=timeout, headers=headers, **kwargs)
