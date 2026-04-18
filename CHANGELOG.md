# Changelog

All notable changes to this project will be documented in this file.

---

## [Unreleased] — 2026-04-18

### Portfolio Manager: Deterministic Signal Weighting Engine

New `SignalWeightingEngine` class computes weighted scores programmatically (valuation 30% / fundamentals 25% / technical 20% / macro 15% / sentiment 10%), so LLM is only used for qualitative overlay and explanation — no longer controls weight allocation. Confidence parsing handles percentages, decimals, and string formats uniformly.

### Debate System: Multi-Round Bull/Bear Debate

Replaced static single-pass bull→bear→debate flow with conditional edges: bull and bear researchers alternate up to 5 rounds (10 total turns) before entering the debate room. `should_continue_debate()` tracks round count and routes accordingly. Debate room refactored to use `get_chat_completion_with_validation` and robust JSON parsing.

### HTTP Client: Anti-Scraping Resilience

`smart_get` / `smart_post` now include per-domain rate limiting (thread-safe), `curl_cffi` Chrome TLS fingerprint impersonation with automatic fallback to `requests`, and exponential backoff retry for SSL/connection errors. Rate limits configured for eastmoney (500ms), 10jqka (300ms), sina (200ms).

### Data Layer: DataSourceManager Integration

`get_northbound_flow`, `get_macro_indicators`, and `get_financial_metrics` in `api.py` now route through `DataSourceManager.fetch_with_fallback()` with SQLite-backed cache (TTL: 4h for price data, 24h for financials/macro). Data sources include akshare, tushare, baostock in fallback order.

### News Crawler: Tier 0 Eastmoney Direct API

Added `_fetch_news_from_eastmoney_direct()` as the highest-priority news source, using the enhanced HTTP client to call Eastmoney's search API directly. This bypasses akshare's slower news function and benefits from anti-scraping rate limiting. News tier order is now: Tier 0 (eastmoney direct) → Tier 1 (eastmoney akshare + sina + 10jqka) → Tier 2 (Playwright) → Tier 3 (akshare fallback).

### Agent State: Deep Merge for Incremental Sub-States

Replaced `merge_dicts` with `merge_dicts_deep` for `AgentState["data"]`, so nested sub-states (e.g. `debate_state`, `technical_report`) are merged incrementally across agents rather than overwritten wholesale by the last writer.

### Agent Reports: Persistent in State

All agents now store their full report in `state["data"]` under dedicated keys (`technical_report`, `fundamentals_report`, `valuation_report`, `sentiment_report`, `risk_report`, `macro_report`, `macro_news_report`), enabling downstream agents to access structured analysis data.

### Workflow Graph: Restructured Edges

`macro_news_agent` now feeds into `researcher_bull` (previously separate path to both researchers and portfolio manager). Debate flow uses conditional edges instead of static edges. `macro_news_agent` no longer directly connects to `portfolio_management_agent`.

### Test Fix: Northbound Flow Cache Bypass

Fixed `test_get_northbound_flow_signal_bullish` by monkeypatching `DataSourceManager.fetch_with_fallback` to skip cache, ensuring the akshare mock is actually invoked instead of returning stale cached data.

### Dependencies: ChromaDB as Optional Extra

Added `chromadb` as an optional dependency under `poetry install -E memory`. Added `src/data/memory/` to `.gitignore`.

---

## [0.3.0] — 2026-04-12

### Full Chinese Localization + News 3-Tier Architecture

All agent outputs, workflow status, analysis reports, and LLM prompts unified to Chinese. News crawling upgraded to three-tier priority system (HTTP APIs → Playwright browsers → akshare fallback) with fuzzy deduplication and incremental caching.

---

## [0.2.0] — 2025-06-22

### News Search Upgrade

Integrated smart search engine for more precise financial news retrieval. Added incremental caching, deduplication, and time-range filtering. Multi-source news from Sina, NetEase, Eastmoney.

---

## [0.1.0] — 2025-04-27

### Macro Analyst Agent

Added macro analyst agent for individual stock macro analysis. Added macro news agent for market-wide news summarization (CSI 300 index).
