# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered A-share (Chinese stock market) investment analysis system using multi-agent collaboration and LLM-enhanced decision making. Adapted from [ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) for the Chinese A-share market.

## Commands

```bash
# Install dependencies
poetry install

# Run stock analysis (CLI mode)
poetry run python src/main.py --ticker 301155
poetry run python src/main.py --ticker 301155 --show-reasoning
poetry run python src/main.py --ticker 301155 --show-reasoning --summary --num-of-news 20

# Run with FastAPI backend (API mode, serves on port 8000)
poetry run python run_with_backend.py
poetry run python run_with_backend.py --ticker 301155 --show-reasoning

# Backtesting
poetry run python src/backtester.py --ticker 301157 --start-date 2024-12-11 --end-date 2025-01-07 --num-of-news 20

# Linting / formatting (dev dependencies)
poetry run black .
poetry run isort .
poetry run flake8 .

# Tests
poetry run pytest
```

## Architecture

The system is a **LangGraph StateGraph** workflow defined in `src/main.py`. All agents share an `AgentState` (defined in `src/agents/state.py`) containing `messages`, `data`, and `metadata` dicts that merge across nodes.

### Agent Workflow (execution order)

```
market_data_agent
  ├─→ technical_analyst_agent ──┐
  ├─→ fundamentals_agent ──────┤
  ├─→ sentiment_agent ─────────┤──→ researcher_bull_agent ──┐
  ├─→ valuation_agent ─────────┤──→ researcher_bear_agent ──┤
  │                             │                            │
  └─→ macro_news_agent ─────── │ ──────────────────────┐    │
                                                        │    ▼
                                                        │  debate_room_agent
                                                        │    │
                                                        │  risk_management_agent
                                                        │    │
                                                        │  macro_analyst_agent
                                                        │    │
                                                        └──→ portfolio_management_agent → END
```

- **Parallel fan-out**: `market_data_agent` fans out to 5 analysis agents simultaneously
- **Bull/Bear convergence**: 4 analysts feed into both `researcher_bull` and `researcher_bear`
- **Debate → Decision**: Researchers debate in `debate_room`, then flows through risk → macro → portfolio manager
- **News parallel path**: `macro_news_agent` runs independently and joins at `portfolio_management_agent`

### Key Directories

- `src/agents/` — Agent node functions, each decorated with `@agent_endpoint` from `src/utils/api_utils.py`
- `src/tools/api.py` — Market data fetching via **akshare** (A-share data: prices, financials, real-time quotes)
- `src/tools/openrouter_config.py` — LLM call entry point (`get_chat_completion`), uses `LLMClientFactory`
- `src/utils/llm_clients.py` — `GeminiClient` and `OpenAICompatibleClient` implementations with retry/backoff
- `backend/` — FastAPI backend for API-driven analysis; routers in `backend/routers/`, state in `backend/state.py`

### LLM Configuration

The system supports two LLM backends, auto-selected by `LLMClientFactory` in `src/utils/llm_clients.py`:
- **OpenAI-compatible API** (priority if all 3 env vars set): `OPENAI_COMPATIBLE_API_KEY`, `OPENAI_COMPATIBLE_BASE_URL`, `OPENAI_COMPATIBLE_MODEL`
- **Google Gemini**: `GEMINI_API_KEY`, `GEMINI_MODEL`

Environment variables are loaded from `.env` file (copy `.env.example` to `.env`).

### Agent Pattern

Each agent in `src/agents/` follows this pattern:
1. Decorated with `@agent_endpoint(name, description)` which registers it with the backend API
2. Takes `AgentState` as input, reads from `state["data"]` and `state["metadata"]`
3. Performs analysis (data processing + optional LLM call via `get_chat_completion`)
4. Returns updated state with a `HumanMessage` appended to `messages`, using agent name as `name` field
5. Uses `show_agent_reasoning()` / `show_workflow_status()` from `src/agents/state.py` for output

### Data Sources

All market data comes from **akshare** library in `src/tools/api.py`:
- `stock_zh_a_hist` — historical price data
- `stock_zh_a_spot_em` — real-time quotes
- `stock_financial_analysis_indicator` — financial metrics (Sina Finance)
- `stock_financial_report_sina` — financial statements
- News crawling via `src/tools/news_crawler.py` with intelligent caching in `src/data/`

### News Crawling Tiered Architecture

`src/tools/news_crawler.py` uses a three-tier priority system for news acquisition:
- **Tier 1** (fast HTTP APIs): 东方财富 + 新浪财经 + 同花顺 — always executed first
- **Tier 2** (Playwright browser): 百度资讯 + Bing + Google — only if tier 1 is insufficient
- **Tier 3** (fallback): akshare `stock_news_em` — only if tiers 1+2 both fail

Deduplication uses exact title matching + fuzzy substring matching (normalized titles with punctuation removed).

## Optimization Roadmap

The following improvements are planned, ordered by impact. Check items off as they are completed.

### P0: Core Defects (Severely Impact Accuracy)

- [ ] **Macro analyst data source mismatch** — `macro_analyst.py` fetches *stock-specific* news but tries to analyze the *macro economy*. Should integrate real macro data: GDP, CPI, PMI, M2, LPR, yield curves. akshare provides `macro_china_gdp`, `macro_china_cpi`, `macro_china_pmi`, etc.
- [ ] **Valuation model too crude** — DCF discount rate hardcoded at 10% (should compute WACC dynamically); negative earnings growth floored to 0 (overvalues declining companies); missing comparable company valuation (industry P/E, P/B, EV/EBITDA); missing PEG ratio; enterprise value vs equity value conflated (FCF compared directly to market cap without subtracting net debt).
- [ ] **Fundamentals thresholds are industry-agnostic** — ROE>15%, net margin>20%, debt ratio<50% applied uniformly across all sectors. Banks, tech, brokerages all have different norms. Need industry classification and relative percentile comparison.
- [ ] **Sentiment analysis too narrow** — Only analyzes news articles; missing retail investor sentiment (Eastmoney guba, Xueqiu/Snowball, Weibo); all news collapsed to single -1~1 score losing structural information; no temporal decay weighting.

### P1: Important Gaps (Significantly Affect Quality)

- [ ] **Missing A-share specific data** — Northbound capital flows (`stock_hsgt_north_net_flow_in_em`), margin trading (`stock_margin_detail_szse/sse`), block trades (`stock_dzjy_sctj`), top 10 shareholders (`stock_gdfx_free_holding_analyse_em`), dragon & tiger list (`stock_lhb_detail_em`), restricted share unlocks (`stock_restricted_release_queue_sina`), shareholder count changes (`stock_zh_a_gdhs`).
- [ ] **Technical analysis gaps** — Missing KDJ indicator (most popular in Chinese markets); no MACD divergence detection (only crossover); turnover rate data available but never used; no support/resistance level identification; no MA5/10/20/60 support analysis; first-layer indicators (MACD/RSI/BB/OBV) computed then overwritten by second-layer strategies (dead code).
- [ ] **Risk management too basic** — Only historical VaR, missing CVaR/Expected Shortfall and Monte Carlo; no Beta coefficient calculation; no liquidity risk assessment; stress testing is trivially simple (flat % declines); T+1 settlement rule completely ignored.
- [ ] **Final decision over-relies on LLM** — Portfolio manager's signal weights (valuation 30%/fundamentals 25%/technical 20%/macro 15%/sentiment 10%) exist only in the prompt text with no programmatic enforcement. Should compute weighted scores in code, use LLM only for qualitative adjustment and explanation.

### P2: Architecture Improvements (Robustness)

- [ ] **Data pipeline redundancy** — `api.py` pre-computes momentum/volatility/Hurst, `technicals.py` recomputes all of them (with different Hurst algorithm); sentiment agent and macro analyst use same news with different LLM prompts (double-counting risk).
- [ ] **No multi-period trend analysis** — All agents only look at latest period. ROE dropping from 30% to 16% still scores bullish because 16%>15%. Should compare 3-5 year trends.
- [ ] **No transaction cost modeling** — A-share costs: commission (~0.025%), stamp tax (0.1% sell-side), transfer fee. Not accounted for in any decision.
- [ ] **Single-stock decisions only** — No multi-stock portfolio construction, sector allocation, correlation-based diversification, or rebalancing logic.
- [ ] **Shenzhen stock financial statements bug** — `get_financial_statements()` hardcodes `sh` prefix for Sina API calls; stocks starting with 0 or 3 (Shenzhen-listed) will fail to retrieve financial data. `_normalize_ak_symbol()` exists but is not used for these calls.
- [ ] **`format_decision()` weight mismatch** — Displays 30/35/25 weights for fundamental/valuation/technical, contradicting the 25/30/20 in the system prompt.

## Known Issues

- Eastmoney API intermittently returns `RemoteDisconnected` — system falls back to Sina/Tencent automatically
- Bing Playwright search returns 0 results in current environment — graceful degradation, does not block workflow
- `OutputLogger` stdout redirect can conflict with `@agent_endpoint` decorator's StringIO capture during parallel agent execution — some print statements may be lost
