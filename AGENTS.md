# AGENTS.md

This file provides guidance when working with code in this repository.

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

## Documentation Sync Rule

After every code change, check whether project documentation must be updated before considering the task complete. At minimum, review `AGENTS.md`, `README.md`, and `CHANGELOG.md` against the actual code diff and update roadmap statuses, completed items, remaining gaps, commands, architecture notes, and verification evidence so documentation stays current with implementation.

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

`src/tools/news_crawler.py` uses a four-tier priority system for news acquisition:
- **Tier 0** (direct API via enhanced HTTP client): 东方财富直连搜索API — highest priority, benefits from anti-scraping rate limiting
- **Tier 1** (fast HTTP APIs): 东方财富(akshare) + 新浪财经 + 同花顺 — always executed after tier 0
- **Tier 2** (Playwright browser): 百度资讯 + Bing + Google — only if tiers 0+1 are insufficient
- **Tier 3** (fallback): akshare `stock_news_em` — only if tiers 0+1+2 all fail

Deduplication uses exact title matching + fuzzy substring matching (normalized titles with punctuation removed).

## Optimization Roadmap

The following improvements are planned, ordered by impact and importance within each priority level. Check items off as they are completed.

### P0: Core Defects (Severely Impact Accuracy)

*Critical issues that produce fundamentally wrong analysis results. Fix immediately.*

- [x] **Macro analyst data source mismatch** — Fixed for the current workflow: `macro_analyst.py` now uses `get_macro_indicators()` for GDP/CPI/PMI/M2/LPR-style inputs, `get_industry_news()` for industry context, and deterministic `score_macro_environment()` evidence as both LLM prompt context and fallback output. Tests cover deterministic scoring and non-stock-specific macro fallback behavior. Remaining future refinement: yield-curve/liquidity validation and broader macro data quality checks.
  *Completed: 2026-04-26; Future refinement: 4-8h*

- [x] **Valuation model too crude** — Improved in `src/agents/valuation.py`: dynamic WACC is used for DCF/owner-earnings, growth is bounded instead of flooring all negative growth to zero, PEG diagnostics are included, industry-threshold PE/PB/PS comparable valuation is averaged when available, and net debt adjusts enterprise value to equity value when cash/debt fields are present. Remaining future refinement: true peer-set EV/EBITDA/comparable-company data instead of static industry threshold proxies.
  *Completed: 2026-04-26; Future refinement: 8-12h*

- [x] **Fundamentals thresholds are industry-agnostic** — Improved in `src/agents/fundamentals.py`: market data carries industry classification, static industry threshold profiles are applied, optional peer percentile comparison is available through `calculate_peer_percentile_signal()`, and `analyze_metric_trend()` penalizes multi-period deterioration even when absolute metrics remain above thresholds. Remaining future refinement: populate real peer universes automatically instead of relying on optional peer metric inputs.
  *Completed: 2026-04-26; Future refinement: 6-10h*

- [ ] **Sentiment analysis too narrow** — Partially completed: sentiment analysis now includes temporal decay weighting, Eastmoney guba/forum sentiment, and margin-trading leverage sentiment from `get_margin_trading_sentiment()`. Remaining work: broader platforms such as Xueqiu/Snowball and Weibo, plus richer structural decomposition of news/forum/margin sentiment instead of a single aggregate score.
  *Priority: CRITICAL, Effort: 14-24h remaining*

- [x] **Missing KDJ indicator** — Implemented in `src/agents/technicals.py` via `calculate_kdj()`. KDJ (9,3,3) signals: bullish when J<0 and K<20 (oversold), bearish when J>100 and K>80 (overbought).
  *Completed: 2026-04-18*

- [x] **Missing northbound capital flow data** — Implemented in `src/tools/api.py` via `get_northbound_flow()` with `DataSourceManager` cache (4h TTL). Uses `stock_hsgt_north_net_flow_in_em` with multi-source fallback.
  *Completed: 2026-04-18*

- [x] **Missing MACD divergence detection** — Implemented in `src/agents/technicals.py` via `detect_macd_divergence()`. Detects top divergence (bearish) and bottom divergence (bullish) over 60-day window.
  *Completed: 2026-04-18*

- [x] **No forced tool calling / fallback mechanism for final decision** — Implemented for the final portfolio decision in `src/agents/portfolio_manager.py`: if the LLM returns `None`, invalid JSON, or a payload missing required decision fields, `build_forced_decision()` falls back to the deterministic `SignalWeightingEngine` base decision. Remaining future refinement: apply the same tool-enforcement pattern to every upstream data-fetching LLM agent.
  *Completed: 2026-04-26; Future refinement: 8-12h*
  ```python
  def agent_with_forced_tools(state, tools, prompt, llm):
      response = llm.generate(prompt, tools=tools)
      if not response.tool_calls:  # LLM refused to call tools
          # Force tool invocation in code
          raw_data = tools[0].func(state["symbol"])
          forced_prompt = f"{prompt}\n\n🔴 MANDATORY: You must analyze this real data: {json.dumps(raw_data)}"
          response = llm.generate(forced_prompt)  # Force analysis of real data
      return response
  ```

### P1: Important Gaps (Significantly Affect Quality)

*Important features that significantly limit system capabilities.*

- [ ] **Missing A-share specific data** — Northbound capital flows (`stock_hsgt_north_net_flow_in_em`) and margin trading sentiment (`stock_margin_detail_szse/sse`) are implemented. Remaining gaps: block trades (`stock_dzjy_sctj`), top 10 shareholders (`stock_gdfx_free_holding_analyse_em`), dragon & tiger list (`stock_lhb_detail_em`), restricted share unlocks (`stock_restricted_release_queue_sina`), shareholder count changes (`stock_zh_a_gdhs`).
  *Priority: HIGH, Effort: 8-14h remaining*

- [x] **Risk management too basic** — Enhanced in `src/agents/risk_manager.py`: adds CVaR/Expected Shortfall, beta-aware market risk, liquidity risk, T+1 settlement constraints, and beta/volatility-aware stress scenarios while preserving the existing VaR/volatility/max-drawdown output. Remaining future refinement: portfolio-level correlation and multi-role risk debate.
  *Completed: 2026-04-26; Future refinement: 12-20h*

- [x] **Final decision over-relies on LLM** — ~~Portfolio manager's signal weights exist only in prompt text with no programmatic enforcement.~~ `SignalWeightingEngine` now computes weighted scores deterministically in code (valuation 30%/fundamentals 25%/technical 20%/macro 15%/sentiment 10%). The LLM still handles qualitative adjustment and explanation, but invalid or missing LLM output falls back to `build_forced_decision()` and the final payload carries `base_decision`/`base_confidence`/`weighted_score` for auditability.
  *Completed: 2026-04-26*

- [ ] **Technical analysis gaps** — ~~Missing KDJ indicator (most popular in Chinese markets); no MACD divergence detection (only crossover);~~ turnover rate data available but never used; no support/resistance level identification; no MA5/10/20/60 support analysis; first-layer indicators (MACD/RSI/BB/OBV) computed then overwritten by second-layer strategies (dead code).
  *Note: KDJ and MACD divergence now implemented in P0*
  *Priority: MEDIUM, Effort: 10-14h*

- [ ] **No anti-scraping protection** — ~~Eastmoney API intermittently returns `RemoteDisconnected` because current requests lack browser TLS fingerprint simulation.~~ `curl_cffi` with Chrome impersonation now implemented in `http_client.py` with per-domain rate limiting and exponential backoff. Remaining gap: still seeing `RemoteDisconnected` on some eastmoney endpoints; `curl_cffi` impersonation may need version updates or fallback tuning.
  *From TradingAgents-CN, Priority: MEDIUM (downgraded from HIGH), Effort: 2-4h (remaining)*

- [ ] **No persistent caching** — ~~System refetches all K-line and financial data on every run, wasting API quota and time.~~ Restored in Task 1: `src/tools/cache/db_cache.py` provides `SimpleCache`, and `DataSourceManager` accepts cache injection while using SQLite-backed cache for key API functions (`get_northbound_flow`, `get_macro_indicators`, `get_financial_metrics`, `get_margin_trading_sentiment`, and related DSM call sites). `prewarm_symbol_cache()` now best-effort warms key symbol data. Remaining gap: not all data functions use DSM yet; no scheduled sync cron job.
  *From TradingAgents-CN, Priority: MEDIUM (downgraded from HIGH), Effort: 4-8h remaining*

- [x] **Valuation missing dynamic WACC** — Implemented in `src/tools/api.py` via `calculate_wacc()`, using `calculate_beta()` plus a conservative cost-of-equity/debt blend and 8%-20% bounds. `valuation.py` passes this dynamic WACC into DCF and owner-earnings valuation. Remaining valuation gaps are tracked under the P0 valuation item: PEG, comparables, and net-debt/EV adjustments.
  *Completed: 2026-04-26*

- [x] **Missing margin trading data** — Implemented in `src/tools/api.py` via `get_margin_trading_sentiment()`, using SZSE or SSE margin-detail endpoints based on symbol prefix and returning margin balance change, margin/short ratio, sentiment, numeric signal, and availability flag. `src/agents/sentiment.py` blends the margin signal into final sentiment when available.
  *Completed: 2026-04-26*
  ```python
  def get_margin_trading_sentiment(symbol):
      # Shenzhen or Shanghai market
      if symbol.startswith(('0', '3')):
          df = ak.stock_margin_detail_szse(symbol=symbol)
      else:
          df = ak.stock_margin_detail_sse(symbol=symbol)
      margin_change = df.iloc[0]['融资余额'] - df.iloc[1]['融资余额']
      ms_ratio = df.iloc[0]['融资余额'] / df.iloc[0]['融券余额'] if df.iloc[0]['融券余额'] > 0 else 9999.0
      # Bullish if margin increasing significantly
      if margin_change > 0 and ms_ratio > 10:
          sentiment = "strong_bullish"
      elif margin_change > 0:
          sentiment = "bullish"
      elif margin_change < -0.05 * df.iloc[0]['融资余额']:
          sentiment = "bearish"
      else:
          sentiment = "neutral"
      return {"margin_change": margin_change, "margin_short_ratio": ms_ratio, "sentiment": sentiment}
  ```

- [ ] **No cross-library fallback** — Current `_fetch_with_fallback` only switches between AKShare internal functions. When AKShare completely fails (not just one function), system has no backup. Should add Tushare (commercial) and BaoStock (free) as cross-library fallbacks.
  *From TradingAgents-CN, Priority: MEDIUM, Effort: 8-12h*

### P2: Architecture Improvements (Robustness)

*Improvements that enhance system robustness, maintainability, and scalability.*

- [x] **Shenzhen stock financial statements bug** — Fixed in `get_financial_statements()`: balance sheet, income statement, and cash-flow calls use `_normalize_ak_symbol()` so Shenzhen symbols (`0`/`3`) are sent as `sz...` and Shanghai symbols as `sh...`.
  *Completed: 2026-04-26*
  ```python
  # Fix: Use _normalize_ak_symbol instead of hardcoded prefix
  def get_financial_statements(symbol):
      normalized = _normalize_ak_symbol(symbol)  # Returns sh601xxx or sz000xxx
      return ak.stock_financial_report_sina(stock=normalized, symbol="资产负债表")
  ```

- [x] **`format_decision()` weight mismatch** — Fixed and covered by `tests/unit/test_portfolio_costs_p1.py`: `format_decision()` derives displayed labels from `SIGNAL_WEIGHTS` (valuation 30% / fundamentals 25% / technical 20% / macro 15% / sentiment 10%) instead of stale hardcoded text.
  *Completed: 2026-04-26*

- [ ] **Data pipeline redundancy** — `api.py` pre-computes momentum/volatility/Hurst, `technicals.py` recomputes all of them (with different Hurst algorithm); sentiment agent and macro analyst use same news with different LLM prompts (double-counting risk).
  *Priority: MEDIUM, Effort: 6-10h*

- [x] **No multi-period trend analysis** — Implemented for fundamentals through `analyze_metric_trend()` and wired into `fundamentals_agent` when `metrics_history` is present. Declining multi-period ROE/revenue-growth/debt trends can now push the evidence bearish even if latest absolute values still pass thresholds. Remaining future refinement: ensure market-data collection always provides 3-5 year metric histories.
  *Completed: 2026-04-26; Future refinement: 4-8h*
  ```python
  def analyze_metric_trend(metrics_history, metric_name="roe"):
      """Analyze multi-period trend for a metric"""
      values = [m[metric_name] for m in metrics_history[-5:]]  # Last 5 years
      if len(values) < 2:
          return {"trend": "insufficient_data", "signal": "neutral"}

      trend_direction = "improving" if values[-1] > values[0] else "declining"
      trend_strength = abs(values[-1] - values[0]) / abs(values[0]) if values[0] != 0 else 0

      # Adjust signal based on trend
      if trend_direction == "declining" and trend_strength > 0.20:
          return {"trend": "declining", "signal": "bearish", "reason": f"{metric_name} dropped {trend_strength:.1%} over 5 years"}
      return {"trend": trend_direction, "signal": "neutral"}
  ```

- [x] **No transaction cost modeling** — Implemented in `src/agents/portfolio_manager.py` via `calculate_transaction_costs()`: buy/sell costs include commission and transfer fee, sell also includes stamp tax, and the final decision payload includes gross return, transaction cost, net return, cost rate, and profitability flag.
  *Completed: 2026-04-26*
  ```python
  def calculate_transaction_costs(action, quantity, price, expected_return=0.0):
      """Calculate A-share transaction costs and net expected return."""
      if action == "buy":
          cost_rate = 0.00025 + 0.00002  # Commission + transfer fee
      elif action == "sell":
          cost_rate = 0.00025 + 0.001 + 0.00002  # Commission + stamp tax + transfer fee
      else:
          cost_rate = 0

      gross_return = quantity * price * expected_return
      transaction_cost = quantity * price * cost_rate
      net_return = gross_return - transaction_cost

      return {
          "gross_return": gross_return,
          "transaction_cost": transaction_cost,
          "net_return": net_return,
          "profitable": net_return > 0
      }
  ```

- [ ] **Single-stock decisions only** — No multi-stock portfolio construction, sector allocation, correlation-based diversification, or rebalancing logic. System analyzes stocks independently without considering portfolio-level risk.
  *Priority: LOW, Effort: 30-40h*

- [ ] **Bing search fallback unreliable** — Bing Playwright search returns 0 results in current environment. Should prioritize Eastmoney/Sina API tier and reduce reliance on Playwright-based search, or implement better fallback handling.
  *Converted from Known Issues, Priority: LOW, Effort: 4-6h*

- [ ] **OutputLogger stdout redirect conflicts** — `OutputLogger` stdout redirect can conflict with `@agent_endpoint` decorator's StringIO capture during parallel agent execution — some print statements may be lost. Needs thread-safe logging refactor.
  *Converted from Known Issues, Priority: LOW, Effort: 8-12h*

- [x] **Decision memory system wired into final decision workflow** — `src/tools/memory.py` defines optional ChromaDB-backed `AShareDecisionMemory`; `portfolio_manager.py` best-effort injects similar historical decision memory into the final prompt and stores the validated final decision. `build_outcome_reflection()` and `update_decision_outcome()` now support best-effort outcome updates after realized returns are known. Remaining future refinement: scheduled outcome jobs and performance evaluation of memory retrieval quality.
  *Completed: 2026-04-26; Future refinement: 6-10h*
  ```python
  from chromadb import Client
  class FinancialMemory:
      def __init__(self):
          self.client = Client()
          self.collection = self.client.create_collection("trading_decisions")

      def store(self, situation_features, decision, outcome):
          """Store decision with its outcome for future learning"""
          vector = self.embed(situation_features)  # Convert market situation to vector
          self.collection.add(
              embeddings=[vector],
              documents=[json.dumps({"decision": decision, "outcome": outcome})],
              metadatas=[{"date": datetime.now().isoformat(), "return": outcome}]
          )

      def recall_similar(self, current_situation, k=5):
          """Retrieve similar historical situations and their outcomes"""
          vector = self.embed(current_situation)
          results = self.collection.query(query_embeddings=[vector], n_results=k)
          return results['metadatas'][0]  # Return k most similar cases

  # Usage in portfolio_manager prompt:
  # similar_cases = memory.recall_similar(current_market_features)
  # prompt += f"\n\nLearn from {len(similar_cases)} similar historical cases: {similar_cases}"
  ```

- [ ] **Single-layer risk management** — Current risk agent is monolithic. Could enhance with multi-role debate architecture (aggressive/conservative/neutral risk analysts debate before final decision), similar to TradingAgents-CN's 5-stage architecture.
  *From TradingAgents-CN, Priority: LOW, Effort: 40-60h*
  ```python
  # Stage 4: Risk debate (after trader creates initial plan)
  def risk_debate_stage(trading_plan, market_data):
      risky_view = aggressive_risk_agent(trading_plan, market_data)
      safe_view = conservative_risk_agent(trading_plan, market_data)
      neutral_view = neutral_risk_agent(trading_plan, market_data)
      return {"risky": risky_view, "safe": safe_view, "neutral": neutral_view}

  # Stage 5: Risk judge synthesizes debate into final decision
  def risk_judge_stage(debate_results, historical_memory):
      prompt = f"Synthesize these risk perspectives: {debate_results}"
      similar_cases = historical_memory.recall_similar(current_situation)
      prompt += f"\nConsider these similar historical outcomes: {similar_cases}"
      return llm.generate(prompt)
  ```

---

## Quick Reference: TradingAgents-CN Insights

Key innovations from [TradingAgents-CN](https://github.com/hsliuping/TradingAgents-CN) that have been incorporated above:

### Architecture Patterns
- **5-Stage Workflow**: Analysts → Research Debate → Trader Plan → Risk Debate → Final Decision
- **Forced Tool Calling**: Code-level fallback when LLM refuses to invoke tools
- **Memory System**: ChromaDB vector store for learning from historical decisions
- **Multi-Role Debate**: Aggressive/Conservative/Neutral analysts debate before consensus

### Data Layer Improvements
- **Multi-Source Fallback**: MongoDB → Tushare → AKShare (curl_cffi) → BaoStock
- **Anti-Scraping**: curl_cffi with Chrome impersonation to bypass TLS fingerprinting
- **Persistent Caching**: SQLite/MongoDB for 90% local hit rate

### A-Share Specific Features
- **Northbound Flows**: Foreign capital tracking via `stock_hsgt_north_net_flow_in_em`
- **Margin Trading**: Leverage sentiment via `stock_margin_detail_szse/sse`
- **KDJ Indicator**: Most popular Chinese market technical indicator
- **MACD Divergence**: Top/bottom reversal signal detection

### LLM Engineering
- **Quick/Deep Model Separation**: Use fast models (Gemini Flash) for tool calls, slow models (Gemini Pro) for complex reasoning
- **Strict Prompting**: "🔴 MANDATORY" and "🚫 ABSOLUTELY FORBIDDEN" patterns to enforce tool usage
- **Weighted Scoring**: Programmatic signal combination with LLM-only qualitative overlay

---

## Implementation Priority Summary

**Immediate (This Week)** — Fix P0 critical defects affecting accuracy:
1. ~~Macro analyst data source mismatch~~ ✅
2. ~~Shenzhen stock financial statements bug~~ ✅
3. ~~`format_decision()` weight mismatch~~ ✅
4. ~~Missing KDJ indicator~~ ✅
5. ~~Missing northbound capital flow~~ ✅

**Short Term (Next 2-4 Weeks)** — Fill P1 important gaps:
6. ~~Valuation model improvements (PEG, comparables, net-debt/EV adjustments)~~ ✅
7. ~~Industry-specific thresholds, peer percentile helper, and multi-period trend analysis~~ ✅
8. ~~Anti-scraping protection (curl_cffi)~~ ✅ (core done, tuning remaining)
9. ~~Persistent caching layer and symbol cache prewarm~~ (partially done: `SimpleCache` restored, DSM cache injection working, `prewarm_symbol_cache()` added; broader function coverage remains)
10. ~~Risk management enhancements (CVaR, Beta, liquidity, T+1, stress testing)~~ ✅
11. ~~Margin trading sentiment data~~ ✅

**Medium Term (1-3 Months)** — P2 architecture improvements:
12. ~~Decision memory workflow integration and outcome update helpers~~ ✅
13. ~~Transaction cost modeling~~ ✅
14. Broader sentiment platforms (Xueqiu/Snowball, Weibo)
15. Technical gaps: turnover, support/resistance, MA support, dead-code cleanup
16. Cross-library fallback hardening and broader DSM coverage

**Long Term (3+ Months)** — Advanced features:
17. Multi-stock portfolio optimization
18. Multi-role risk debate architecture
19. Decision-memory scheduled outcome evaluation

---

*Last Updated: 2026-04-26*
*Total Optimization Items: 28*
*Completed: 17 (KDJ, northbound flow, MACD divergence, SignalWeightingEngine, dynamic WACC, cache restoration, valuation PEG/comparables/net-debt, risk CVaR/liquidity/T+1, final decision memory/fallback, macro evidence scoring/fallback, industry thresholds/peer percentile/trend analysis, margin trading sentiment, symbol cache prewarm, Shenzhen financial statement prefix fix, format_decision weight sync, transaction cost modeling, memory outcome update helpers)*
*Partially Completed/In Progress: 5 (sentiment breadth, anti-scraping tuning, broader persistent caching coverage, cross-library fallback hardening, technical-analysis cleanup)*
*From TradingAgents-CN: 11*
*Estimated Remaining Effort: 90-130 hours*
