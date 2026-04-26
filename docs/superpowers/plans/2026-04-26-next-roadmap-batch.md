# Next Roadmap Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the next finite roadmap batch with no plan-level leftovers: macro evidence scoring, fundamentals trend/peer checks, A-share margin data, transaction costs, cache prewarm, memory outcome update, and stale roadmap short-item cleanup.

**Architecture:** Keep agent boundaries intact. Add deterministic helper functions with unit tests first, then wire their outputs into existing agent reports and downstream prompts. External data calls must degrade to neutral/unavailable and use `DataSourceManager` where persistent cache is useful.

**Tech Stack:** Python 3.9+, LangGraph agents, pandas/numpy, akshare, optional ChromaDB, pytest.

---

## File Structure

- Modify: `src/agents/macro_analyst.py`
  - Deterministic macro scoring and prompt evidence guardrails.
- Modify: `src/agents/fundamentals.py`
  - Peer percentile and multi-period trend helper analysis.
- Modify: `src/tools/api.py`
  - Margin trading sentiment accessor and cache prewarm function.
- Modify: `src/agents/sentiment.py`
  - Include margin trading sentiment in structured sentiment reasoning when available.
- Modify: `src/agents/portfolio_manager.py`
  - Transaction cost helper and final decision annotation; confirm weight display is consistent.
- Modify: `src/tools/memory.py`
  - Outcome update helper for stored decision/reflection memory.
- Modify: `AGENTS.md`, `README.md`, `CHANGELOG.md`
  - Sync roadmap status after implementation, per Documentation Sync Rule.
- Add tests:
  - `tests/unit/test_macro_analyst_p1.py`
  - `tests/unit/test_fundamentals_p1.py`
  - `tests/unit/test_api_margin_cache_p1.py`
  - `tests/unit/test_portfolio_costs_p1.py`
  - `tests/unit/test_memory_p1.py`

---

### Task 1: Macro Evidence Scoring

**Files:**
- Modify: `src/agents/macro_analyst.py`
- Test: `tests/unit/test_macro_analyst_p1.py`

- [ ] Add tests for `score_macro_environment()`:
  - Positive PMI/GDP/M2 and mild CPI produces `positive`.
  - Weak PMI/GDP and high CPI produces `negative`.
  - Empty indicators returns neutral with `data_available=False`.
- [ ] Verify RED with `PYTHONPATH=. pytest tests/unit/test_macro_analyst_p1.py -q`.
- [ ] Implement `score_macro_environment(macro_indicators)` returning `macro_environment`, `impact_on_stock`, `score`, `key_factors`, `data_available`, and `reasoning`.
- [ ] In `get_macro_news_analysis()`, use deterministic score as fallback when LLM fails or returns invalid JSON; include deterministic score in returned payload and prompt context.
- [ ] Verify GREEN and full unit tests.

### Task 2: Fundamentals Peer Percentiles and Multi-Period Trends

**Files:**
- Modify: `src/agents/fundamentals.py`
- Test: `tests/unit/test_fundamentals_p1.py`

- [ ] Add tests for `analyze_metric_trend()` and `calculate_peer_percentile_signal()`.
- [ ] Verify RED.
- [ ] Implement trend helper: declining >20% over history is bearish, improving >20% bullish, otherwise neutral.
- [ ] Implement peer percentile helper using optional `industry_peer_metrics` from state data. Higher-is-better metrics should score bullish above 70th percentile, bearish below 30th; lower-is-better metrics invert that.
- [ ] Wire helpers into `fundamentals_agent()` reasoning without breaking current static thresholds.
- [ ] Verify GREEN and full unit tests.

### Task 3: A-Share Margin Trading Sentiment

**Files:**
- Modify: `src/tools/api.py`
- Modify: `src/agents/sentiment.py`
- Test: `tests/unit/test_api_margin_cache_p1.py`

- [ ] Add tests for `get_margin_trading_sentiment()` by monkeypatching SZSE/SSE akshare functions through `DataSourceManager.fetch_with_fallback`.
- [ ] Verify RED.
- [ ] Implement `get_margin_trading_sentiment(symbol)` with neutral fallback and DSM cache.
- [ ] Add margin sentiment into `sentiment_agent()` reasoning and weighted score when data is available.
- [ ] Verify GREEN and full unit tests.

### Task 4: Transaction Cost Modeling and Weight Display Audit

**Files:**
- Modify: `src/agents/portfolio_manager.py`
- Test: `tests/unit/test_portfolio_costs_p1.py`

- [ ] Add tests for `calculate_transaction_costs()` buy/sell/hold costs.
- [ ] Add test asserting `format_decision()` weight labels match `SIGNAL_WEIGHTS`: valuation 30, fundamentals 25, technical 20, macro 15, sentiment 10.
- [ ] Verify RED for missing transaction helper.
- [ ] Implement `calculate_transaction_costs(action, quantity, price, expected_return=0.0)`.
- [ ] Annotate final decision JSON with `transaction_costs` before message creation.
- [ ] Verify GREEN and full unit tests.

### Task 5: Cache Prewarm and DSM Coverage Helper

**Files:**
- Modify: `src/tools/api.py`
- Test: `tests/unit/test_api_margin_cache_p1.py`

- [ ] Add tests for `prewarm_symbol_cache(symbol, start_date, end_date)` with monkeypatched data functions.
- [ ] Implement prewarm helper that calls `get_price_history`, `get_financial_metrics`, `get_financial_statements`, `get_market_data`, `get_northbound_flow`, `get_macro_indicators`, and `get_margin_trading_sentiment`, returning per-step status instead of raising.
- [ ] Verify tests and full unit suite.

### Task 6: Decision Memory Outcome Update

**Files:**
- Modify: `src/tools/memory.py`
- Test: `tests/unit/test_memory_p1.py`

- [ ] Add tests for pure helper `build_outcome_reflection()` and best-effort `update_decision_outcome()`.
- [ ] Implement `build_outcome_reflection(decision, actual_return, days_held)` with clear lessons for buy/sell/hold.
- [ ] Implement `update_decision_outcome(ticker, date, situation_summary, decision, actual_return, days_held)` that calls `store_reflection()` and never raises when memory is unavailable.
- [ ] Verify GREEN and full unit tests.

### Task 7: Roadmap Cleanup and Final Verification

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`
- Modify: `CHANGELOG.md`

- [ ] Mark Shenzhen financial statement prefix bug complete if code still uses `_normalize_ak_symbol()` for all financial statement calls.
- [ ] Mark `format_decision()` weight mismatch complete if tests prove output labels match `SIGNAL_WEIGHTS`.
- [ ] Mark macro, fundamentals, margin trading, transaction cost, cache prewarm, and memory outcome update statuses to match implementation.
- [ ] Run:
  - `PYTHONPATH=. pytest tests/unit -q`
  - `PYTHONPATH=. python3 -c "import src.main"`
  - `PYTHONPATH=. python3 -m compileall src tests`
  - `git diff --check`
  - `rg -n "still only implements|仍未完成|仍在实现中|尚未完成|hardcoded 10%|weight mismatch|Shenzhen stock financial statements bug" AGENTS.md README.md CHANGELOG.md`

---

## Completion Criteria

- Every task above is implemented or explicitly proven already complete.
- All new helper functions have focused unit tests.
- Full unit suite passes.
- `src.main` imports successfully.
- Documentation reflects the actual code state.
- No generated cache DB changes are left in the working tree.
