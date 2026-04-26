# Roadmap Iteration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the project to a testable baseline, sync roadmap documentation with the actual code, and complete the next planned accuracy improvements for risk, valuation, and LLM/data-grounding.

**Architecture:** Keep the existing LangGraph workflow and agent boundaries. Add small, testable helper functions close to their consumers, preserve graceful degradation when optional dependencies or external data sources are unavailable, and expose all new deterministic outputs through `state["data"]` so downstream agents can use structured facts instead of prompt-only text.

**Tech Stack:** Python 3.9+, LangGraph, pandas/numpy, akshare, optional ChromaDB, pytest.

---

## File Structure

- Create: `src/tools/cache/__init__.py` and `src/tools/cache/db_cache.py`
  - SQLite-backed JSON/pickle cache used by `DataSourceManager`.
- Modify: `src/tools/data_source_manager.py`
  - Keep cache fallback behavior, make imports resolvable, and support cache age checks.
- Modify: `tests/unit/test_api_p0.py`
  - Add cache and import regression coverage.
- Modify: `AGENTS.md`, `README.md`, `CHANGELOG.md`
  - Align roadmap progress with actual implementation and remaining gaps.
- Modify: `src/agents/risk_manager.py`
  - Add CVaR/Expected Shortfall, beta-aware market risk, liquidity inputs, T+1 rule metadata, and richer stress tests.
- Modify: `src/tools/api.py`
  - Add helper data accessors for margin/liquidity or valuation comparables when feasible without hard dependencies.
- Modify: `src/agents/valuation.py`
  - Add PEG/comparable valuation and net-debt/EV-aware valuation adjustments.
- Modify: `src/agents/portfolio_manager.py`
  - Inject decision memory into final prompt, store final decisions, and add forced-data fallback for empty/invalid LLM results.
- Modify: `src/tools/memory.py`
  - Keep optional ChromaDB behavior and provide safe no-op prompt/store behavior when unavailable.
- Add/modify tests under `tests/unit/`
  - Focused unit tests for each new deterministic behavior.

---

### Task 1: Restore Runnable Baseline

**Files:**
- Create: `src/tools/cache/__init__.py`
- Create: `src/tools/cache/db_cache.py`
- Modify: `tests/unit/test_api_p0.py`

- [ ] **Step 1: Write failing cache/import tests**

Add tests proving `DataSourceManager` imports cleanly and `SimpleCache` respects TTL expiration:

```python
from src.tools.cache.db_cache import SimpleCache
from src.tools.data_source_manager import DataSourceManager


def test_data_source_manager_import_and_cache_roundtrip(tmp_path):
    cache = SimpleCache(db_path=tmp_path / "cache.db")
    cache.set("k", {"value": 1}, ttl_hours=1)
    assert cache.get("k", max_age_hours=1) == {"value": 1}
    assert DataSourceManager(cache=cache).fetch_with_fallback(
        cache_key="k",
        fetchers=[lambda: {"value": 2}],
        source_names=["fallback"],
        cache_ttl_hours=1,
    ) == {"value": 1}


def test_simple_cache_treats_zero_age_as_expired(tmp_path):
    cache = SimpleCache(db_path=tmp_path / "cache.db")
    cache.set("expired", {"value": 1}, ttl_hours=1)
    assert cache.get("expired", max_age_hours=0) is None
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m pytest tests/unit/test_api_p0.py -q`

Expected before implementation: fails with `ModuleNotFoundError: No module named 'src.tools.cache'`.

- [ ] **Step 3: Implement `SimpleCache`**

Create a SQLite-backed cache that stores JSON when possible and pickle as fallback. API:

```python
cache = SimpleCache(db_path="src/data/cache.db")
cache.get(cache_key, max_age_hours=24)
cache.set(cache_key, value, ttl_hours=24)
```

Behavior:
- Creates parent directory and table automatically.
- Returns `None` when key missing, TTL expired, deserialization fails, or stored payload is invalid.
- Does not crash if cache write fails; logs and continues.
- Supports pandas/numpy-containing values through pickle fallback.

- [ ] **Step 4: Make `DataSourceManager` injectable**

Change constructor to:

```python
def __init__(self, cache: SimpleCache | None = None) -> None:
    self.cache = cache or SimpleCache()
```

- [ ] **Step 5: Verify GREEN**

Run:

```bash
python3 -m pytest tests/unit/test_api_p0.py -q
python3 -c "import src.tools.data_source_manager; import src.tools.api"
```

Expected: tests pass and imports exit 0.

---

### Task 2: Sync Roadmap Documentation

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update roadmap statuses**

Reflect actual code state:
- Macro analyst data source mismatch: partially completed. Macro indicators and industry news are integrated; remaining work is richer macro scoring and validation.
- Valuation model: partially completed. Dynamic WACC and negative growth handling are integrated; remaining work is comparables/PEG/net debt.
- Fundamentals thresholds: partially completed. Static industry thresholds exist; remaining work is percentile-based peer comparison.
- Sentiment: partially completed. Time decay and forum sentiment exist; remaining work is broader platforms and structural decomposition.
- Persistent caching: blocked/fixed by Task 1; remaining work is full function coverage and cache pre-warming.
- Decision memory: module exists; Task 5 integrates it into the workflow.

- [ ] **Step 2: Add current execution note**

Add a dated `2026-04-26` note to `CHANGELOG.md` describing:
- Restored cache module.
- Risk management enhancements.
- Valuation comparables/PEG additions.
- Memory/forced fallback integration.
- Verification commands used.

- [ ] **Step 3: Verify docs references**

Run: `rg -n "hardcoded 10%|Missing KDJ|No persistent caching|Decision memory|CVaR|PEG|SimpleCache|2026-04-26" AGENTS.md README.md CHANGELOG.md`

Expected: no stale statements claiming completed code is fully missing; remaining gaps are explicit.

---

### Task 3: Enhance Risk Management

**Files:**
- Modify: `src/agents/risk_manager.py`
- Modify: `tests/unit/test_risk_manager_p1.py`

- [ ] **Step 1: Write risk metric tests**

Add focused tests for pure helpers:

```python
import pandas as pd
from src.agents.risk_manager import (
    calculate_expected_shortfall,
    calculate_liquidity_risk,
    calculate_t1_settlement_constraint,
)


def test_expected_shortfall_averages_tail_losses():
    returns = pd.Series([-0.10, -0.05, -0.02, 0.01, 0.03])
    assert calculate_expected_shortfall(returns, confidence=0.8) == -0.10


def test_liquidity_risk_flags_low_turnover_and_amount():
    result = calculate_liquidity_risk(
        prices_df=pd.DataFrame({"turnover": [0.1, 0.2, 0.15], "amount": [1_000_000, 900_000, 800_000]}),
        max_position_size=5_000_000,
    )
    assert result["score"] >= 2
    assert result["signal"] in {"elevated", "high"}


def test_t1_constraint_blocks_same_day_sell_without_prior_position():
    result = calculate_t1_settlement_constraint(portfolio={"stock": 100, "sellable_stock": 0})
    assert result["sellable_stock"] == 0
    assert result["can_sell"] is False
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m pytest tests/unit/test_risk_manager_p1.py -q`

Expected: fails because helpers do not exist.

- [ ] **Step 3: Implement helpers and integrate output**

Add helper functions:
- `calculate_expected_shortfall(returns, confidence=0.95) -> float`
- `calculate_liquidity_risk(prices_df, max_position_size) -> dict`
- `calculate_t1_settlement_constraint(portfolio) -> dict`
- optional `calculate_beta_risk(beta) -> dict`

Include new fields under `risk_metrics`:
- `conditional_value_at_risk_95`
- `liquidity_risk`
- `t1_settlement`
- `beta_risk`
- `stress_test_results` with at least three named scenarios using volatility-aware shocks.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python3 -m pytest tests/unit/test_risk_manager_p1.py -q
python3 -m pytest tests/unit -q
```

Expected: all unit tests pass.

---

### Task 4: Enhance Valuation

**Files:**
- Modify: `src/agents/valuation.py`
- Modify: `tests/unit/test_valuation_p1.py`

- [ ] **Step 1: Write valuation helper tests**

Add tests:

```python
from src.agents.valuation import calculate_peg_ratio, calculate_comparable_value, adjust_equity_value_for_net_debt


def test_calculate_peg_ratio_handles_positive_growth():
    assert calculate_peg_ratio(pe_ratio=20, earnings_growth=0.25) == 0.8


def test_calculate_peg_ratio_returns_none_for_nonpositive_growth():
    assert calculate_peg_ratio(pe_ratio=20, earnings_growth=0) is None
    assert calculate_peg_ratio(pe_ratio=20, earnings_growth=-0.1) is None


def test_comparable_value_uses_pe_pb_average_when_available():
    value = calculate_comparable_value(
        metrics={"earnings_per_share": 2.0, "book_value_per_share": 10.0},
        comparable_multiples={"pe": 15, "pb": 2},
        shares_outstanding=1000,
    )
    assert value == 32_500


def test_adjust_equity_value_subtracts_net_debt():
    assert adjust_equity_value_for_net_debt(enterprise_value=100, cash=20, total_debt=40) == 80
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m pytest tests/unit/test_valuation_p1.py -q`

Expected: fails because helpers do not exist.

- [ ] **Step 3: Implement helpers and integrate reasoning**

Add:
- `calculate_peg_ratio(pe_ratio, earnings_growth)`
- `calculate_comparable_value(metrics, comparable_multiples, shares_outstanding)`
- `adjust_equity_value_for_net_debt(enterprise_value, cash, total_debt)`

Use available metrics only; if shares/cash/debt are unavailable, degrade gracefully and mark data unavailable in reasoning.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python3 -m pytest tests/unit/test_valuation_p1.py -q
python3 -m pytest tests/unit -q
```

Expected: all unit tests pass.

---

### Task 5: Integrate Forced Fallback and Decision Memory

**Files:**
- Modify: `src/agents/portfolio_manager.py`
- Modify: `src/tools/memory.py`
- Modify: `tests/unit/test_portfolio_manager_p1.py`

- [ ] **Step 1: Write deterministic fallback tests**

Add tests for pure helper functions:

```python
from src.agents.portfolio_manager import build_forced_decision, enrich_prompt_with_memory


def test_build_forced_decision_uses_base_decision():
    result = build_forced_decision(
        engine_result={"base_decision": "buy", "weighted_score": 0.4},
        agent_signals={"technical": {"signal": "bullish", "confidence": "80%"}},
    )
    assert result["action"] == "buy"
    assert result["reasoning"]


def test_enrich_prompt_with_memory_appends_memory_when_available():
    prompt = enrich_prompt_with_memory("base", "memory text")
    assert "base" in prompt
    assert "memory text" in prompt
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m pytest tests/unit/test_portfolio_manager_p1.py -q`

Expected: fails because helper functions do not exist.

- [ ] **Step 3: Implement helpers and integrate**

Add:
- `build_forced_decision(engine_result, agent_signals, current_price=None) -> dict`
- `enrich_prompt_with_memory(user_prompt, memory_prompt) -> str`

In `portfolio_management_agent`:
- Before LLM call, call `get_memory().get_memory_prompt(state["data"])` and append when non-empty.
- If LLM returns `None`, invalid JSON, or misses required decision fields, use `build_forced_decision`.
- After validation and formatting, call `get_memory().store_decision(...)` best-effort; never fail the workflow if memory is unavailable.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python3 -m pytest tests/unit/test_portfolio_manager_p1.py -q
python3 -m pytest tests/unit -q
python3 -c "import src.main"
```

Expected: all unit tests pass and workflow imports.

---

## Final Verification

- [ ] Run `python3 -m pytest tests/unit -q`.
- [ ] Run `python3 -c "import src.main"`.
- [ ] Run `python3 -m compileall src tests`.
- [ ] Run `git status --short` and review all changed files.
- [ ] Update this plan checkboxes or summarize any deviations in the final response.
