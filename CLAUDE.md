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

The following improvements are planned, ordered by impact and importance within each priority level. Check items off as they are completed.

### P0: Core Defects (Severely Impact Accuracy)

*Critical issues that produce fundamentally wrong analysis results. Fix immediately.*

- [ ] **Macro analyst data source mismatch** — `macro_analyst.py` fetches *stock-specific* news but tries to analyze the *macro economy*. Should integrate real macro data: GDP, CPI, PMI, M2, LPR, yield curves. akshare provides `macro_china_gdp`, `macro_china_cpi`, `macro_china_pmi`, etc.
  *Priority: CRITICAL, Effort: 8-12h*

- [ ] **Valuation model too crude** — DCF discount rate hardcoded at 10% (should compute WACC dynamically); negative earnings growth floored to 0 (overvalues declining companies); missing comparable company valuation (industry P/E, P/B, EV/EBITDA); missing PEG ratio; enterprise value vs equity value conflated (FCF compared directly to market cap without subtracting net debt).
  *Priority: CRITICAL, Effort: 16-24h*

- [ ] **Fundamentals thresholds are industry-agnostic** — ROE>15%, net margin>20%, debt ratio<50% applied uniformly across all sectors. Banks, tech, brokerages all have different norms. Need industry classification and relative percentile comparison.
  *Priority: CRITICAL, Effort: 12-18h*

- [ ] **Sentiment analysis too narrow** — Only analyzes news articles; missing retail investor sentiment (Eastmoney guba, Xueqiu/Snowball, Weibo); all news collapsed to single -1~1 score losing structural information; no temporal decay weighting.
  *Priority: CRITICAL, Effort: 24-36h*

- [ ] **Missing KDJ indicator** — KDJ (9,3,3) is the most popular technical indicator in Chinese markets, but current system only has MACD/RSI/BB/OBV. Retail investors and institutions both heavily rely on KDJ golden/dead crosses and overbought/oversold signals.
  *From TradingAgents-CN, Priority: HIGH, Effort: 2-4h*
  ```python
  def calculate_kdj(prices_df, n=9, m1=3, m2=3):
      low_list = prices_df['low'].rolling(window=n, min_periods=n).min()
      high_list = prices_df['high'].rolling(window=n, min_periods=n).max()
      rsv = (prices_df['close'] - low_list) / (high_list - low_list) * 100
      k = rsv.ewm(alpha=1/m1, adjust=False).mean()
      d = k.ewm(alpha=1/m2, adjust=False).mean()
      j = 3 * k - 2 * d
      # Signal: bullish when J<0 and K<20 (oversold), bearish when J>100 and K>80 (overbought)
      current_k, current_d, current_j = k.iloc[-1], d.iloc[-1], j.iloc[-1]
      if current_j < 0 and current_k < 20:
          signal = "bullish"
      elif current_j > 100 and current_k > 80:
          signal = "bearish"
      else:
          signal = "neutral"
      return {"k": current_k, "d": current_d, "j": current_j, "signal": signal}
  ```

- [ ] **Missing northbound capital flow data** — Foreign capital flows (北向资金) are critical "smart money" signals in A-share markets. Current system completely ignores this A-share specific data source.
  *From TradingAgents-CN, Priority: HIGH, Effort: 1-2h*
  ```python
  import akshare as ak
  def get_northbound_flow(days=5):
      df = ak.stock_hsgt_north_net_flow_in_em()
      total_inflow = df.tail(days)['净流入'].sum()
      # Signal: bullish if total_inflow > 10 billion (moderate inflow)
      if total_inflow > 50:
          trend, signal = "strong_inflow", "bullish"
      elif total_inflow > 10:
          trend, signal = "moderate_inflow", "bullish"
      elif total_inflow > -10:
          trend, signal = "neutral", "neutral"
      elif total_inflow > -50:
          trend, signal = "moderate_outflow", "bearish"
      else:
          trend, signal = "strong_outflow", "bearish"
      return {"net_inflow_billion": total_inflow, "trend": trend, "signal": signal}
  ```

- [ ] **Missing MACD divergence detection** — Current system only detects MACD crossovers. Top/bottom divergence (顶背离/底背离) is a critical reversal signal: when price makes new high but MACD doesn't = top divergence (bearish); price makes new low but MACD doesn't = bottom divergence (bullish).
  *From TradingAgents-CN, Priority: HIGH, Effort: 4-6h*
  ```python
  def detect_macd_divergence(prices_df):
      macd_line, signal_line = calculate_macd(prices_df)
      recent_prices = prices_df['close'].tail(60)
      recent_macd = macd_line.tail(60)
      # Top divergence: price new high, MACD not new high
      price_highs = recent_prices[recent_prices == recent_prices.rolling(5).max()]
      macd_highs = recent_macd[recent_macd == recent_macd.rolling(5).max()]
      if len(price_highs) >= 2 and len(macd_highs) >= 2:
          if price_highs.iloc[-1] > price_highs.iloc[-2] and macd_highs.iloc[-1] < macd_highs.iloc[-2]:
              return {"divergence": "top", "signal": "bearish", "message": "Top divergence: Price new high but MACD not - potential reversal"}
      # Bottom divergence detection similarly...
      price_lows = recent_prices[recent_prices == recent_prices.rolling(5).min()]
      macd_lows = recent_macd[recent_macd == recent_macd.rolling(5).min()]
      if len(price_lows) >= 2 and len(macd_lows) >= 2:
          if price_lows.iloc[-1] < price_lows.iloc[-2] and macd_lows.iloc[-1] > macd_lows.iloc[-2]:
              return {"divergence": "bottom", "signal": "bullish", "message": "Bottom divergence: Price new low but MACD not - potential bounce"}
      return {"divergence": None, "signal": "neutral"}
  ```

- [ ] **No forced tool calling mechanism** — LLM sometimes ignores tools and generates hallucinated analysis with fake data. TradingAgents-CN implements code-level fallback: if LLM returns empty tool_calls, Python code directly invokes the data tool and forces LLM to analyze real data.
  *From TradingAgents-CN, Priority: MEDIUM, Effort: 8-12h*
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

- [ ] **Missing A-share specific data** — Northbound capital flows (`stock_hsgt_north_net_flow_in_em`), margin trading (`stock_margin_detail_szse/sse`), block trades (`stock_dzjy_sctj`), top 10 shareholders (`stock_gdfx_free_holding_analyse_em`), dragon & tiger list (`stock_lhb_detail_em`), restricted share unlocks (`stock_restricted_release_queue_sina`), shareholder count changes (`stock_zh_a_gdhs`).
  *Priority: HIGH, Effort: 10-16h*

- [ ] **Risk management too basic** — Only historical VaR, missing CVaR/Expected Shortfall and Monte Carlo; no Beta coefficient calculation; no liquidity risk assessment; stress testing is trivially simple (flat % declines); T+1 settlement rule completely ignored.
  *Priority: HIGH, Effort: 12-18h*

- [ ] **Final decision over-relies on LLM** — Portfolio manager's signal weights (valuation 30%/fundamentals 25%/technical 20%/macro 15%/sentiment 10%) exist only in the prompt text with no programmatic enforcement. Should compute weighted scores in code, use LLM only for qualitative adjustment and explanation.
  *Priority: HIGH, Effort: 10-16h*

- [ ] **Technical analysis gaps** — Missing KDJ indicator (most popular in Chinese markets); no MACD divergence detection (only crossover); turnover rate data available but never used; no support/resistance level identification; no MA5/10/20/60 support analysis; first-layer indicators (MACD/RSI/BB/OBV) computed then overwritten by second-layer strategies (dead code).
  *Note: Partially addressed by KDJ and MACD divergence in P0*
  *Priority: MEDIUM, Effort: 14-20h*

- [ ] **No anti-scraping protection** — Eastmoney API intermittently returns `RemoteDisconnected` because current requests lack browser TLS fingerprint simulation. TradingAgents-CN uses `curl_cffi` to impersonate Chrome browser and bypass anti-bot measures.
  *From TradingAgents-CN, Priority: HIGH, Effort: 4-6h*
  ```python
  from curl_cffi import requests as curl_requests
  def fetch_with_impersonation(url):
      # Simulate Chrome 110 TLS fingerprint to bypass anti-scraping
      response = curl_requests.get(url, impersonate="chrome110")
      return response.json()
  ```

- [ ] **No persistent caching** — System refetches all K-line and financial data on every run, wasting API quota and time. Should implement SQLite/MongoDB cache layer so 90% of requests hit local database.
  *From TradingAgents-CN, Priority: HIGH, Effort: 12-16h*
  ```python
  # Daily cron job syncs data to local DB
  def sync_daily_data():
      for symbol in watchlist:
          data = ak.stock_zh_a_hist(symbol=symbol)
          sqlite_db.insert(f"hist_{symbol}", data, date=today)
  
  # Agent queries local DB first
  def get_prices(symbol):
      if cached := sqlite_db.query(f"hist_{symbol}", date=today):
          return cached  # 90% hit rate
      return fetch_from_api(symbol)  # Fallback
  ```

- [ ] **Valuation missing dynamic WACC** — Current DCF uses hardcoded 10% discount rate, overvaluing high-risk stocks and undervaluing low-risk ones. Should calculate WACC dynamically using Beta coefficient.
  *From TradingAgents-CN, Priority: MEDIUM, Effort: 8-12h*
  ```python
  def calculate_wacc(symbol, risk_free_rate=0.03):
      beta = ak.stock_beta_em(symbol=symbol)  # From akshare
      market_premium = 0.06  # A-share historical risk premium
      cost_of_equity = risk_free_rate + beta * market_premium
      debt_cost = get_debt_cost_from_financials(symbol)
      debt_ratio = get_debt_ratio(symbol)
      wacc = cost_of_equity * (1 - debt_ratio) + debt_cost * debt_ratio * 0.75  # Tax shield
      return max(wacc, 0.08)  # Minimum 8% floor
  ```

- [ ] **Missing margin trading data** — Margin trading (融资融券) reflects leverage sentiment: high margin balance = bullish retail sentiment; increasing short selling = bearish. This is A-share specific and currently missing.
  *From TradingAgents-CN, Priority: MEDIUM, Effort: 2-4h*
  ```python
  def get_margin_trading_sentiment(symbol):
      # Shenzhen or Shanghai market
      if symbol.startswith(('0', '3')):
          df = ak.stock_margin_detail_szse(symbol=symbol)
      else:
          df = ak.stock_margin_detail_sse(symbol=symbol)
      margin_change = df.iloc[0]['融资余额'] - df.iloc[1]['融资余额']
      ms_ratio = df.iloc[0]['融资余额'] / df.iloc[0]['融券余额'] if df.iloc[0]['融券余额'] > 0 else float('inf')
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
  ```python
  def get_stock_data_with_fallback(symbol):
      # Priority: MongoDB -> Tushare -> AKShare (with curl_cffi) -> BaoStock
      if data := mongo_cache.get(symbol):
          return data
      if data := try_tushare(symbol):
          mongo_cache.set(symbol, data)
          return data
      if data := try_akshare_curl(symbol):
          mongo_cache.set(symbol, data)
          return data
      return try_baostock(symbol)  # Final fallback
  ```

### P2: Architecture Improvements (Robustness)

*Improvements that enhance system robustness, maintainability, and scalability.*

- [ ] **Shenzhen stock financial statements bug** — `get_financial_statements()` hardcodes `sh` prefix for Sina API calls; stocks starting with 0 or 3 (Shenzhen-listed) will fail to retrieve financial data. `_normalize_ak_symbol()` exists but is not used for these calls.
  *Priority: HIGH, Effort: 1-2h*
  ```python
  # Fix: Use _normalize_ak_symbol instead of hardcoded prefix
  def get_financial_statements(symbol):
      normalized = _normalize_ak_symbol(symbol)  # Returns sh601xxx or sz000xxx
      return ak.stock_financial_report_sina(stock=normalized, symbol="资产负债表")
  ```

- [ ] **`format_decision()` weight mismatch** — Displays 30/35/25 weights for fundamental/valuation/technical, contradicting the 25/30/20 in the system prompt. Need to unify weight definitions across system prompt, code logic, and output display.
  *Priority: HIGH, Effort: 1-2h*

- [ ] **Data pipeline redundancy** — `api.py` pre-computes momentum/volatility/Hurst, `technicals.py` recomputes all of them (with different Hurst algorithm); sentiment agent and macro analyst use same news with different LLM prompts (double-counting risk).
  *Priority: MEDIUM, Effort: 6-10h*

- [ ] **No multi-period trend analysis** — All agents only look at latest period. ROE dropping from 30% to 16% still scores bullish because 16%>15%. Should compare 3-5 year trends and penalize declining trends even if absolute value is above threshold.
  *Priority: MEDIUM, Effort: 8-12h*
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

- [ ] **No transaction cost modeling** — A-share costs: commission (~0.025%), stamp tax (0.1% sell-side), transfer fee. Not accounted for in any decision. Should calculate net return after costs before recommending trades.
  *Priority: MEDIUM, Effort: 4-6h*
  ```python
  def calculate_net_return(action, quantity, price, expected_return):
      """Calculate return after A-share transaction costs"""
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

- [ ] **No decision memory system** — System doesn't learn from past mistakes. TradingAgents-CN uses ChromaDB to store "market situation vectors" and retrieval-augmented generation (RAG) to recall similar historical scenarios, enabling the system to "remember" and avoid repeating errors.
  *From TradingAgents-CN, Priority: LOW, Effort: 20-30h*
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
1. Macro analyst data source mismatch
2. Shenzhen stock financial statements bug
3. `format_decision()` weight mismatch
4. Missing KDJ indicator
5. Missing northbound capital flow

**Short Term (Next 2-4 Weeks)** — Fill P1 important gaps:
6. Valuation model improvements (dynamic WACC, PEG)
7. Industry-specific thresholds
8. Anti-scraping protection (curl_cffi)
9. Persistent caching layer
10. Risk management enhancements (CVaR, Beta, stress testing)

**Medium Term (1-3 Months)** — P2 architecture improvements:
11. Decision memory system (ChromaDB)
12. Multi-period trend analysis
13. Transaction cost modeling
14. Multi-role risk debate architecture

**Long Term (3+ Months)** — Advanced features:
15. Multi-stock portfolio optimization
16. Advanced sentiment analysis (Weibo, Xueqiu)

---

*Last Updated: 2025-04-17*
*Total Optimization Items: 28*
*From TradingAgents-CN: 11*
*Estimated Total Effort: 150-200 hours*
