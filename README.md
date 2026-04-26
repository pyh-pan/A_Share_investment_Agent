# AI 投资系统 — A 股多智能体分析

基于 LangGraph 的 A 股多智能体投资分析系统。12 个 Agent 协同工作，通过多轮多空辩论、确定性信号加权和 LLM 增强，为单只 A 股生成买入/卖出/持有决策。

> ⚠️ 本项目仅用于教育和研究目的，不构成任何投资建议。

---

## 功能亮点

- **多轮多空辩论** — 看多/看空研究员交替辩论（最多 5 轮），辩论室 LLM 裁判综合评判
- **确定性信号加权** — `SignalWeightingEngine` 在代码中计算加权评分（估值 30% / 基本面 25% / 技术 20% / 宏观 15% / 情绪 10%），LLM 仅做定性调整
- **A 股专属指标** — 北向资金流向、KDJ (9,3,3)、MACD 顶/底背离、WACC 动态折现率
- **四级新闻源** — 东方财富直连 API → 东方财富 + 新浪 + 同花顺 → Playwright 搜索引擎 → akshare 兜底
- **SQLite 缓存 + 多源降级** — `DataSourceManager` 统一缓存（行情 4h / 财务 24h），akshare → tushare → baostock 降级链
- **反反爬** — `curl_cffi` Chrome TLS 指纹模拟 + 域名级限速 + 指数退避重试

---

## 系统架构

![System Architecture](src/data/img/structure_v4.png)

### 工作流

```
                    ┌──→ 技术分析师 ──────────┐
                    ├──→ 基本面分析师 ─────────┤
 市场数据 Agent ────┼──→ 情绪分析师 ───────────┼──→ 看多研究员 ←──→ 看空研究员 ──→ 辩论室
                    ├──→ 估值分析师 ───────────┤     (5 轮条件边)        │
                    └──→ 宏观新闻 Agent ───────┘                         │
                                                                         ▼
                                                                   风险管理 Agent
                                                                         │
                                                                   宏观分析 Agent
                                                                         │
                                                                   投资组合管理 Agent → END
```

### Agent 职责

| Agent | 职责 |
|---|---|
| 市场数据 | 入口节点，获取行情/财务/新闻数据，通过 `DataSourceManager` 缓存 |
| 技术分析 | ADX 趋势、MACD 背离、KDJ 超买超卖、布林带、RSI、OBV 等 |
| 基本面分析 | ROE、净利率、营收增长、资产负债率，按行业阈值评分 |
| 情绪分析 | 四级新闻源 + 股吧情绪 + LLM 评分，含时间衰减加权 |
| 估值分析 | DCF 估值 + 所有者收益估值，WACC 动态计算 |
| 宏观新闻 | 沪深 300 指数新闻摘要（独立并行路径） |
| 看多/看空研究员 | 基于分析师报告交替辩论，生成对立论点并反驳 |
| 辩论室 | LLM 裁判评估多空观点，输出 bearish/bullish + 置信度 |
| 风险管理 | VaR、最大回撤、波动率、压力测试 |
| 宏观分析 | GDP/CPI/PMI/M2/LPR 宏观指标 + 行业新闻 |
| 投资组合管理 | `SignalWeightingEngine` 确定性加权 + LLM 定性调整，最终决策 |

---

## 快速开始

### 前置要求

- Python 3.11+
- Poetry

### 安装

```bash
git clone https://github.com/24mlight/A_Share_investment_Agent.git
cd A_Share_investment_Agent
poetry install
cp .env.example .env   # 填入 API Key
```

### 配置 LLM

编辑 `.env` 文件，二选一配置：

```env
# 方案一：OpenAI Compatible API（优先）
OPENAI_COMPATIBLE_API_KEY=your-key
OPENAI_COMPATIBLE_BASE_URL=https://your-api-endpoint.com/v1
OPENAI_COMPATIBLE_MODEL=your-model-name

# 方案二：Google Gemini
GEMINI_API_KEY=your-key
GEMINI_MODEL=gemini-1.5-flash
```

系统自动选择：若 OpenAI Compatible 三项均配置则优先使用，否则回退到 Gemini。

### 运行分析

```bash
# 基本分析
poetry run python src/main.py --ticker 002594

# 显示详细推理过程
poetry run python src/main.py --ticker 002594 --show-reasoning

# 指定新闻数量
poetry run python src/main.py --ticker 002594 --show-reasoning --num-of-news 20
```

**命令行参数：**

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--ticker` | 股票代码（必需） | — |
| `--show-reasoning` | 显示每个 Agent 的详细分析 | false |
| `--initial-capital` | 初始资金 | 100,000 |
| `--num-of-news` | 情绪分析新闻数量（最大 100） | 5 |

### 回测

```bash
poetry run python src/backtester.py --ticker 002594 --start-date 2024-12-11 --end-date 2025-01-07 --num-of-news 20
```

### API 服务模式

```bash
poetry run python run_with_backend.py
```

启动后访问 `http://localhost:8000/docs` 查看 Swagger UI。详见 [backend/README.md](./backend/README.md)。

---

## 项目结构

```
A_Share_investment_Agent/
├── backend/              # FastAPI 后端（API 路由、状态管理、日志存储）
├── src/
│   ├── agents/           # 12 个 Agent 定义 + LangGraph 工作流
│   ├── crawler/          # Playwright 搜索引擎模块（百度/Bing/Google）
│   ├── data/             # 缓存文件、新闻数据、图片
│   ├── tools/            # 数据获取、新闻爬虫、LLM 调用、HTTP 客户端、数据源管理
│   ├── utils/            # 日志、LLM 客户端、序列化、结构化输出
│   ├── backtester.py     # 回测系统
│   └── main.py           # LangGraph StateGraph 定义 + CLI 入口
├── tests/                # pytest 单元测试
├── .env.example          # 环境变量示例
├── CHANGELOG.md          # 变更日志
├── AGENTS.md              # 优化路线图
└── pyproject.toml        # Poetry 项目配置
```

---

## 数据源

### 行情与财务

| 数据类型 | 主源 | 降级链 | 缓存 TTL |
|---|---|---|---|
| 历史行情 | akshare (东方财富) | 新浪 → 腾讯 → tushare → baostock | 4h |
| 财务指标 | akshare (新浪) | 东方财富实时 → 新浪日线 | 24h |
| 财务报表 | akshare (新浪) | — | 24h |
| 北向资金 | akshare `stock_hsgt_north_net_flow_in_em` | — | 4h |
| 宏观指标 | akshare (GDP/CPI/PMI/M2/LPR) | — | 24h |

### 新闻

| 梯队 | 来源 | 方式 |
|---|---|---|
| Tier 0 | 东方财富搜索 API | 增强HTTP客户端直连 |
| Tier 1 | 东方财富 + 新浪 + 同花顺 | HTTP API |
| Tier 2 | 百度资讯 + Bing + Google | Playwright 浏览器渲染 |
| Tier 3 | akshare `stock_news_em` | HTTP API 兜底 |

去重策略：精确标题匹配 + 模糊子串匹配（去标点后比对）。

---

## 致谢

- [ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) — 本项目原始灵感来源
- [TradingAgents-CN](https://github.com/hsliuping/TradingAgents-CN) — 多源降级、防反爬、KDJ/MACD 背离、ChromaDB 记忆等思路参考

---

## 许可证

本项目使用双重许可证：

- **原始代码**（来自 ai-hedge-fund）：[MIT License](https://opensource.org/licenses/MIT)
- **修改和新增代码**（由 24mlight 创建）：GNU GPL v3 + 非商业条款 — 允许非商业使用、修改和分发，**严格禁止商业用途**

详见 `LICENSE` 文件。
