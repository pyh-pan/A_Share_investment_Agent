from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.news_crawler import (
    get_stock_news,
    get_news_sentiment,
    get_news_sentiment_details,
    get_forum_sentiment,
)
from src.utils.logging_config import setup_logger
from src.utils.api_utils import agent_endpoint, log_llm_interaction
import json
from datetime import datetime, timedelta
import math

# 设置日志记录
logger = setup_logger('sentiment_agent')


def _temporal_decay_weights(news_list: list, half_life_days: float = 3.0) -> list:
    now = datetime.now()
    weights = []
    for news in news_list:
        publish_time = news.get("publish_time")
        if not publish_time:
            weights.append(0.5)
            continue
        try:
            dt = datetime.strptime(publish_time, "%Y-%m-%d %H:%M:%S")
            days_ago = max((now - dt).total_seconds() / 86400.0, 0.0)
            weights.append(math.pow(0.5, days_ago / half_life_days))
        except Exception:
            weights.append(0.5)
    total = sum(weights)
    if total <= 0:
        return [1.0 / len(news_list)] * len(news_list)
    return [w / total for w in weights]


@agent_endpoint("sentiment", "情感分析师，分析市场新闻和社交媒体情绪")
def sentiment_agent(state: AgentState):
    """Responsible for sentiment analysis"""
    show_workflow_status("情绪分析师")
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    symbol = data["ticker"]
    logger.info(f"正在分析股票: {symbol}")
    # 从命令行参数获取新闻数量，默认为20条
    num_of_news = data.get("num_of_news", 20)

    # 获取 end_date 并传递给 get_stock_news
    end_date = data.get("end_date")  # 从 run_hedge_fund 传递来的 end_date

    # 获取新闻数据并分析情感，添加 date 参数
    news_list = get_stock_news(symbol, max_news=num_of_news, date=end_date)

    # 过滤7天内的新闻（只对有publish_time字段的新闻进行过滤）
    cutoff_date = datetime.now() - timedelta(days=7)
    recent_news = []
    for news in news_list:
        if 'publish_time' in news:
            try:
                news_date = datetime.strptime(
                    news['publish_time'], '%Y-%m-%d %H:%M:%S')
                if news_date > cutoff_date:
                    recent_news.append(news)
            except ValueError:
                # 如果时间格式无法解析，默认包含这条新闻
                recent_news.append(news)
        else:
            # 如果没有publish_time字段，默认包含这条新闻
            recent_news.append(news)

    if not recent_news:
        sentiment_score = 0.0
        signal = "neutral"
        confidence = "0%"
        message_content = {
            "signal": signal,
            "confidence": confidence,
            "reasoning": {
                "summary": "没有可用的近期新闻用于情绪分析。",
                "overall_score": 0.0,
                "decay_weighted_score": 0.0,
            },
            "news_count": 0,
            "status": "unavailable",
        }
        sentiment_payload = {"overall_score": 0.0, "news_scores": []}
    else:
        sentiment_score = get_news_sentiment(recent_news, num_of_news=num_of_news)
        sentiment_payload = get_news_sentiment_details(recent_news, num_of_news=num_of_news)
        forum_payload = get_forum_sentiment(symbol)

        decay_weights = _temporal_decay_weights(recent_news)
        score_map = {}
        for item in sentiment_payload.get("news_scores", []):
            idx = item.get("idx")
            score = item.get("sentiment_score", 0.0)
            if isinstance(idx, int):
                score_map[idx - 1] = float(score)
        weighted_score = 0.0
        if decay_weights:
            for i, weight in enumerate(decay_weights):
                weighted_score += weight * score_map.get(i, sentiment_score)
        weighted_score = max(-1.0, min(1.0, weighted_score))

        # 融合论坛情绪（仅在可用时占20%）
        if forum_payload.get("data_available"):
            weighted_score = weighted_score * 0.8 + float(forum_payload.get("score", 0.0)) * 0.2
            weighted_score = max(-1.0, min(1.0, weighted_score))

        # 根据情感分数生成交易信号和置信度
        if weighted_score >= 0.5:
            signal = "bullish"
            confidence = str(round(abs(weighted_score) * 100)) + "%"
        elif weighted_score <= -0.5:
            signal = "bearish"
            confidence = str(round(abs(weighted_score) * 100)) + "%"
        else:
            signal = "neutral"
            confidence = str(round((1 - abs(weighted_score)) * 100)) + "%"

        # 生成分析结果
        message_content = {
            "signal": signal,
            "confidence": confidence,
            "reasoning": {
                "summary": sentiment_payload.get("summary", ""),
                "overall_score": sentiment_score,
                "decay_weighted_score": weighted_score,
                "score_distribution": {
                    "positive": sum(1 for s in sentiment_payload.get("news_scores", []) if float(s.get("sentiment_score", 0)) > 0.2),
                    "neutral": sum(1 for s in sentiment_payload.get("news_scores", []) if -0.2 <= float(s.get("sentiment_score", 0)) <= 0.2),
                    "negative": sum(1 for s in sentiment_payload.get("news_scores", []) if float(s.get("sentiment_score", 0)) < -0.2),
                },
                "forum_sentiment": forum_payload,
            },
            "news_count": len(recent_news),
            "status": "ok",
        }

    # 如果需要显示推理过程
    if show_reasoning:
        show_agent_reasoning(message_content, "情绪分析")
        # 保存推理信息到metadata供API使用
        state["metadata"]["agent_reasoning"] = message_content

    # 创建消息
    message = HumanMessage(
        content=json.dumps(message_content),
        name="sentiment_agent",
    )

    show_workflow_status("情绪分析师", "completed")
    # logger.info(
    # f"--- DEBUG: sentiment_agent RETURN messages: {[msg.name for msg in [message]]} ---")
    return {
        "messages": [message],
        "data": {
            **data,
            "sentiment_analysis": {
                "overall_score": sentiment_score,
                "structured": sentiment_payload,
                "signal": message_content["signal"],
                "confidence": message_content["confidence"],
            },
            "sentiment_report": message_content,
        },
        "metadata": state["metadata"],
    }
