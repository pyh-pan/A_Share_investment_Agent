from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.api import get_macro_indicators, get_industry_news
from src.utils.logging_config import setup_logger
from src.utils.api_utils import agent_endpoint, log_llm_interaction
import json
from src.tools.openrouter_config import get_chat_completion_with_validation

# 设置日志记录
logger = setup_logger('macro_analyst_agent')


@agent_endpoint("macro_analyst", "宏观分析师，分析宏观经济环境对目标股票的影响")
def macro_analyst_agent(state: AgentState):
    """Responsible for macro analysis"""
    show_workflow_status("宏观分析师")
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    symbol = data["ticker"]
    logger.info(f"正在进行宏观分析: {symbol}")

    industry = data.get("industry_classification", "default")
    macro_indicators = get_macro_indicators()
    industry_news = get_industry_news(industry, max_news=20)

    macro_analysis = get_macro_news_analysis(macro_indicators, industry_news, symbol, industry)
    message_content = {
        **macro_analysis,
        "macro_indicators": macro_indicators,
        "industry_news_count": len(industry_news),
        "industry": industry,
    }

    # 如果需要显示推理过程
    if show_reasoning:
        show_agent_reasoning(message_content, "宏观分析")
        # 保存推理信息到metadata供API使用
        state["metadata"]["agent_reasoning"] = message_content

    # 创建消息
    message = HumanMessage(
        content=json.dumps(message_content),
        name="macro_analyst_agent",
    )

    show_workflow_status("宏观分析师", "completed")
    # logger.info(f"--- DEBUG: macro_analyst_agent COMPLETED ---")
    # logger.info(
    # f"--- DEBUG: macro_analyst_agent RETURN messages: {[msg.name for msg in (state['messages'] + [message])]} ---")
    return {
        "messages": [message],
        "data": {
            **data,
            "macro_analysis": message_content,
            "macro_indicators": macro_indicators,
            "macro_report": message_content,
        },
        "metadata": state["metadata"],
    }


def get_macro_news_analysis(macro_indicators: dict, industry_news: list, symbol: str, industry: str) -> dict:
    """分析宏观经济新闻对股票的影响

    Args:
        news_list (list): 新闻列表

    Returns:
        dict: 宏观分析结果，包含环境评估、对股票的影响、关键因素和详细推理
    """
    available_values = [
        item.get("value")
        for item in macro_indicators.values()
        if isinstance(item, dict)
    ]
    if not any(v is not None for v in available_values) and not industry_news:
        return {
            "macro_environment": "neutral",
            "impact_on_stock": "neutral",
            "key_factors": [],
            "reasoning": "宏观指标与行业新闻均不可用，返回中性判断。",
        }

    # 检查缓存
    import os
    cache_file = "src/data/macro_analysis_cache.json"
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    prompt_version = "macro_v2_indicators_industry_news"
    news_key = json.dumps(
        {
            "version": prompt_version,
            "symbol": symbol,
            "industry": industry,
            "macro": macro_indicators,
            "news": [
                {
                    "title": n.get("title", ""),
                    "publish_time": n.get("publish_time", ""),
                }
                for n in industry_news[:20]
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
    )

    # 检查缓存
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                if news_key in cache:
                    logger.info("使用缓存的宏观分析结果")
                    return cache[news_key]
        except Exception as e:
            logger.error(f"读取宏观分析缓存出错: {e}")
            cache = {}
    else:
        logger.info("未找到宏观分析缓存文件，将创建新文件")
        cache = {}

    # 准备系统消息
    system_message = {
        "role": "system",
        "content": """你是一位专业的宏观经济分析师，专注于分析宏观环境对A股行业与个股影响。
请优先基于宏观指标做判断，行业新闻仅作补充。禁止编造未提供的数据。

输出 JSON，字段必须包含：
- macro_environment: positive/neutral/negative
- impact_on_stock: positive/neutral/negative
- key_factors: 3-5个关键因素数组
- reasoning: 详细推理（需引用提供的数据）
""",
    }

    news_content = "\n\n".join(
        [
            f"标题：{news.get('title', '')}\n"
            f"来源：{news.get('source', '未知来源')}\n"
            f"时间：{news.get('publish_time', '未知时间')}\n"
            f"内容：{(news.get('content', '') or '')[:400]}"
            for news in industry_news[:20]
        ]
    )

    user_message = {
        "role": "user",
        "content": (
            f"股票代码：{symbol}\n"
            f"所属行业：{industry}\n\n"
            f"宏观指标（主依据）：\n{json.dumps(macro_indicators, ensure_ascii=False)}\n\n"
            f"行业新闻（补充）：\n{news_content if news_content else '无'}\n\n"
            "请给出宏观环境与对目标股票影响的结论。"
        ),
    }

    try:
        # 获取LLM分析结果
        logger.info("正在调用LLM进行宏观分析...")
        result = get_chat_completion_with_validation(
            [system_message, user_message],
            data_context={
                "macro_indicators": macro_indicators,
                "industry_news_titles": [n.get("title", "") for n in industry_news[:20]],
            },
            validation_mode="warn",
        )
        if result is None:
            logger.error("LLM分析失败，无法获取宏观分析结果")
            return {
                "macro_environment": "neutral",
                "impact_on_stock": "neutral",
                "key_factors": [],
                "reasoning": "LLM分析失败，无法获取宏观分析结果"
            }

        # 解析JSON结果
        try:
            # 尝试直接解析
            analysis_result = json.loads(result.strip())
            logger.info("成功解析LLM返回的JSON结果")
        except json.JSONDecodeError:
            # 如果直接解析失败，尝试提取JSON部分
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
            if json_match:
                try:
                    analysis_result = json.loads(json_match.group(1).strip())
                    logger.info("成功从代码块中提取并解析JSON结果")
                except:
                    # 如果仍然失败，返回默认结果
                    logger.error("无法解析代码块中的JSON结果")
                    return {
                        "macro_environment": "neutral",
                        "impact_on_stock": "neutral",
                        "key_factors": [],
                        "reasoning": "无法解析LLM返回的JSON结果"
                    }
            else:
                # 如果没有找到JSON，返回默认结果
                logger.error("LLM未返回有效的JSON格式结果")
                return {
                    "macro_environment": "neutral",
                    "impact_on_stock": "neutral",
                    "key_factors": [],
                    "reasoning": "LLM未返回有效的JSON格式结果"
                }

        # 缓存结果
        cache[news_key] = analysis_result
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            logger.info("宏观分析结果已缓存")
        except Exception as e:
            logger.error(f"写入宏观分析缓存出错: {e}")

        return analysis_result

    except Exception as e:
        logger.error(f"宏观分析出错: {e}")
        return {
            "macro_environment": "neutral",
            "impact_on_stock": "neutral",
            "key_factors": [],
            "reasoning": f"分析过程中出错: {str(e)}"
        }
