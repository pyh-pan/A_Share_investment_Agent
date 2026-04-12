import os
import time
import json
import hashlib
from google import genai
from dotenv import load_dotenv
from dataclasses import dataclass
import backoff
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON
from src.utils.llm_clients import LLMClientFactory

# 设置日志记录
logger = setup_logger('api_calls')


@dataclass
class ChatMessage:
    content: str


@dataclass
class ChatChoice:
    message: ChatMessage


@dataclass
class ChatCompletion:
    choices: list[ChatChoice]


# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(project_root, '.env')

# 加载环境变量
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)
    logger.info(f"{SUCCESS_ICON} 已加载环境变量: {env_path}")
else:
    logger.warning(f"{ERROR_ICON} 未找到环境变量文件: {env_path}")

_gemini_api_key = os.getenv("GEMINI_API_KEY")
_gemini_model = os.getenv("GEMINI_MODEL") or "gemini-1.5-flash"
_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    if not _gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    _gemini_client = genai.Client(api_key=_gemini_api_key)
    logger.info(f"{SUCCESS_ICON} Gemini 客户端初始化成功")
    return _gemini_client


@backoff.on_exception(
    backoff.expo,
    (Exception),
    max_tries=5,
    max_time=300,
    giveup=lambda e: "AFC is enabled" not in str(e)
)
def generate_content_with_retry(model, contents, config=None):
    """带重试机制的内容生成函数"""
    try:
        logger.info(f"{WAIT_ICON} 正在调用 Gemini API...")
        logger.debug(f"请求内容: {contents}")
        logger.debug(f"请求配置: {config}")

        client = _get_gemini_client()
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )

        logger.info(f"{SUCCESS_ICON} API 调用成功")
        logger.debug(f"响应内容: {response.text[:500]}...")
        return response
    except Exception as e:
        error_msg = str(e)
        if "location" in error_msg.lower():
            # 使用红色感叹号和红色文字提示
            logger.info(f"\033[91m❗ Gemini API 地理位置限制错误: 请使用美国节点VPN后重试\033[0m")
            logger.error(f"详细错误: {error_msg}")
        elif "AFC is enabled" in error_msg:
            logger.warning(f"{ERROR_ICON} 触发 API 限制，等待重试... 错误: {error_msg}")
            time.sleep(5)
        else:
            logger.error(f"{ERROR_ICON} API 调用失败: {error_msg}")
        raise e


def get_chat_completion(messages, model=None, max_retries=3, initial_retry_delay=1,
                        client_type="auto", api_key=None, base_url=None):
    """
    获取聊天完成结果，包含重试逻辑

    Args:
        messages: 消息列表，OpenAI 格式
        model: 模型名称（可选）
        max_retries: 最大重试次数
        initial_retry_delay: 初始重试延迟（秒）
        client_type: 客户端类型 ("auto", "gemini", "openai_compatible")
        api_key: API 密钥（可选，仅用于 OpenAI Compatible API）
        base_url: API 基础 URL（可选，仅用于 OpenAI Compatible API）

    Returns:
        str: 模型回答内容或 None（如果出错）
    """
    try:
        # 创建客户端
        client = LLMClientFactory.create_client(
            client_type=client_type,
            api_key=api_key,
            base_url=base_url,
            model=model
        )

        # 获取回答
        return client.get_completion(
            messages=messages,
            max_retries=max_retries,
            initial_retry_delay=initial_retry_delay
        )
    except Exception as e:
        logger.error(f"{ERROR_ICON} get_chat_completion 发生错误: {str(e)}")
        return None


# === LLM 调用缓存 ===
_LLM_CACHE_FILE = os.path.join("src", "data", "llm_cache.json")


def get_chat_completion_cached(messages, cache_ttl=86400, **kwargs):
    """带文件缓存的 LLM 调用，同一输入在 cache_ttl 秒内不重复调用

    Args:
        messages: 消息列表，OpenAI 格式
        cache_ttl: 缓存有效期（秒），默认 86400（1天）
        **kwargs: 传递给 get_chat_completion 的其他参数

    Returns:
        str: 模型回答内容或 None
    """
    os.makedirs(os.path.dirname(_LLM_CACHE_FILE), exist_ok=True)

    # 生成缓存键（消息内容的 MD5 哈希）
    msg_str = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    cache_key = hashlib.md5(msg_str.encode()).hexdigest()

    # 读取缓存
    if os.path.exists(_LLM_CACHE_FILE):
        try:
            with open(_LLM_CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            if cache_key in cache:
                entry = cache[cache_key]
                if time.time() - entry.get('timestamp', 0) < cache_ttl:
                    logger.info(f"{SUCCESS_ICON} LLM 缓存命中: {cache_key[:8]}...")
                    return entry['result']
                else:
                    logger.info(f"LLM 缓存已过期: {cache_key[:8]}...")
        except (json.JSONDecodeError, KeyError):
            pass

    # 缓存未命中，调用 LLM
    result = get_chat_completion(messages, **kwargs)

    # 写入缓存（仅当 LLM 返回有效结果时）
    if result is not None:
        try:
            cache = {}
            if os.path.exists(_LLM_CACHE_FILE):
                with open(_LLM_CACHE_FILE, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
            cache[cache_key] = {
                'result': result,
                'timestamp': time.time(),
            }
            with open(_LLM_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"写入 LLM 缓存失败: {e}")

    return result
