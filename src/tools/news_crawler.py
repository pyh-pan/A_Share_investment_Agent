import os
import sys
import json
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import pandas as pd
from urllib.parse import urlparse
from src.tools.openrouter_config import get_chat_completion, logger as api_logger
from src.tools.http_client import smart_get, smart_post

# 导入新的搜索模块
try:
    from src.crawler.search import google_search_sync, bing_search_sync, baidu_search_sync, SearchOptions
except ImportError:
    print("警告: 无法导入新的搜索模块，将回退到 akshare")
    google_search_sync = None
    bing_search_sync = None
    baidu_search_sync = None
    SearchOptions = None

# 保留 akshare 作为备用
try:
    import akshare as ak
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("警告: akshare 不可用")
    ak = None


def build_search_query(symbol: str, date: str = None) -> str:
    """
    构建针对股票新闻的 Google 搜索查询

    Args:
        symbol: 股票代码，如 "300059"
        date: 截止日期，格式 "YYYY-MM-DD"

    Returns:
        构建好的搜索查询字符串
    """
    # 基础查询：股票代码 + 新闻关键词
    base_query = f"{symbol} 股票 新闻 财经"

    # 添加时间限制（搜索指定日期之前的新闻）
    if date:
        try:
            # 解析日期并计算一周前的日期作为开始时间
            end_date = datetime.strptime(date, "%Y-%m-%d")
            start_date = end_date - timedelta(days=7)  # 搜索过去一周的新闻

            # Google 搜索时间语法：after:YYYY-MM-DD before:YYYY-MM-DD
            base_query += f" after:{start_date.strftime('%Y-%m-%d')} before:{date}"
        except ValueError:
            print(f"日期格式错误: {date}，忽略时间限制")

    # 限制新闻网站 - 只选择主要的财经网站
    news_sites = [
        "site:sina.com.cn",
        "site:163.com",
        "site:eastmoney.com",
        "site:cnstock.com",
        "site:hexun.com"
    ]

    # 添加网站限制
    query = f"{base_query} ({' OR '.join(news_sites)})"

    return query


def extract_domain(url: str) -> str:
    """从 URL 提取域名作为新闻来源"""
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except:
        return "未知来源"


def convert_search_results_to_news_format(search_results, symbol: str) -> list:
    """
    将搜索结果转换为现有新闻格式

    Args:
        search_results: Google 搜索结果
        symbol: 股票代码

    Returns:
        符合现有格式的新闻列表
    """
    news_list = []

    for result in search_results:
        # 过滤掉明显不相关的结果
        if any(keyword in result.title.lower() for keyword in ['招聘', '求职', '广告', '登录', '注册']):
            continue

        # 尝试从snippet中提取时间信息
        publish_time = None
        if result.snippet:
            # 查找常见的时间模式
            import re
            time_patterns = [
                r'(\d{1,2}天前)',
                r'(\d{1,2}小时前)',
                r'(\d{4}-\d{2}-\d{2})',
                r'(\d{4}年\d{1,2}月\d{1,2}日)',
                r'(\d{2}-\d{2})'
            ]

            for pattern in time_patterns:
                match = re.search(pattern, result.snippet)
                if match:
                    time_str = match.group(1)
                    try:
                        # 处理相对时间
                        if '天前' in time_str:
                            days = int(time_str.replace('天前', ''))
                            publish_date = datetime.now() - timedelta(days=days)
                            publish_time = publish_date.strftime(
                                '%Y-%m-%d %H:%M:%S')
                        elif '小时前' in time_str:
                            hours = int(time_str.replace('小时前', ''))
                            publish_date = datetime.now() - timedelta(hours=hours)
                            publish_time = publish_date.strftime(
                                '%Y-%m-%d %H:%M:%S')
                        # YYYY-MM-DD格式
                        elif '-' in time_str and len(time_str) == 10:
                            publish_time = f"{time_str} 00:00:00"
                        break
                    except:
                        continue

        news_item = {
            "title": result.title,
            "content": result.snippet or result.title,
            "source": extract_domain(result.link),
            "url": result.link,
            "keyword": symbol,
            "search_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 搜索时间
        }

        # 只有当能提取到发布时间时才添加，否则不包含这个字段
        if publish_time:
            news_item["publish_time"] = publish_time

        news_list.append(news_item)

    return news_list


def _fetch_news_from_eastmoney(symbol: str, max_news: int = 100) -> list:
    """直接从东方财富搜索API获取新闻（修复akshare的HTTP/HTTPS问题）"""
    import json as js
    import re

    url = "https://search-api-web.eastmoney.com/search/jsonp"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Referer": "https://so.eastmoney.com/",
    }
    cb = "jQuery3510875346244069884_1668256937995"
    params = {
        "cb": cb,
        "param": js.dumps({
            "uid": "",
            "keyword": symbol,
            "type": ["cmsArticleWebOld"],
            "client": "web",
            "clientType": "web",
            "clientVersion": "curr",
            "param": {
                "cmsArticleWebOld": {
                    "searchScope": "default",
                    "sort": "default",
                    "pageIndex": 1,
                    "pageSize": max_news,
                    "preTag": "<em>",
                    "postTag": "</em>",
                }
            },
        }, ensure_ascii=False),
    }
    try:
        r = smart_get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        text = r.text.strip()
        if not text:
            return []
        json_str = text[len(cb) + 1 : -1]  # strip JSONP wrapper
        data = js.loads(json_str)
        articles = data.get("result", {}).get("cmsArticleWebOld", [])
        news_list = []
        for a in articles:
            title = re.sub(r"</?em>", "", a.get("title", "")).strip()
            content = re.sub(r"</?em>", "", a.get("content", "")).strip()
            if not title:
                continue
            news_list.append({
                "title": title,
                "content": content or title,
                "publish_time": a.get("date", ""),
                "source": a.get("mediaName", "东方财富").strip(),
                "url": a.get("url", "").strip(),
                "keyword": symbol,
            })
        return news_list[:max_news]
    except Exception as e:
        print(f"东方财富搜索API获取新闻失败: {e}")
        return []


def _fetch_news_from_eastmoney_direct(symbol: str, max_news: int = 50) -> list:
    """Direct Eastmoney news API via smart_post (bypasses akshare, uses rate-limited HTTP client)."""
    try:
        url = "https://search-api-web.eastmoney.com/search/jsonp"
        params = {
            "cb": "jQuery",
            "param": json.dumps({
                "uid": "",
                "keyword": symbol,
                "type": ["cmsArticleWebOld"],
                "client": "web",
                "clientType": "web",
                "clientVersion": "curr",
                "param": {
                    "cmsArticleWebOld": {
                        "searchScope": "default",
                        "sort": "default",
                        "pageIndex": 1,
                        "pageSize": max_news,
                        "preTag": "",
                        "postTag": "",
                    }
                },
            }, ensure_ascii=False),
        }
        response = smart_get(url, params=params, timeout=10)
        response.raise_for_status()

        text = response.text.strip()
        if not text:
            return []

        # Robust JSONP parsing: find outermost parentheses
        json_str = text[text.index("(") + 1 : text.rindex(")")]
        data = json.loads(json_str)
        raw = data.get("result", {}).get("cmsArticleWebOld", [])
        # API versions differ: sometimes a list directly, sometimes {list: [...]}
        if isinstance(raw, dict):
            articles = raw.get("list", [])
        else:
            articles = raw

        import re
        news_items = []
        for article in articles[:max_news]:
            title = re.sub(r"</?em>", "", article.get("title", "")).strip()
            if not title:
                continue
            news_items.append({
                "title": title,
                "content": re.sub(r"</?em>", "", article.get("content", ""))[:400],
                "publish_time": article.get("date", ""),
                "source": article.get("mediaName", "eastmoney_direct").strip() or "eastmoney_direct",
                "url": article.get("url", "").strip(),
                "keyword": symbol,
            })
        print(f"东方财富直连API获取到 {len(news_items)} 条新闻")
        return news_items
    except Exception as e:
        print(f"东方财富直连API失败: {e}")
        return []


def _fetch_news_from_sina(symbol: str, max_news: int = 50) -> list:
    """从新浪财经个股新闻页面获取新闻"""
    import requests as req
    import re

    # 确定交易所前缀：6 开头为上交所(sh)，其他为深交所(sz)
    prefix = "sh" if symbol.startswith("6") else "sz"
    url = f"https://vip.stock.finance.sina.com.cn/corp/go.php/vCB_AllNewsStock/symbol/{prefix}{symbol}.phtml"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://finance.sina.com.cn/",
    }

    try:
        r = req.get(url, headers=headers, timeout=15)
        r.raise_for_status()

        if not r.text or len(r.text) < 500:
            print(f"新浪个股新闻页面返回内容过少")
            return []

        soup = BeautifulSoup(r.text, 'html.parser')
        news_list = []

        # 新浪个股新闻页面的链接通常包含 "doc-" 模式
        links = soup.select('a[href*="doc-"]')
        if not links:
            links = soup.select('.datalist a') or soup.select('ul.list01 a')

        for link_tag in links:
            if len(news_list) >= max_news:
                break

            title = link_tag.get_text(strip=True)
            link = link_tag.get('href', '')

            if not title or len(title) < 5:
                continue
            if not link:
                continue
            if link.startswith('//'):
                link = 'https:' + link
            elif not link.startswith('http'):
                continue

            # 过滤无关内容（广告、配资等）
            if any(kw in title for kw in ['招聘', '求职', '广告', '登录', '注册', '配资平台', '配资推荐', '配资排行']):
                continue

            # 从 <a> 前面的文本节点提取时间
            # 页面结构：时间文本 <a>标题</a> <br/>
            publish_time = ""
            prev = link_tag.previous_sibling
            if prev and isinstance(prev, str):
                time_match = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})', prev)
                if time_match:
                    publish_time = f"{time_match.group(1)}:00"

            news_list.append({
                "title": title,
                "content": title,
                "publish_time": publish_time,
                "source": "新浪财经",
                "url": link,
                "keyword": symbol,
            })

        print(f"新浪财经个股新闻获取到 {len(news_list)} 条")
        return news_list[:max_news]
    except Exception as e:
        print(f"新浪财经获取新闻失败: {e}")
        return []


def _fetch_news_from_10jqka(symbol: str, max_news: int = 50) -> list:
    """从同花顺 F10 页面获取个股新闻"""
    import requests as req
    import re

    url = f"https://basic.10jqka.com.cn/{symbol}/news.html"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://basic.10jqka.com.cn/",
    }

    try:
        r = req.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        r.encoding = 'gbk'
        text = r.text

        if not text or len(text) < 500:
            print("同花顺页面返回内容过少")
            return []

        # 新闻数据在 id="linkagedata" 的隐藏元素中，是一个 JSON 数组
        match = re.search(r'id="linkagedata"[^>]*>\s*(\[.*?\])\s*<', text, re.DOTALL)
        if not match:
            print("同花顺页面未找到新闻数据")
            return []

        import json as js
        raw_json = match.group(1)
        articles = js.loads(raw_json)

        news_list = []
        for a in articles:
            if len(news_list) >= max_news:
                break

            title = a.get("title", "").strip()
            if not title or len(title) < 5:
                continue

            # 过滤广告和配资内容
            if any(kw in title for kw in ['配资平台', '配资推荐', '配资排行', '招聘', '求职', '广告']):
                continue

            # ctime 是 Unix 时间戳，转换为标准格式
            publish_time = ""
            ctime = a.get("ctime")
            if ctime:
                try:
                    publish_time = datetime.fromtimestamp(int(ctime)).strftime('%Y-%m-%d %H:%M:%S')
                except (ValueError, OSError):
                    pass

            news_list.append({
                "title": title,
                "content": title,  # F10 页面只有标题，无正文摘要
                "publish_time": publish_time,
                "source": a.get("source", "同花顺").strip() or "同花顺",
                "url": a.get("curl", "").replace("\\/", "/"),
                "keyword": symbol,
            })

        print(f"同花顺个股新闻获取到 {len(news_list)} 条")
        return news_list[:max_news]
    except Exception as e:
        print(f"同花顺获取新闻失败: {e}")
        return []


def _search_news_via_bing(symbol: str, max_news: int = 20, date: str = None) -> list:
    """通过 Bing 搜索获取新闻（使用 Playwright 渲染）"""
    if not bing_search_sync or not SearchOptions:
        print("Bing Playwright 搜索不可用，跳过")
        return []

    query = f"{symbol} 股票 新闻 财经"
    try:
        options = SearchOptions(limit=max_news * 2, timeout=15000, locale="zh-CN")
        response = bing_search_sync(query, options)
        if not response or not response.results:
            print("Bing Playwright 搜索无结果")
            return []

        news_list = convert_search_results_to_news_format(response.results, symbol)
        print(f"Bing Playwright 搜索获取到 {len(news_list)} 条新闻")
        return news_list[:max_news]
    except Exception as e:
        print(f"Bing Playwright 搜索失败: {e}")
        return []


def _search_news_via_baidu(symbol: str, max_news: int = 20, date: str = None) -> list:
    """通过百度资讯搜索获取新闻（使用 Playwright 渲染）"""
    if not baidu_search_sync or not SearchOptions:
        print("百度 Playwright 搜索不可用，跳过")
        return []

    query = f"{symbol} 股票 新闻"
    try:
        options = SearchOptions(limit=max_news * 2, timeout=15000, locale="zh-CN")
        response = baidu_search_sync(query, options)
        if not response or not response.results:
            print("百度 Playwright 搜索无结果")
            return []

        news_list = convert_search_results_to_news_format(response.results, symbol)
        print(f"百度 Playwright 搜索获取到 {len(news_list)} 条新闻")
        return news_list[:max_news]
    except Exception as e:
        print(f"百度 Playwright 搜索失败: {e}")
        return []


def merge_and_deduplicate(*news_lists) -> list:
    """合并多个新闻列表并去重（精确匹配 + 相似度匹配）"""
    import re as _re
    seen_titles = set()
    normalized_titles = []  # 去除标点后的标题，用于模糊匹配
    combined = []
    for news_list in news_lists:
        if not news_list:
            continue
        for item in news_list:
            title = item.get('title', '').strip()
            if not title:
                continue
            # 精确匹配
            if title in seen_titles:
                continue
            # 模糊匹配：去除标点和空格后比较子串
            normalized = _re.sub(r'[^\w]', '', title)
            if not normalized:
                continue
            is_dup = False
            for existing in normalized_titles:
                shorter = min(len(normalized), len(existing))
                if shorter > 0 and (normalized in existing or existing in normalized):
                    is_dup = True
                    break
            if is_dup:
                continue
            seen_titles.add(title)
            normalized_titles.append(normalized)
            combined.append(item)
    # 按发布时间倒序排列
    try:
        combined.sort(key=lambda x: x.get('publish_time', ''), reverse=True)
    except:
        pass
    return combined


def get_stock_news_via_akshare(symbol: str, max_news: int = 10) -> list:
    """使用 akshare 获取股票新闻的原始方法"""
    # 先尝试直接调用东方财富API（修复HTTPS问题）
    direct_news = _fetch_news_from_eastmoney(symbol, max_news)
    if direct_news:
        print(f"通过东方财富搜索API成功获取到{len(direct_news)}条新闻")
        return direct_news

    if ak is None:
        return []

    try:
        # 获取新闻列表
        news_df = ak.stock_news_em(symbol=symbol)
        if news_df is None or len(news_df) == 0:
            print(f"未获取到{symbol}的新闻数据")
            return []

        print(f"成功获取到{len(news_df)}条新闻")

        # 实际可获取的新闻数量
        available_news_count = len(news_df)
        if available_news_count < max_news:
            print(f"警告：实际可获取的新闻数量({available_news_count})少于请求的数量({max_news})")
            max_news = available_news_count

        # 获取指定条数的新闻（考虑到可能有些新闻内容为空，多获取50%）
        news_list = []
        for _, row in news_df.head(int(max_news * 1.5)).iterrows():
            try:
                # 获取新闻内容
                content = row["新闻内容"] if "新闻内容" in row and not pd.isna(
                    row["新闻内容"]) else ""
                if not content:
                    content = row["新闻标题"]

                # 只去除首尾空白字符
                content = content.strip()
                if len(content) < 10:  # 内容太短的跳过
                    continue

                # 获取关键词
                keyword = row["关键词"] if "关键词" in row and not pd.isna(
                    row["关键词"]) else ""

                # 添加新闻
                news_item = {
                    "title": row["新闻标题"].strip(),
                    "content": content,
                    "publish_time": row["发布时间"],
                    "source": row["文章来源"].strip(),
                    "url": row["新闻链接"].strip(),
                    "keyword": keyword.strip()
                }
                news_list.append(news_item)
                print(f"成功添加新闻: {news_item['title']}")

            except Exception as e:
                print(f"处理单条新闻时出错: {e}")
                continue

        # 按发布时间排序
        news_list.sort(key=lambda x: x["publish_time"], reverse=True)

        # 只保留指定条数的有效新闻
        return news_list[:max_news]

    except Exception as e:
        print(f"akshare 获取新闻数据时出错: {e}")
        return []


def get_stock_news(symbol: str, max_news: int = 10, date: str = None) -> list:
    """获取并处理个股新闻

    Args:
        symbol (str): 股票代码，如 "300059"
        max_news (int, optional): 获取的新闻条数，默认为10条。最大支持100条。
        date (str, optional): 截止日期，格式 "YYYY-MM-DD"，用于限制获取新闻的时间范围，
                             获取该日期及之前的新闻。如果不指定，则使用当前日期。

    Returns:
        list: 新闻列表，每条新闻包含标题、内容、发布时间等信息。
              新闻来源通过智能搜索引擎获取，包含各大财经网站的相关报道。
    """

    # 限制最大新闻条数
    max_news = min(max_news, 100)

    # 获取当前日期或使用指定日期
    cache_date = date if date else datetime.now().strftime("%Y-%m-%d")

    # 构建新闻文件路径
    news_dir = os.path.join("src", "data", "stock_news")
    print(f"新闻保存目录: {news_dir}")

    # 确保目录存在
    try:
        os.makedirs(news_dir, exist_ok=True)
        print(f"成功创建或确认目录存在: {news_dir}")
    except Exception as e:
        print(f"创建目录失败: {e}")
        return []

    # 缓存文件名包含日期信息
    news_file = os.path.join(news_dir, f"{symbol}_news_{cache_date}.json")
    print(f"新闻文件路径: {news_file}")

    # 检查缓存是否存在且有效
    cached_news = []
    cache_valid = False

    if os.path.exists(news_file):
        try:
            # 检查缓存文件的修改时间（时效性检查）
            file_mtime = os.path.getmtime(news_file)
            current_time = time.time()
            # 缓存有效期：当天的缓存在当天有效，历史日期的缓存始终有效
            if date:  # 如果指定了历史日期，缓存始终有效
                cache_valid = True
            else:  # 如果是当天数据，检查是否在同一天创建
                cache_date_obj = datetime.fromtimestamp(file_mtime).date()
                today = datetime.now().date()
                cache_valid = cache_date_obj == today

            if cache_valid:
                with open(news_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    cached_news = data.get("news", [])

                    if len(cached_news) >= max_news:
                        print(
                            f"使用缓存的新闻数据: {news_file} (缓存数量: {len(cached_news)})")
                        return cached_news[:max_news]
                    else:
                        print(
                            f"缓存的新闻数量({len(cached_news)})不足，需要获取更多新闻({max_news}条)")
            else:
                print(f"缓存文件已过期，将重新获取新闻")

        except Exception as e:
            print(f"读取缓存文件失败: {e}")
            cached_news = []

    print(f'开始获取{symbol}的新闻数据...')

    # === 第零梯队：东方财富直连API（最高优先级，使用增强HTTP客户端） ===
    print("第零梯队：东方财富直连API...")
    em_direct_news = _fetch_news_from_eastmoney_direct(symbol, max_news)

    # === 第一梯队：专业财经三源（东方财富 + 新浪 + 同花顺） ===
    print("第一梯队：尝试东方财富 + 新浪财经 + 同花顺...")
    eastmoney_news = _fetch_news_from_eastmoney(symbol, max_news)
    sina_news = _fetch_news_from_sina(symbol, max_news)
    ths_news = _fetch_news_from_10jqka(symbol, max_news)
    new_news_list = merge_and_deduplicate(em_direct_news, eastmoney_news, sina_news, ths_news)
    fetch_method = "eastmoney_direct+eastmoney+sina+10jqka"

    if len(new_news_list) >= max_news * 0.5:
        print(f"第一梯队获取到 {len(new_news_list)} 条新闻，满足需求")
    else:
        # === 第二梯队：搜索引擎补充（百度 + Bing + Google，均使用 Playwright） ===
        print(f"第一梯队仅获取到 {len(new_news_list)} 条，尝试第二梯队...")

        # 2a. 百度资讯搜索（Playwright 渲染）
        baidu_news = _search_news_via_baidu(symbol, max_news, date)
        if baidu_news:
            new_news_list = merge_and_deduplicate(new_news_list, baidu_news)
            fetch_method += "+baidu"
            print(f"百度补充后共 {len(new_news_list)} 条新闻")

        # 2b. Bing 搜索（Playwright 渲染）
        bing_news = _search_news_via_bing(symbol, max_news, date)
        if bing_news:
            new_news_list = merge_and_deduplicate(new_news_list, bing_news)
            fetch_method += "+bing"
            print(f"Bing 补充后共 {len(new_news_list)} 条新闻")

        # 2c. Google 搜索（可能被墙，保留但降级，超时缩短到 8 秒）
        if len(new_news_list) < max_news * 0.3 and google_search_sync and SearchOptions:
            try:
                print("尝试 Google 搜索补充...")
                search_query = build_search_query(symbol, date)
                search_options = SearchOptions(
                    limit=max_news * 2,
                    timeout=8000,
                    locale="zh-CN"
                )
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        google_search_sync, search_query, search_options)
                    search_response = future.result(timeout=10)
                if search_response.results:
                    google_news = convert_search_results_to_news_format(
                        search_response.results, symbol)
                    new_news_list = merge_and_deduplicate(new_news_list, google_news)
                    fetch_method += "+google"
                    print(f"Google 补充后共 {len(new_news_list)} 条新闻")
            except (FuturesTimeoutError, Exception) as e:
                print(f"Google 搜索失败（预期内）: {e}")

        # === 第三梯队：akshare 兜底 ===
        if not new_news_list:
            print("前两梯队均无结果，使用 akshare 兜底...")
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        get_stock_news_via_akshare, symbol, max_news)
                    new_news_list = future.result(timeout=12)
                fetch_method = "akshare"
            except FuturesTimeoutError:
                print("akshare 获取新闻超时")
                new_news_list = []

    # 合并缓存和新获取的新闻
    combined_news = merge_and_deduplicate(cached_news, new_news_list) if cached_news else new_news_list
    if cached_news and new_news_list:
        print(f"合并缓存({len(cached_news)}条)和新获取新闻，总计{len(combined_news)}条")

    # 只保留指定条数的新闻
    final_news_list = combined_news[:max_news]

    # 保存到文件
    if new_news_list or not cache_valid:
        try:
            save_data = {
                "date": cache_date,
                "method": fetch_method,
                "news": combined_news,
                "cached_count": len(cached_news),
                "new_count": len(new_news_list),
                "total_count": len(combined_news),
                "last_updated": datetime.now().isoformat()
            }
            with open(news_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            print(f"成功保存{len(combined_news)}条新闻到文件: {news_file}")
        except Exception as e:
            print(f"保存新闻数据到文件时出错: {e}")

    return final_news_list


def get_news_sentiment_details(news_list: list, num_of_news: int = 5) -> dict:
    """Analyze sentiment with structured output and prompt-version cache key."""
    if not news_list:
        return {
            "overall_score": 0.0,
            "news_scores": [],
            "summary": "无可用新闻",
        }

    prompt_version = "sentiment_v2_structured"
    cache_file = "src/data/sentiment_cache.json"
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    news_key = "|".join(
        [
            f"{prompt_version}|{news.get('title', '')}|{news.get('content', '')[:100]}|{news.get('publish_time', '')}"
            for news in news_list[:num_of_news]
        ]
    )

    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            if news_key in cache and isinstance(cache[news_key], dict):
                return cache[news_key]
        except Exception:
            cache = {}
    else:
        cache = {}

    news_content = "\n\n".join([
        f"[{idx + 1}] 标题：{news.get('title', '')}\n"
        f"来源：{news.get('source', '未知来源')}\n"
        f"时间：{news.get('publish_time', '未知时间')}\n"
        f"内容：{(news.get('content', '') or '')[:500]}"
        for idx, news in enumerate(news_list[:num_of_news])
    ])

    system_message = {
        "role": "system",
        "content": (
            "你是A股情绪分析师。请对每条新闻输出 sentiment_score(-1到1)、importance(1到3)、"
            "impact_scope(company/industry/market)，并给出 overall_score(-1到1)。"
            "仅输出JSON，格式："
            "{\"overall_score\": float, \"news_scores\": [{\"idx\": int, \"sentiment_score\": float, \"importance\": int, \"impact_scope\": str, \"reason\": str}], \"summary\": str}"
        )
    }
    user_message = {
        "role": "user",
        "content": f"请分析如下新闻：\n\n{news_content}",
    }

    try:
        result = get_chat_completion([system_message, user_message])
        if result is None:
            return {"overall_score": 0.0, "news_scores": [], "summary": "LLM返回空"}
        parsed = None
        try:
            parsed = json.loads(result.strip())
        except Exception:
            import re

            match = re.search(r"```json\s*(.*?)\s*```", result, re.DOTALL)
            if match:
                parsed = json.loads(match.group(1))
        if not isinstance(parsed, dict):
            return {"overall_score": 0.0, "news_scores": [], "summary": "解析失败"}

        parsed["overall_score"] = float(max(-1.0, min(1.0, float(parsed.get("overall_score", 0.0)))))
        if not isinstance(parsed.get("news_scores"), list):
            parsed["news_scores"] = []

        cache[news_key] = parsed
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        return parsed
    except Exception:
        return {"overall_score": 0.0, "news_scores": [], "summary": "分析失败"}


def get_forum_sentiment(symbol: str) -> dict:
    """Best-effort forum sentiment from Eastmoney guba pages with fallback."""
    url = f"https://guba.eastmoney.com/list,{symbol}.html"
    try:
        resp = smart_get(url, timeout=8)
        text = resp.text if hasattr(resp, "text") else ""
        if not text:
            return {"score": 0.0, "data_available": False, "source": "guba"}
        positive_words = ["利好", "上涨", "增持", "买入", "反弹"]
        negative_words = ["利空", "下跌", "减持", "卖出", "暴雷"]
        pos = sum(text.count(x) for x in positive_words)
        neg = sum(text.count(x) for x in negative_words)
        total = pos + neg
        if total == 0:
            return {"score": 0.0, "data_available": False, "source": "guba"}
        score = (pos - neg) / total
        return {"score": max(-1.0, min(1.0, score)), "data_available": True, "source": "guba"}
    except Exception:
        return {"score": 0.0, "data_available": False, "source": "guba"}


def get_news_sentiment(news_list: list, num_of_news: int = 5) -> float:
    """分析新闻情感得分

    Args:
        news_list (list): 新闻列表
        num_of_news (int): 用于分析的新闻数量，默认为5条

    Returns:
        float: 情感得分，范围[-1, 1]，-1最消极，1最积极
    """
    if not news_list:
        return 0.0

    details = get_news_sentiment_details(news_list, num_of_news=num_of_news)
    try:
        return float(details.get("overall_score", 0.0))
    except Exception:
        return 0.0
