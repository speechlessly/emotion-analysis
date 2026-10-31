# -*- coding: utf-8 -*-
"""在线爬虫模块（可选在线模式）。

针对上证 e 互动平台（sns.sseinfo.com）的全文本问答搜索接口实现分页爬取。
本模块只负责“抓取并解析成统一 schema”，不负责写缓存、不负责回退逻辑（回退由 data_source 层处理）。

合规与健壮性约定：
- 无 Cookie 不爬（由调用方在进入本模块前判断，本模块对空 Cookie 直接抛异常）。
- 每页限速（sleep delay）、带重试与超时、逐页校验 HTTP 状态码。
- 空页提前终止；结果去重。
- Cookie 永不写入日志、永不出现在返回值与导出文件中。
"""
from __future__ import annotations

import logging
import time
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup

import config

logger = logging.getLogger(__name__)


class CrawlerError(Exception):
    """爬虫相关异常，携带可读的中文原因，供上层回退逻辑使用。"""


def _parse_cookie(cookie_str: str) -> dict[str, str]:
    """把形如 ``k1=v1; k2=v2`` 的 Cookie 字符串解析为字典。

    参数:
        cookie_str: 用户填写的 Cookie 字符串。

    返回:
        键值对字典。空项会被忽略。
    """
    cookies: dict[str, str] = {}
    for part in cookie_str.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        cookies[key.strip()] = value.strip()
    return cookies


def _build_headers(company: str) -> dict[str, str]:
    """构造请求头，其中 Referer 的关键词与目标公司一一对应。

    参数:
        company: 公司关键词（用于拼入 Referer 的 keyword 参数）。

    返回:
        请求头字典。
    """
    return {
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Origin": config.BASE_URL,
        "Referer": f"{config.SEARCH_URL}?keyword={quote(company)}",
        "User-Agent": config.USER_AGENT,
        "X-Requested-With": "XMLHttpRequest",
    }


def _post_with_retry(
    session: requests.Session,
    url: str,
    data: dict[str, Any],
    headers: dict[str, str],
    cookies: dict[str, str],
    retries: int,
    timeout: int,
    backoff: float,
) -> requests.Response:
    """带重试与指数退避的 POST 请求。

    参数:
        session: 复用连接与 Cookie 的会话对象。
        url: 目标接口地址。
        data: 表单数据。
        headers: 请求头。
        cookies: Cookie 字典。
        retries: 最大重试次数。
        timeout: 单次请求超时（秒）。
        backoff: 退避系数（指数）。

    返回:
        成功的响应对象。

    异常:
        CrawlerError: 重试耗尽后仍失败。
    """
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.post(
                url, data=data, headers=headers, cookies=cookies, timeout=timeout
            )
            return response
        except requests.RequestException as exc:  # noqa: PERF203
            last_error = exc
            wait = backoff ** attempt
            logger.warning(
                "请求失败（第 %d/%d 次）：%s，%.1f 秒后重试", attempt, retries, exc, wait
            )
            time.sleep(wait)

    raise CrawlerError(f"请求连续失败 {retries} 次，最后错误：{last_error}") from last_error


def _first_text(item: BeautifulSoup, selector: str) -> str:
    """安全地提取某 CSS 选择器下的第一个文本片段，失败返回空串而非抛异常。

    参数:
        item: 单个 feed 条目节点。
        selector: CSS 选择器。

    返回:
        去除首尾空白后的文本；找不到或解析失败时返回空字符串。
    """
    try:
        nodes = item.select(selector)
        if not nodes:
            return ""
        return nodes[0].get_text(strip=True)
    except Exception as exc:  # noqa: BLE001 —— 明确记录，禁止裸 pass
        logger.debug("解析选择器 %s 失败：%s", selector, exc)
        return ""


def crawl_comments(
    company: str,
    page_start: int = 1,
    page_end: int = 1,
    cookie_str: str | None = None,
    delay: float | None = None,
    code: str = "",
    market: str = "",
) -> pd.DataFrame:
    """爬取指定公司的投资者互动问答评论，返回统一 schema 的 DataFrame。

    参数:
        company: 公司关键词（同时用于搜索与 Referer）。
        page_start: 起始页码（含，1 起）。
        page_end: 结束页码（含）。
        cookie_str: 用户填写的 Cookie 字符串；为空时直接抛异常，由上层回退缓存。
        delay: 每页限速间隔（秒）；为 None 时使用配置默认值。
        code: 股票代码（可选，仅用于日志标识，不参与请求与导出）。
        market: 市场（可选，仅用于日志标识）。

    返回:
        列固定为 ``["company", "date", "question", "answer"]`` 的 DataFrame。
        单页无数据会提前终止；最终结果按 (date, question, answer) 去重。

    异常:
        CrawlerError: 无 Cookie、页码非法、状态码异常或重试耗尽。
    """
    # 1. 前置校验：无 Cookie 不爬（合规红线，绝不打印/记录 Cookie 内容）
    if not cookie_str or not cookie_str.strip():
        raise CrawlerError("未提供 Cookie，无法在线爬取（请在界面填写 Cookie 或改用本地缓存/上传）。")

    if page_start < 1 or page_end < page_start:
        raise CrawlerError(f"页码参数非法：page_start={page_start}, page_end={page_end}")

    delay = config.REQUEST_DELAY if delay is None else delay
    cookies = _parse_cookie(cookie_str)
    headers = _build_headers(company)

    rows: list[dict[str, str]] = []
    session = requests.Session()

    try:
        for page in range(page_start, page_end + 1):
            form_data = {
                "page": page,
                "keyword": company,
                "sdate": "",
                "edate": "",
            }
            logger.info(
                "开始爬取第 %d 页（公司=%s%s%s）",
                page,
                company,
                f"，代码={code}" if code else "",
                f"，市场={market}" if market else "",
            )

            response = _post_with_retry(
                session=session,
                url=config.POST_URL,
                data=form_data,
                headers=headers,
                cookies=cookies,
                retries=config.REQUEST_RETRIES,
                timeout=config.REQUEST_TIMEOUT,
                backoff=config.REQUEST_BACKOFF,
            )

            # 2. 逐页校验状态码
            if response.status_code != 200:
                raise CrawlerError(
                    f"第 {page} 页返回异常状态码 {response.status_code}，请检查 Cookie 是否有效。"
                )

            soup = BeautifulSoup(response.text, "lxml")
            items = soup.select(".m_feed_item")

            # 3. 空页提前终止
            if not items:
                logger.info("第 %d 页无 .m_feed_item 数据，提前终止爬取", page)
                break

            for item in items:
                question = _first_text(item, ".m_feed_detail .m_feed_txt")
                date = _first_text(item, ".m_feed_from span")
                answer = _first_text(item, ".m_feed_detail.m_qa .m_feed_txt")
                # 时间与问答全空视为无效条目，跳过
                if not date and not question and not answer:
                    continue
                rows.append(
                    {
                        "company": company,
                        "date": date,
                        "question": question,
                        "answer": answer,
                    }
                )

            # 4. 限速：每页之间必须停顿，避免高频访问
            if page < page_end:
                time.sleep(delay)
    finally:
        session.close()

    df = pd.DataFrame(rows, columns=list(config.COMMENT_COLUMNS))
    if not df.empty:
        df = df.drop_duplicates(subset=["date", "question", "answer"]).reset_index(drop=True)
    logger.info("爬取完成：共获取 %d 条（去重后）", len(df))
    return df
