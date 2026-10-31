# -*- coding: utf-8 -*-
"""平台公司列表加载（可选增强，默认关闭）。

用途：界面勾选「尝试加载平台公司列表」后，尝试从 akshare 股票映射或
上证 e 互动公司列表接口获取公司清单，与预设列表合并。

约定：本模块所有函数**绝不抛异常**，任何失败都记录日志并返回空列表，
由调用方静默回退到预设列表，不阻塞主流程。
"""
from __future__ import annotations

import logging
from typing import Any

import requests

import config

logger = logging.getLogger(__name__)


def _market_from_code(code: str) -> str:
    """由股票代码推断所属市场代码（统一 SZ/SH/BJ，展示时再转中文）。"""
    if code.startswith(("6", "9")):
        return config.MARKET_SH
    if code.startswith(("0", "3")):
        return config.MARKET_SZ
    if code.startswith(("4", "8")):
        return config.MARKET_BJ
    return ""


def _load_from_akshare() -> list[dict[str, str]]:
    """通过 akshare 股票映射加载公司列表（名称 + 代码 + 市场）。

    返回:
        公司条目列表；失败返回空列表。
    """
    if not config.AKSHARE_ENABLED:
        return []
    try:
        import akshare as ak  # 延迟导入

        df = ak.stock_info_a_code_name()
        if df is None or df.empty:
            logger.warning("akshare 股票映射返回空数据")
            return []

        companies: list[dict[str, str]] = []
        for _, row in df.iterrows():
            code = str(row.get("code", "")).strip().zfill(6)
            name = str(row.get("name", "")).strip()
            if code and name:
                companies.append(
                    {"name": name, "code": code, "market": _market_from_code(code)}
                )
        logger.info("已通过 akshare 加载平台公司列表 %d 家", len(companies))
        return companies
    except Exception as exc:  # noqa: BLE001 —— 静默回退边界，明确记录
        logger.warning("通过 akshare 加载平台公司列表失败：%s", exc)
        return []


def _extract_name_code(item: dict[str, Any]) -> tuple[str, str] | None:
    """从接口返回的单个条目中尽力提取 (name, code)，取不到返回 None。"""
    name_aliases = ("name", "公司名称", "公司简称", "证券简称", "name_zh")
    code_aliases = ("code", "stock_code", "证券代码", "股票代码", "symbol")
    name = next((str(item[k]).strip() for k in name_aliases if k in item and item[k]), "")
    code = next((str(item[k]).strip().zfill(6) for k in code_aliases if k in item and item[k]), "")
    if name and code:
        return name, code
    return None


def _load_from_url() -> list[dict[str, str]]:
    """通过上证 e 互动公司列表接口加载（URL 由 config 配置，默认留空不启用）。

    返回:
        公司条目列表；失败返回空列表。
    """
    if not config.PLATFORM_COMPANY_URL:
        return []
    try:
        response = requests.get(config.PLATFORM_COMPANY_URL, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()

        # 尽力适配常见返回结构：列表 / {"data": [...]} / {"list": [...]}
        items: list[Any] = []
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            for key in ("data", "list", "items", "companies"):
                if isinstance(payload.get(key), list):
                    items = payload[key]
                    break

        companies: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            parsed = _extract_name_code(item)
            if parsed:
                name, code = parsed
                companies.append({"name": name, "code": code, "market": _market_from_code(code)})
        logger.info("已通过平台接口加载公司列表 %d 家", len(companies))
        return companies
    except Exception as exc:  # noqa: BLE001 —— 静默回退边界，明确记录
        logger.warning("通过平台接口加载公司列表失败：%s", exc)
        return []


def load_platform_companies() -> list[dict[str, str]]:
    """加载平台公司列表（akshare 映射 + 可选接口），失败静默返回空列表。

    返回:
        合并去重后的公司条目列表（name / code / market）。
    """
    companies: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in _load_from_akshare() + _load_from_url():
        key = (item["name"], item["code"])
        if key not in seen:
            seen.add(key)
            companies.append(item)
    return companies
