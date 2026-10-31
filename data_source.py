# -*- coding: utf-8 -*-
"""数据获取统一入口。

负责三种数据来源的分发与回退：
- 在线爬取：先爬，成功写缓存；失败自动回退本地缓存。
- 本地缓存：直接读取 data/cache/{company}.xlsx。
- 上传文件：读取用户上传的 Excel / CSV。

同时提供「公司下拉列表合并」逻辑：预设公司 + 缓存文件名 +（可选）平台公司列表 + 上传数据中的 company 列，
以及「公司名 → 代码/市场」的元数据解析。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

import config
from cleaner import read_excel_smart
from crawler import CrawlerError, crawl_comments

logger = logging.getLogger(__name__)

# 数据来源标识（供界面显示）
ORIGIN_ONLINE = "在线"
ORIGIN_CACHE = "缓存"
ORIGIN_UPLOAD = "上传"
ORIGIN_DEMO = "演示"

# 数据来源模式（界面选项）
MODE_ONLINE = "在线爬取"
MODE_CACHE = "本地缓存"
MODE_UPLOAD = "上传文件"
MODE_DEMO = "演示模式"

# 文件名中不允许出现的字符（Windows 兼容）
_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]+')


class DataSourceError(Exception):
    """数据获取异常，携带可读的中文原因。"""


@dataclass
class DataResult:
    """一次数据获取的结果。

    属性:
        df: 统一 schema 的评论数据。
        data_origin: 数据来源（在线 / 缓存 / 上传）。
        code: 股票代码（可能为空，供行情模块使用）。
        market: 市场代码（SZ/SH/BJ，可能为空）。
        message: 供界面展示的说明文字（如回退原因）。
    """

    df: pd.DataFrame
    data_origin: str
    code: str = ""
    market: str = ""
    message: str = ""


# 平台公司列表的内存缓存（惰性加载一次，避免重复请求）
_platform_companies_cache: list[dict[str, str]] | None = None


def _safe_filename(company: str) -> str:
    """把公司名转为安全的文件名（去除非法字符、去首尾空白）。"""
    cleaned = _INVALID_FILENAME_CHARS.sub("", company).strip()
    return cleaned or "unknown"


def _cache_path(company: str) -> Path:
    """返回指定公司的缓存文件路径。"""
    return config.CACHE_DIR / f"{_safe_filename(company)}.xlsx"


def _read_cache(company: str) -> pd.DataFrame:
    """读取本地缓存文件（智能选择能被识别出数据模式的 sheet）。

    缓存 Excel 可能含多个 sheet，有效数据未必在第一个 sheet（列名也可能为中文），
    因此复用 ``cleaner.read_excel_smart`` 智能读取，而不是直接 ``pd.read_excel`` 默认读第一个 sheet。
    """
    path = _cache_path(company)
    if not path.exists():
        raise DataSourceError(f"本地缓存不存在：{path}，请先在线爬取或上传数据。")

    try:
        df, sheet_name = read_excel_smart(path)
        logger.info(
            "已读取本地缓存 %s（实际使用 sheet「%s」），共 %d 行",
            path, sheet_name, len(df),
        )
        return df
    except Exception as exc:  # noqa: BLE001 —— 明确记录并抛出
        raise DataSourceError(f"读取缓存文件失败：{path}，原因：{exc}") from exc


def _demo_path(company: str) -> Path:
    """返回演示数据文件路径（data/demo，与爬取缓存 data/cache 分离）。"""
    return config.DEMO_DIR / f"{_safe_filename(company)}.xlsx"


def _read_demo(company: str) -> pd.DataFrame:
    """读取演示数据（已分析的四类情绪词表，含占比）。

    演示数据位于 ``data/demo/``，与在线爬取落盘的 ``data/cache/`` 相互独立，
    避免爬取结果覆盖演示样例。读取时同样用 ``read_excel_smart`` 智能选择有效 sheet。
    """
    path = _demo_path(company)
    if not path.exists():
        raise DataSourceError(
            f"演示数据不存在：{path}，请将已分析的四类情绪词表放入 data/demo/ 目录。"
        )

    try:
        df, sheet_name = read_excel_smart(path)
        logger.info(
            "已读取演示数据 %s（实际使用 sheet「%s」），共 %d 行",
            path, sheet_name, len(df),
        )
        return df
    except Exception as exc:  # noqa: BLE001 —— 明确记录并抛出
        raise DataSourceError(f"读取演示数据失败：{path}，原因：{exc}") from exc


def _write_cache(df: pd.DataFrame, company: str) -> Path:
    """把在线爬取结果写入本地缓存。"""
    path = _cache_path(company)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False, engine="openpyxl")
    logger.info("已写入缓存 %s，共 %d 行", path, len(df))
    return path


def _read_upload(uploaded_file: Any) -> pd.DataFrame:
    """读取用户上传的文件（Excel 或 CSV）。

    Excel 会自动选择一个能被识别出数据模式的 sheet（跳过无法识别的汇总 sheet）。
    """
    if uploaded_file is None:
        raise DataSourceError("未提供上传文件。")

    if isinstance(uploaded_file, (str, Path)):
        name = str(uploaded_file)
        source: Any = uploaded_file
    else:
        name = getattr(uploaded_file, "name", "")
        source = uploaded_file

    try:
        sheet_name = ""
        if name.lower().endswith((".csv", ".tsv")):
            df = pd.read_csv(source, encoding="utf-8-sig")
        else:
            df, sheet_name = read_excel_smart(source)
        if sheet_name:
            logger.info(
                "已读取上传文件 %s 的 sheet「%s」，共 %d 行",
                name or "<未知>", sheet_name, len(df),
            )
        else:
            logger.info("已读取上传文件 %s，共 %d 行", name or "<未知>", len(df))
        return df
    except Exception as exc:  # noqa: BLE001
        raise DataSourceError(f"读取上传文件失败：{name or '<未知>'}, 原因：{exc}") from exc


def _load_preset_companies() -> list[dict[str, str]]:
    """读取 companies.json 中的预设公司列表。"""
    path = config.COMPANIES_JSON
    if not path.exists():
        logger.warning("预设公司文件不存在：%s", path)
        return []
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return list(data.get("companies", []))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("读取预设公司列表失败：%s", exc)
        return []


def _load_platform_companies() -> list[dict[str, str]]:
    """惰性加载平台公司列表（可选增强，失败返回空列表，绝不阻塞）。

    返回:
        平台公司条目列表（含 name / code / market）。
    """
    global _platform_companies_cache
    if _platform_companies_cache is not None:
        return _platform_companies_cache
    try:
        # 延迟导入：platform_companies 为可选增强，避免影响主流程启动
        from platform_companies import load_platform_companies

        _platform_companies_cache = load_platform_companies()
    except Exception as exc:  # noqa: BLE001 —— 静默回退边界，明确记录
        logger.warning("加载平台公司列表失败：%s", exc)
        _platform_companies_cache = []
    return _platform_companies_cache


def get_company_options(
    extra_companies: list[str] | None = None,
    include_platform: bool = False,
) -> list[str]:
    """合并生成公司下拉列表。

    参数:
        extra_companies: 额外公司名列表（通常来自上传数据的 company 列 unique 值）。
        include_platform: 是否尝试并入平台公司列表（对应界面可选开关）。

    返回:
        去重保序的公司名列表。
    """
    options: list[str] = []

    # 1) 预设公司（companies.json）
    for item in _load_preset_companies():
        name = item.get("name", "")
        if name:
            options.append(name)

    # 2) data/demo 中的演示数据文件名（去掉扩展名，优先展示）
    if config.DEMO_DIR.exists():
        for demo_file in sorted(config.DEMO_DIR.glob("*.xlsx")):
            options.append(demo_file.stem)

    # 3) data/cache 中已有的缓存文件名（去掉扩展名）
    if config.CACHE_DIR.exists():
        for cache_file in sorted(config.CACHE_DIR.glob("*.xlsx")):
            options.append(cache_file.stem)

    # 4) 平台公司列表（可选，失败静默回退，不阻塞）
    if include_platform:
        for item in _load_platform_companies():
            name = item.get("name", "")
            if name:
                options.append(name)

    # 5) 上传数据中的 company 列 unique 值
    for name in extra_companies or []:
        if name:
            options.append(str(name))

    # 去重且保持顺序
    seen: set[str] = set()
    merged: list[str] = []
    for name in options:
        if name not in seen:
            seen.add(name)
            merged.append(name)
    return merged


def resolve_company(name: str) -> dict[str, str]:
    """根据公司名解析代码与市场（优先预设，其次已加载的平台列表）。

    参数:
        name: 公司名称。

    返回:
        形如 ``{"name", "code", "market", "event_date"}`` 的字典；
        无法解析时 code / market / event_date 均为空字符串。

    说明:
        平台列表仅在「已被加载」（用户勾选过开关）后才参与匹配，避免手动输入
        未收录公司时触发一次额外的网络请求。
    """
    for item in _load_preset_companies():
        if item.get("name") == name:
            return {
                "name": name,
                "code": str(item.get("code", "")),
                "market": str(item.get("market", "")),
                "event_date": str(item.get("event_date", "")),
            }
    if _platform_companies_cache is not None:
        for item in _platform_companies_cache:
            if item.get("name") == name:
                return {
                    "name": name,
                    "code": str(item.get("code", "")),
                    "market": str(item.get("market", "")),
                    "event_date": str(item.get("event_date", "")),
                }
    return {"name": name, "code": "", "market": "", "event_date": ""}


def _fallback_to_cache(company: str, reason: str, code: str, market: str) -> DataResult:
    """在线爬取失败时的回退逻辑。"""
    try:
        df = _read_cache(company)
    except DataSourceError:
        raise DataSourceError(
            f"{reason}；且本地缓存也不存在，请改用「上传文件」提供数据。"
        ) from None
    return DataResult(
        df=df,
        data_origin=ORIGIN_CACHE,
        code=code,
        market=market,
        message=f"{reason}；已自动回退到本地缓存（{len(df)} 条）。",
    )


def get_data(
    company: str,
    source_mode: str,
    uploaded_file: Any = None,
    cookie_str: str | None = None,
    page_range: tuple[int, int] | None = None,
    code: str = "",
    market: str = "",
) -> DataResult:
    """按来源模式获取评论数据，返回统一结果。

    参数:
        company: 公司名称 / 关键词。
        source_mode: 来源模式（在线爬取 / 本地缓存 / 上传文件）。
        uploaded_file: 上传的文件对象（仅上传模式使用）。
        cookie_str: 用户填写的 Cookie（仅在线模式使用）。
        page_range: (起始页, 结束页) 元组，默认 (1, 1)。
        code: 股票代码（可选，由界面在选择公司后传入，透传给爬虫/行情模块）。
        market: 市场（可选，同上）。

    返回:
        DataResult：包含 df、data_origin、code、market、message。

    异常:
        DataSourceError: 数据源不可用（如缓存不存在且无回退路径）。
        ValueError: 来源模式非法。
    """
    page_start, page_end = page_range or (1, 1)

    # ---- 演示模式：读取 data/demo 中已分析的四类情绪词表 ----
    if source_mode == MODE_DEMO:
        df = _read_demo(company)
        return DataResult(
            df=df,
            data_origin=ORIGIN_DEMO,
            code=code,
            market=market,
            message=f"已读取演示数据（四类情绪词表），共 {len(df)} 条。",
        )

    # ---- 在线爬取：先爬，失败回退缓存 ----
    if source_mode == MODE_ONLINE:
        if not cookie_str or not cookie_str.strip():
            return _fallback_to_cache(company, "未填写 Cookie，跳过在线爬取", code, market)

        try:
            df = crawl_comments(
                company=company,
                page_start=page_start,
                page_end=page_end,
                cookie_str=cookie_str,
                delay=config.REQUEST_DELAY,
                code=code,
                market=market,
            )
            if df.empty:
                raise DataSourceError("在线爬取返回空结果（可能关键词无结果或页面结构变化）")
            _write_cache(df, company)
            return DataResult(
                df=df,
                data_origin=ORIGIN_ONLINE,
                code=code,
                market=market,
                message=f"在线爬取成功，共 {len(df)} 条，已同步写入本地缓存。",
            )
        except (CrawlerError, DataSourceError, requests.RequestException) as exc:
            logger.warning("在线爬取失败，回退缓存：%s", exc)
            return _fallback_to_cache(company, f"在线爬取失败：{exc}", code, market)
        except Exception as exc:  # noqa: BLE001 —— 回退边界，明确记录
            logger.exception("在线爬取发生未预期异常，回退缓存")
            return _fallback_to_cache(company, f"在线爬取异常：{exc}", code, market)

    # ---- 本地缓存 ----
    if source_mode == MODE_CACHE:
        df = _read_cache(company)
        return DataResult(
            df=df,
            data_origin=ORIGIN_CACHE,
            code=code,
            market=market,
            message=f"已读取本地缓存，共 {len(df)} 条。",
        )

    # ---- 上传文件 ----
    if source_mode == MODE_UPLOAD:
        df = _read_upload(uploaded_file)
        return DataResult(
            df=df,
            data_origin=ORIGIN_UPLOAD,
            code=code,
            market=market,
            message=f"已读取上传数据，共 {len(df)} 条。",
        )

    raise ValueError(f"未知的数据来源模式：{source_mode}")
