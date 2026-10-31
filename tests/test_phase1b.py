# -*- coding: utf-8 -*-
"""阶段 1b 测试用例（数据源修订）。

覆盖点：
1. 公司列表合并（含可选平台列表）与公司元数据解析（海康威视代码修正为 002415）。
2. 行情五级回退（akshare 东财→akshare 新浪→腾讯→内置 CSV→上传）。
3. 平台公司列表加载失败时静默回退（不抛异常、不阻塞）。
4. 只有 close、没有 pe 时降级（has_pe=False，界面隐藏情绪-PE相关性）。
5. 上传行情表列名映射（中英文列名 → date/close/pe）。

运行方式（在项目根目录）：
    pytest -q tests/test_phase1b.py
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest

import config
import data_source
import market_data
import platform_companies
from data_source import get_company_options, resolve_company
from market_data import (
    MARKET_COLUMNS,
    MODE_AKSHARE,
    MODE_AUTO,
    MODE_UPLOAD,
    ORIGIN_BUILTIN,
    ORIGIN_UPLOAD,
    MarketDataError,
    _map_uploaded_columns,
    _normalize_market_df,
    build_symbol,
    get_market,
)


@pytest.fixture(autouse=True)
def _reset_platform_cache():
    """每个用例前重置平台公司缓存，避免用例间串扰。"""
    data_source._platform_companies_cache = None
    yield
    data_source._platform_companies_cache = None


# ---------------------------------------------------------------------------
# 1. 公司列表与元数据解析
# ---------------------------------------------------------------------------
def test_companies_json_hikvision_code_fixed() -> None:
    """海康威视代码必须更正为 002415（深交所），而不是此前的 600690。"""
    with config.COMPANIES_JSON.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    hik = next(c for c in data["companies"] if c["name"] == "海康威视")
    assert hik["code"] == "002415"
    assert hik["market"] == config.MARKET_SZ


def test_resolve_company_preset() -> None:
    """预设公司应能解析出代码与市场。"""
    meta = resolve_company("海康威视")
    assert meta["code"] == "002415"
    assert meta["market"] == config.MARKET_SZ


def test_resolve_company_unknown() -> None:
    """手动输入且未收录的公司，code/market 应为空而非报错。"""
    meta = resolve_company("某未收录公司")
    assert meta["code"] == ""
    assert meta["market"] == ""


def test_market_prefix_symbol_building() -> None:
    """resolve_company 返回 market=SZ，且 market_data 能拼出 sz002415。"""
    meta = resolve_company("海康威视")
    assert meta["code"] == "002415"
    assert meta["market"] == config.MARKET_SZ
    assert build_symbol(meta["code"], meta["market"]) == "sz002415"


def test_build_symbol_falls_back_to_code_prefix() -> None:
    """market 为空时按代码前缀推断：600519 → sh600519，002415 → sz002415。"""
    assert build_symbol("600519") == "sh600519"
    assert build_symbol("002415") == "sz002415"


def test_market_display_name() -> None:
    """市场代码应能转换为中文展示名，空值原样返回。"""
    assert config.market_display_name(config.MARKET_SZ) == "深交所"
    assert config.market_display_name(config.MARKET_SH) == "上交所"
    assert config.market_display_name("") == ""


def test_build_symbol_conflict_warns(caplog) -> None:
    """code 与 market 前缀冲突时，以代码推断为准并记录 warning。"""
    import logging

    with caplog.at_level(logging.WARNING):
        symbol = build_symbol("600519", "SZ")
    assert symbol == "sh600519"
    assert "冲突" in caplog.text
    assert "code=600519" in caplog.text


def test_build_symbol_bj_prefix_no_error() -> None:
    """北交所代码（4/8 开头）推断为 bj 前缀，不抛异常，供上层正常降级。"""
    assert build_symbol("430047") == "bj430047"


@patch(
    "data_source._load_platform_companies",
    return_value=[{"name": "平台公司A", "code": "600000", "market": "上交所"}],
)
def test_company_options_include_platform(_mock) -> None:
    """勾选平台列表开关后，平台公司应并入下拉且预设仍在。"""
    options = get_company_options(include_platform=True)
    assert "海康威视" in options
    assert "平台公司A" in options
    assert len(options) == len(set(options))  # 去重


# ---------------------------------------------------------------------------
# 2. 平台公司列表静默回退
# ---------------------------------------------------------------------------
@patch("platform_companies.load_platform_companies", side_effect=RuntimeError("接口挂了"))
def test_platform_silent_fallback(_mock) -> None:
    """平台列表加载抛异常时，公司下拉应静默回退到预设，不崩溃。"""
    options = get_company_options(include_platform=True)
    assert "海康威视" in options
    assert "萤石网络" in options


@patch("akshare.stock_info_a_code_name", side_effect=RuntimeError("网络错误"), create=True)
def test_platform_akshare_failure_returns_empty(_mock) -> None:
    """akshare 股票映射失败时应返回空列表而非抛异常。"""
    result = platform_companies.load_platform_companies()
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 3. 行情五级回退
# ---------------------------------------------------------------------------
@patch("market_data._fetch_akshare_hist", side_effect=MarketDataError("东财不可用"))
@patch("market_data._fetch_akshare_daily", side_effect=MarketDataError("新浪不可用"))
@patch("market_data._fetch_tencent_kline", side_effect=MarketDataError("腾讯不可用"))
def test_market_fallback_to_builtin_csv(_h, _d, _t) -> None:
    """前三级失败时应降级到内置 CSV（海康威视样例，含 close 与 pe）。"""
    result = get_market("002415")
    assert result.data_origin == ORIGIN_BUILTIN
    assert not result.empty
    assert result.has_pe
    assert list(result.df.columns) == list(MARKET_COLUMNS)
    assert result.df["close"].notna().any()


@patch("market_data._fetch_akshare_hist", side_effect=MarketDataError("no"))
@patch("market_data._fetch_akshare_daily", side_effect=MarketDataError("no"))
@patch("market_data._fetch_tencent_kline", side_effect=MarketDataError("no"))
@patch("market_data._read_builtin_csv", side_effect=MarketDataError("无内置样例"))
def test_market_completely_missing_returns_empty(_h, _d, _t, _b) -> None:
    """五级全部缺失时应返回空结果并提示（缺内置样例时给出明确提示），而不是抛异常崩溃。"""
    result = get_market("999999", uploaded_file=None)
    assert result.empty
    assert "未找到该公司内置行情样例" in result.message


def test_market_upload_mode(tmp_path) -> None:
    """上传行情表应正确读取并归一化。"""
    file_path = tmp_path / "m.csv"
    file_path.write_text("date,close,pe\n2024-01-01,10.5,25.0\n", encoding="utf-8")
    result = get_market("002415", mode=MODE_UPLOAD, uploaded_file=str(file_path))
    assert result.data_origin == ORIGIN_UPLOAD
    assert result.has_pe
    assert result.df.loc[0, "close"] == 10.5


# ---------------------------------------------------------------------------
# 4. 无 PE 降级（仅 close）
# ---------------------------------------------------------------------------
def test_normalize_close_only_has_pe_false() -> None:
    """只有 close 无 pe 时，has_pe 必须为 False。"""
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "close": [10.0, 11.0]})
    out, has_pe = _normalize_market_df(df)
    assert has_pe is False
    assert list(out.columns) == list(MARKET_COLUMNS)


@patch(
    "market_data._fetch_akshare_hist",
    return_value=(pd.DataFrame({"date": ["2024-01-02"], "close": [10.0]}), False),
)
def test_close_only_downgrade_message(_mock) -> None:
    """仅收盘价来源（且本地无含 PE 样例）应在 message 中说明“不含 PE，降级为股价相关性”。"""
    result = get_market("999999", mode=MODE_AKSHARE)
    assert result.has_pe is False
    assert "不含 PE" in result.message
    assert "降级" in result.message


@patch(
    "market_data._fetch_akshare_hist",
    return_value=(pd.DataFrame({"date": ["2024-01-02"], "close": [10.0]}), False),
)
def test_network_no_pe_falls_back_to_builtin(_mock) -> None:
    """网络接口无 PE 时，应主动切换为本地含 PE 的内置 CSV 行情（避免 PE 图缺失）。"""
    result = get_market("002415", mode=MODE_AKSHARE)
    assert result.data_origin == ORIGIN_BUILTIN
    assert result.has_pe is True
    assert "pe" in result.df.columns
    assert result.df["pe"].notna().any()


# ---------------------------------------------------------------------------
# 5. 上传行情表列名映射（中英文别名）
# ---------------------------------------------------------------------------
def test_map_uploaded_columns_cn() -> None:
    """中文列名（日期/收盘/市盈率）应能映射为 date/close/pe。"""
    df = pd.DataFrame({"日期": ["2024-01-01"], "收盘": ["10.5"], "市盈率": ["25.0"]})
    out, has_pe = _map_uploaded_columns(df)
    assert list(out.columns) == list(MARKET_COLUMNS)
    assert has_pe
    assert out.loc[0, "close"] == 10.5


def test_map_uploaded_columns_missing_close_raises() -> None:
    """缺少 close 列时应抛出明确异常（说明需要哪些列）。"""
    df = pd.DataFrame({"date": ["2024-01-01"], "pe": ["25"]})
    with pytest.raises(MarketDataError):
        _map_uploaded_columns(df)
