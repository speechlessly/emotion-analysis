# -*- coding: utf-8 -*-
"""阶段 1 测试用例。

覆盖点：
1. 无 Cookie 在线爬取 → 自动回退本地缓存。
2. Cookie 错误（爬取失败）→ 自动回退本地缓存。
3. 上传文件读取。
4. 公司下拉列表合并（预设 + 缓存 + 上传）。
5. 爬虫空 Cookie 直接抛异常、Cookie 解析等边界。

运行方式（在项目根目录）：
    pytest -q tests/test_phase1.py
"""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

import data_source
from crawler import CrawlerError, _parse_cookie, crawl_comments
from data_source import (
    MODE_CACHE,
    MODE_ONLINE,
    MODE_UPLOAD,
    ORIGIN_CACHE,
    ORIGIN_UPLOAD,
    _safe_filename,
    get_company_options,
    get_data,
)


# ---------------------------------------------------------------------------
# 用例 1：无 Cookie → 回退缓存
# ---------------------------------------------------------------------------
def test_no_cookie_falls_back_to_cache(monkeypatch) -> None:
    """在线模式下未填 Cookie，应跳过爬取并回退到本地缓存。"""
    monkeypatch.setattr(
        data_source,
        "_read_cache",
        lambda company: pd.DataFrame({"date": ["2024-01-01"], "question": ["q"], "answer": ["a"]}),
    )
    result = get_data("海康威视", MODE_ONLINE, cookie_str=None)
    assert result.data_origin == ORIGIN_CACHE
    assert not result.df.empty
    assert "回退" in result.message


# ---------------------------------------------------------------------------
# 用例 2：Cookie 错误（爬取抛异常）→ 回退缓存
# ---------------------------------------------------------------------------
@patch("data_source.crawl_comments", side_effect=CrawlerError("Cookie 已失效"))
def test_bad_cookie_falls_back_to_cache(_mock_crawl, monkeypatch) -> None:
    """爬取失败时应记录原因并回退缓存，而不是抛出异常中断。"""
    monkeypatch.setattr(
        data_source,
        "_read_cache",
        lambda company: pd.DataFrame({"date": ["2024-01-01"], "question": ["q"], "answer": ["a"]}),
    )
    result = get_data("海康威视", MODE_ONLINE, cookie_str="invalid=cookie")
    assert result.data_origin == ORIGIN_CACHE
    assert not result.df.empty
    assert "回退" in result.message


# ---------------------------------------------------------------------------
# 用例 3：上传文件读取（Excel / CSV）
# ---------------------------------------------------------------------------
def test_upload_file_read_excel(tmp_path) -> None:
    """上传模式应正确读取 Excel 并标记来源为「上传」。"""
    source_df = pd.DataFrame(
        {
            "company": ["测试公司"],
            "date": ["2024-01-05 10:30"],
            "question": ["公司进展如何？"],
            "answer": ["感谢关注。"],
        }
    )
    file_path = tmp_path / "up.xlsx"
    source_df.to_excel(file_path, index=False)

    result = get_data("任意公司", MODE_UPLOAD, uploaded_file=str(file_path))
    assert result.data_origin == ORIGIN_UPLOAD
    assert len(result.df) == 1
    assert result.df.loc[0, "company"] == "测试公司"


def test_upload_file_read_csv(tmp_path) -> None:
    """上传模式应正确读取 CSV。"""
    file_path = tmp_path / "up.csv"
    file_path.write_text("company,date,question,answer\nA,2024-01-01,q,a\n", encoding="utf-8")

    result = get_data("任意公司", MODE_UPLOAD, uploaded_file=str(file_path))
    assert result.data_origin == ORIGIN_UPLOAD
    assert len(result.df) == 1


def test_upload_missing_file_raises() -> None:
    """未提供上传文件时应抛出明确异常。"""
    with pytest.raises(data_source.DataSourceError):
        get_data("任意公司", MODE_UPLOAD, uploaded_file=None)


# ---------------------------------------------------------------------------
# 用例 4：公司列表合并
# ---------------------------------------------------------------------------
def test_company_options_merge() -> None:
    """预设公司 + 缓存文件名 + 上传 company 列应合并且去重保序。"""
    options = get_company_options(extra_companies=["测试公司", "海康威视"])

    assert "海康威视" in options       # 来自预设 + 缓存
    assert "萤石网络" in options       # 来自预设
    assert "测试公司" in options       # 来自上传
    # 去重：不能出现重复项
    assert len(options) == len(set(options))
    # 顺序：预设公司在前，上传公司在后
    assert options.index("海康威视") < options.index("测试公司")


# ---------------------------------------------------------------------------
# 边界用例：爬虫本身
# ---------------------------------------------------------------------------
def test_crawl_empty_cookie_raises() -> None:
    """爬虫入口对空 Cookie 必须显式抛异常，交由上层回退。"""
    with pytest.raises(CrawlerError):
        crawl_comments("海康威视", 1, 1, cookie_str=None)


def test_crawl_invalid_page_range_raises() -> None:
    """非法页码必须抛异常。"""
    with pytest.raises(CrawlerError):
        crawl_comments("海康威视", 3, 1, cookie_str="a=b")


def test_parse_cookie() -> None:
    """Cookie 字符串解析为字典，空项被忽略。"""
    parsed = _parse_cookie("a=1; b=2; ; c=3=4")
    assert parsed == {"a": "1", "b": "2", "c": "3=4"}


def test_safe_filename() -> None:
    """公司名转安全文件名，去除非法字符。"""
    assert _safe_filename("*ST 公司") == "*ST 公司".replace("*", "")
    assert _safe_filename(" A/B:C ") == "ABC"
