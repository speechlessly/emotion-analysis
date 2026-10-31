# -*- coding: utf-8 -*-
"""阶段 4 测试用例（可视化 visualization.py 与结论草稿 report_generator.py）。

覆盖点：
1. 空数据：词云 / 行情 / 相关性图返回提示图，不报错。
2. 只有正向词：词云正常生成。
3. 中文字体缺失：词云降级，不报错。
4. 无行情：双轴图返回提示图。
5. 无 PE：双轴图只出股价并提示。
6. R² 低：相关性散点图与结论草稿都强制带「解释力有限，不宜作因果推断」。
7. n<30：散点图不出拟合线、图注带 warning；结论带「样本量不足」。
8. 结论含数值 / 不编造数据。

运行方式（项目根目录）：
    pytest -q tests/test_phase4.py
"""
from __future__ import annotations

import matplotlib
import matplotlib.figure
import pandas as pd
import plotly.graph_objects as go

from analysis import CorrelationResult, EventWindowResult, LinearFitResult
from report_generator import generate_report
from visualization import (
    make_correlation_scatter,
    make_event_window_bar,
    make_monthly_trend,
    make_price_pe_dual_axis,
    make_wordcloud,
    make_wordcloud_grid,
)


# ---------------------------------------------------------------------------
# 词云
# ---------------------------------------------------------------------------
def test_make_wordcloud_empty() -> None:
    """空词频表返回提示图，不报错。"""
    fig = make_wordcloud(pd.DataFrame(columns=["word", "count"]), title="空词云")
    assert isinstance(fig, matplotlib.figure.Figure)


def test_make_wordcloud_only_pos_words() -> None:
    """只有正向词也能正常生成词云。"""
    word_freq = pd.DataFrame([{"word": "增长", "count": 5, "category": "pos_annual_words"}])
    fig = make_wordcloud(word_freq, title="正向词云")
    assert isinstance(fig, matplotlib.figure.Figure)


def test_make_wordcloud_grid() -> None:
    """四类词云网格：category 带 ``_words`` 后缀（与 word_frequency_from_detail 一致），
    某类无词时仍返回 Figure（不报错）。"""
    word_freq = pd.DataFrame(
        [{"word": "增长", "count": 3, "category": "pos_annual_words"}]
    )
    fig = make_wordcloud_grid(word_freq)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 4


def test_make_wordcloud_missing_font(monkeypatch) -> None:
    """中文字体缺失时词云降级为提示，不报错。"""
    monkeypatch.setattr(
        "visualization._resolve_font", lambda font_path=None: None
    )
    word_freq = pd.DataFrame([{"word": "增长", "count": 3, "category": "pos_annual_words"}])
    fig = make_wordcloud(word_freq, title="缺字体")
    assert isinstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# 月度趋势 / 事件前后柱状 / 双轴
# ---------------------------------------------------------------------------
def test_make_monthly_trend_empty() -> None:
    fig = make_monthly_trend(pd.DataFrame())
    assert isinstance(fig, go.Figure)


def test_make_monthly_trend_with_event() -> None:
    monthly = pd.DataFrame(
        {
            "month": ["2024-01", "2024-02"],
            "pos_ratio": [0.3, 0.5],
            "neg_ratio": [0.7, 0.5],
            "count": [10, 12],
        }
    )
    fig = make_monthly_trend(monthly, event_date="2024-01-15")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_make_event_window_bar() -> None:
    result = EventWindowResult(
        event_date="2024-01-15",
        window_days=90,
        valid=True,
        before_count=20,
        after_count=18,
        before_pos_mean=0.4,
        after_pos_mean=0.5,
        before_neg_mean=0.6,
        after_neg_mean=0.5,
        pos_diff=0.1,
        neg_diff=-0.1,
    )
    fig = make_event_window_bar(result)
    assert isinstance(fig, go.Figure)


def test_make_price_pe_dual_axis_empty() -> None:
    """无行情时返回提示图，不报错。"""
    fig = make_price_pe_dual_axis(pd.DataFrame(columns=["date", "close", "pe"]))
    assert isinstance(fig, go.Figure)


def test_make_price_pe_dual_axis_no_pe() -> None:
    """无 PE 时只出股价并提示，不报错。"""
    market = pd.DataFrame({"date": ["2024-01-01"], "close": [10.0]})
    fig = make_price_pe_dual_axis(market)
    assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# 相关性散点
# ---------------------------------------------------------------------------
def test_make_correlation_scatter_low_r2_caution() -> None:
    """R² 低时图注强制带「解释力有限，不宜作因果推断」。"""
    result = CorrelationResult(
        target="close", valid=True, r=0.2, p=0.2, n=40, r_squared=0.04,
        caution="解释力有限，不宜作因果推断",
    )
    fit = LinearFitResult(valid=True, n=40, slope=0.1, intercept=10.0, r_squared=0.04, fit_ok=True)
    fig = make_correlation_scatter([0.2, 0.3], [10, 11], result=result, fit=fit)
    assert isinstance(fig, go.Figure)
    annotation_text = fig.layout.annotations[0].text
    assert "解释力有限，不宜作因果推断" in annotation_text
    # 拟合线存在（fit_ok=True → 2 条 trace：样本 + 拟合线）
    assert len(fig.data) == 2


def test_make_correlation_scatter_small_n_no_fitline() -> None:
    """n<30 时散点照出、不出拟合线，图注带 warning。"""
    result = CorrelationResult(
        target="close", valid=True, r=0.8, p=0.01, n=15, r_squared=0.64,
        warning="样本量不足（n=15<30），相关性仅供参考",
    )
    fit = LinearFitResult(valid=True, n=15, slope=1.0, intercept=0.0, r_squared=0.64, fit_ok=False)
    fig = make_correlation_scatter([0.2, 0.3], [10, 11], result=result, fit=fit)
    assert isinstance(fig, go.Figure)
    # 只有样本 trace，没有拟合线
    assert len(fig.data) == 1
    annotation_text = fig.layout.annotations[0].text
    assert "样本量不足" in annotation_text


# ---------------------------------------------------------------------------
# 结论草稿
# ---------------------------------------------------------------------------
def _base_data() -> dict:
    return {
        "company": "海康威视",
        "code": "002415",
        "market": "SZ",
        "data_origin": "缓存",
        "mode": "原始模式",
        "n_comments": 201,
        "date_start": "2024-01-05",
        "date_end": "2024-12-31",
        "n_days": 45,
        "pos_ratio_mean": 0.42,
        "neg_ratio_mean": 0.58,
        "top_pos_words": ["增长", "盈利"],
        "top_neg_words": ["下滑", "风险"],
        "has_market": True,
        "has_pe": True,
    }


def test_generate_report_contains_numbers() -> None:
    """结论草稿必须含具体数值。"""
    report = generate_report(_base_data())
    assert "201" in report            # 样本量
    assert "R²=" in report or "%" in report


def test_generate_report_contains_caution() -> None:
    """R²<0.1 时结论草稿强制写「解释力有限，不宜作因果推断」。"""
    data = _base_data()
    data["corr_close"] = CorrelationResult(
        target="close", valid=True, r=0.2, p=0.2, n=40, r_squared=0.04, caution=""
    )
    report = generate_report(data)
    assert "解释力有限，不宜作因果推断" in report


def test_generate_report_small_n_warning() -> None:
    """n<30 时结论草稿含「样本量不足」。"""
    data = _base_data()
    data["corr_close"] = CorrelationResult(
        target="close", valid=True, r=0.5, p=0.05, n=15, r_squared=0.25, warning=""
    )
    report = generate_report(data)
    assert "样本量不足" in report


def test_generate_report_no_pe() -> None:
    """无 PE 时结论草稿说明「不含 PE」。"""
    data = _base_data()
    data["has_pe"] = False
    report = generate_report(data)
    assert "不含 PE" in report


def test_generate_report_no_market() -> None:
    """无行情时结论草稿如实说明，不编造相关性数值。"""
    data = _base_data()
    data["has_market"] = False
    report = generate_report(data)
    assert "未提供行情数据" in report


def test_generate_report_no_fabrication() -> None:
    """空输入时结论草稿不编造数据（应出现「未提供/未计算」）。"""
    report = generate_report({})
    assert "未提供" in report or "未计算" in report
    # 不应出现编造的相关性数值段落
    assert "Pearson r=" not in report
