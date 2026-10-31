# -*- coding: utf-8 -*-
"""阶段 3 测试用例（统计与建模 analysis.py）。

覆盖点：
1. 正常数据：r / R² / n 正确。
2. n < 30：结果带 warning（样本量不足）。
3. R² < 0.1：结果带 caution（解释力有限，不宜作因果推断）。
4. 有 close 无 pe：PE 相关性返回 None，股价相关性正常算。
5. 行情空：返回空结果（valid=False），不报错。
6. 事件窗口一侧样本为 0：标注且不参与差值计算。
7. 可选分组相关性：按事件日分前后两组，带「分组样本量可能很小」提示。
8. 辅助：按日聚合、单变量线性回归（n≥30 出拟合、n<30 只出散点）。

运行方式（在项目根目录）：
    pytest -q tests/test_phase3.py
"""
from __future__ import annotations

import pandas as pd
import pytest

from analysis import (
    aggregate_daily,
    correlate,
    event_window_compare,
    linear_fit,
    optional_grouped_correlate,
    prepare_xy,
)


# ---------------------------------------------------------------------------
# 测试数据构造
# ---------------------------------------------------------------------------
def _daily(n: int = 40) -> pd.DataFrame:
    """构造 n 天的情绪日聚合数据：pos_ratio 线性递增，neg_ratio 递减。"""
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "pos_ratio": [0.1 + 0.01 * i for i in range(n)],
            "neg_ratio": [0.9 - 0.01 * i for i in range(n)],
            "n": [3] * n,
        }
    )


def _market(n: int = 40) -> pd.DataFrame:
    """构造与 pos_ratio 完全线性正相关的行情数据（close 递增，pe 恒定）。"""
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "close": [10.0 + i for i in range(n)],
            "pe": [20.0] * n,
        }
    )


# ---------------------------------------------------------------------------
# 1. 正常数据：r / R² / n
# ---------------------------------------------------------------------------
def test_correlate_normal() -> None:
    res = correlate(_daily(40), _market(40), target="close")
    assert res is not None
    assert res.valid
    assert res.n == 40
    assert res.r == pytest.approx(1.0, abs=1e-3)
    assert res.r_squared == pytest.approx(1.0, abs=1e-3)
    assert res.p is not None


# ---------------------------------------------------------------------------
# 2. n < 30 → warning
# ---------------------------------------------------------------------------
def test_correlate_n_small_warning() -> None:
    res = correlate(_daily(10), _market(10), target="close")
    assert res is not None
    assert res.valid
    assert res.n == 10
    assert "样本量不足" in res.warning


# ---------------------------------------------------------------------------
# 3. R² < 0.1 → caution
# ---------------------------------------------------------------------------
def test_correlate_low_r2_caution() -> None:
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    # pos_ratio 交替、close 单调递增 → 两者近似无关，R² 应很小
    emotion = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "pos_ratio": [0.2, 0.8] * 20,
            "neg_ratio": [0.8, 0.2] * 20,
            "n": [3] * 40,
        }
    )
    market = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "close": [float(i) for i in range(40)],
            "pe": [20.0] * 40,
        }
    )
    res = correlate(emotion, market, target="close")
    assert res is not None
    assert res.valid
    assert res.r_squared < 0.1
    assert "解释力有限，不宜作因果推断" in res.caution


# ---------------------------------------------------------------------------
# 4. 有 close 无 pe → PE 相关性返回 None
# ---------------------------------------------------------------------------
def test_correlate_close_only_pe_none() -> None:
    emotion = _daily(30)
    market = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=30, freq="D").strftime("%Y-%m-%d"),
            "close": [10.0 + i for i in range(30)],
        }
    )
    res_close = correlate(emotion, market, target="close")
    assert res_close is not None
    assert res_close.valid

    res_pe = correlate(emotion, market, target="pe")
    assert res_pe is None  # 无 pe 列 → 返回 None，界面隐藏 PE 图


# ---------------------------------------------------------------------------
# 5. 行情空 → 返回空结果，不报错
# ---------------------------------------------------------------------------
def test_correlate_empty_market_returns_invalid() -> None:
    res = correlate(_daily(30), pd.DataFrame(columns=["date", "close", "pe"]), target="close")
    assert res is not None
    assert res.valid is False
    assert "行情数据为空" in res.message


# ---------------------------------------------------------------------------
# 6. 事件窗口一侧样本为 0
# ---------------------------------------------------------------------------
def test_event_window_one_side_zero() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-05", "2024-01-06"],  # 都在事件日之后
            "pos_ratio": [0.5, 0.6],
            "neg_ratio": [0.5, 0.4],
        }
    )
    res = event_window_compare(df, event_date="2024-01-04", window_days=90)
    assert res.valid
    assert res.before_count == 0
    assert res.after_count == 2
    assert res.pos_diff is None          # 一侧为 0，不参与差值计算
    assert res.neg_diff is None
    assert "事件前窗口样本量为 0" in res.message


def test_event_window_both_sides() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-05", "2024-01-06"],
            "pos_ratio": [0.2, 0.4, 0.6, 0.8],
            "neg_ratio": [0.8, 0.6, 0.4, 0.2],
        }
    )
    res = event_window_compare(df, event_date="2024-01-03", window_days=90)
    assert res.valid
    assert res.before_count == 2
    assert res.after_count == 2
    assert res.pos_diff == pytest.approx(0.4)   # (0.6+0.8)/2 - (0.2+0.4)/2 = 0.7-0.3
    assert res.neg_diff == pytest.approx(-0.4)


# ---------------------------------------------------------------------------
# 7. 可选分组相关性
# ---------------------------------------------------------------------------
def test_optional_grouped_correlate() -> None:
    res = optional_grouped_correlate(_daily(40), _market(40), event_date="2024-01-21", target="close")
    assert res.valid
    assert "分组样本量可能很小" in res.note
    assert res.before is not None and res.after is not None
    assert res.before["n"] > 0 and res.after["n"] > 0
    assert "r" in res.before and "r_squared" in res.before


# ---------------------------------------------------------------------------
# 8. 辅助：按日聚合 + 单变量线性回归
# ---------------------------------------------------------------------------
def test_aggregate_daily() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "pos_ratio": [0.2, 0.4, 0.6],
            "neg_ratio": [0.8, 0.6, 0.4],
        }
    )
    daily = aggregate_daily(df)
    assert list(daily.columns) == ["date", "pos_ratio", "neg_ratio", "n"]
    row = daily[daily["date"] == "2024-01-01"].iloc[0]
    assert row["n"] == 2
    assert row["pos_ratio"] == pytest.approx(0.3)


def test_linear_fit_enough_samples() -> None:
    x = list(range(40))
    y = [2 * i + 1 for i in range(40)]
    res = linear_fit(x, y)
    assert res.valid and res.fit_ok
    assert res.slope == pytest.approx(2.0)
    assert res.r_squared == pytest.approx(1.0, abs=1e-3)


def test_linear_fit_small_n_no_fit() -> None:
    x = list(range(10))
    y = [2 * i for i in range(10)]
    res = linear_fit(x, y)
    assert res.valid
    assert res.fit_ok is False
    assert "散点图" in res.message


# ---------------------------------------------------------------------------
# 9. 语义确认补测（valid 语义 / 日期统一 / 聚合口径 / prepare_xy 一致性）
# ---------------------------------------------------------------------------
def test_correlate_n15_valid_true_with_warning() -> None:
    """确认1：n=15（<30）时 valid=True，r/R²/p 照常返回，仅附加 warning。"""
    res = correlate(_daily(15), _market(15), target="close")
    assert res is not None
    assert res.valid is True
    assert res.n == 15
    assert res.r is not None
    assert res.r_squared is not None
    assert res.p is not None
    assert "样本量不足" in res.warning


def test_linear_fit_small_n_valid_true_with_stats() -> None:
    """确认1：linear_fit n<30 时 valid=True、fit_ok=False，仍返回 r/R²/p。"""
    res = linear_fit(list(range(15)), [2 * i for i in range(15)])
    assert res.valid is True
    assert res.fit_ok is False
    assert res.slope == pytest.approx(2.0)
    assert res.r is not None
    assert res.r_squared is not None
    assert res.p is not None


def test_prepare_xy_chinese_date_join() -> None:
    """确认2：情绪表中文日期 + 行情表标准日期能成功内连接（n>0）。"""
    emotion = pd.DataFrame(
        {
            "date": ["2019年02月21日 08:27", "2019年02月22日 09:00"],
            "pos_ratio": [0.5, 0.6],
            "neg_ratio": [0.5, 0.4],
            "n": [1, 1],
        }
    )
    market = pd.DataFrame(
        {
            "date": ["2019-02-21", "2019-02-22"],
            "close": [10.0, 11.0],
            "pe": [20.0, 20.0],
        }
    )
    x, y, dates = prepare_xy(emotion, market, target="close")
    assert len(x) == len(y) == len(dates) == 2
    res = correlate(emotion, market, target="close")
    assert res is not None
    assert res.valid
    assert res.n == 2


def test_aggregate_daily_simple_mean() -> None:
    """确认3：同日多条用简单平均（0.2 与 0.4 → 0.3，n=2）。"""
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "pos_ratio": [0.2, 0.4],
            "neg_ratio": [0.8, 0.6],
        }
    )
    daily = aggregate_daily(df)
    row = daily.iloc[0]
    assert row["n"] == 2
    assert row["pos_ratio"] == pytest.approx(0.3)
    assert row["neg_ratio"] == pytest.approx(0.7)


def test_correlate_n_equals_prepare_xy_lengths() -> None:
    """确认4：correlate 的 n 等于 prepare_xy 返回的 len(x) == len(y)。"""
    emotion = _daily(40)
    market = _market(40)
    res = correlate(emotion, market, target="close")
    x, y, _dates = prepare_xy(emotion, market, target="close")
    assert res is not None
    assert res.n == len(x) == len(y) == 40
