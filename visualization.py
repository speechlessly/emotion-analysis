# -*- coding: utf-8 -*-
"""可视化模块（阶段 4）。

约定：
- 所有函数只返回 ``matplotlib`` Figure 或 ``plotly`` Figure，**不直接调用** ``st.pyplot``，
  是否渲染由上层（app.py）决定。
- 词云、柱状图等无数据 / 字体缺失时，返回「提示图」而非抛出异常。
- 中文字体路径一律经 ``config.resolve_font_path()`` 解析，支持传入显式路径覆盖。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # 无界面后端，必须在 pyplot 导入之前设置

import matplotlib.pyplot as plt

# 强制设置中文字体，避免 Windows/Linux 下词云与 matplotlib 标题显示为方块。
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import font_manager
from wordcloud import WordCloud

import config
from analysis import CorrelationResult, EventWindowResult, LinearFitResult

logger = logging.getLogger(__name__)

# 四类情绪词的展示名（词云网格用）。
# 注意：这里的 key 必须与词频表 ``word_freq`` 的 ``category`` 列取值一致。
# ``sentiment.word_frequency_from_detail`` 产出的 category 是明细表的标准列名
# ``neg_annual_words / pos_annual_words / neg_social_words / pos_social_words``
# （带 ``_words`` 后缀），而不是词典内部键 ``neg_annual`` 等，否则四格都会过滤为空。
_CATEGORY_LABELS: dict[str, str] = {
    "neg_annual_words": "年报负面",
    "pos_annual_words": "年报正面",
    "neg_social_words": "社媒负面",
    "pos_social_words": "社媒正面",
}


def _resolve_font(font_path: str | Path | None = None) -> str | None:
    """解析可用的中文字体路径。

    参数:
        font_path: 显式字体路径；存在则优先使用，不存在则记录日志并回退 config 解析。

    返回:
        存在的字体文件路径字符串；找不到返回 None（调用方据此降级）。
    """
    if font_path:
        path = Path(font_path)
        if path.exists():
            return str(path)
        logger.warning("指定字体路径不存在：%s，尝试回退 config 解析", font_path)
    resolved = config.resolve_font_path()
    return str(resolved) if resolved else None


def _freq_from_word_freq(word_freq: pd.DataFrame) -> dict[str, float]:
    """把词频表（word/count[/category]）转为 wordcloud 可用的 ``{word: count}``。"""
    freq: dict[str, float] = {}
    if word_freq is None or word_freq.empty:
        return freq
    for _, row in word_freq.iterrows():
        word = str(row.get("word", "")).strip()
        try:
            count = float(row.get("count", 0))
        except (TypeError, ValueError):
            count = 0.0
        if word and count > 0:
            freq[word] = count
    return freq


def _render_wordcloud_on_axis(
    ax: Any,
    freq: dict[str, float],
    font: str | None,
    colormap: str,
) -> None:
    """在给定坐标轴上渲染词云；无词或字体缺失时给出提示文字，不抛异常。"""
    if not freq:
        ax.text(0.5, 0.5, "暂无词汇数据，无法生成词云", ha="center", va="center", fontsize=13)
        ax.set_axis_off()
        return

    if font is None:
        # 字体缺失：仍生成词云（中文可能显示为方块），并提示用户
        wc = WordCloud(
            width=800, height=600, background_color="white", colormap=colormap
        )
        ax.text(
            0.5, 1.02, "未找到中文字体，中文可能显示异常", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, color="gray",
        )
    else:
        wc = WordCloud(
            width=800, height=600, background_color="white",
            font_path=font, colormap=colormap,
        )
    wc.generate_from_frequencies(freq)
    ax.imshow(wc, interpolation="bilinear")
    ax.set_axis_off()


def _set_title(ax: Any, title: str, font: str | None) -> None:
    """设置标题；有字体时用中文字体，避免标题乱码。"""
    if font and Path(font).exists():
        prop = font_manager.FontProperties(fname=font)
        ax.set_title(title, fontproperties=prop)
    else:
        ax.set_title(title)


def make_wordcloud(
    word_freq: pd.DataFrame,
    title: str = "",
    font_path: str | Path | None = None,
    colormap: str = "viridis",
) -> matplotlib.figure.Figure:
    """生成单张词云图（matplotlib Figure）。

    参数:
        word_freq: 词频表，含 ``word/count`` 列（可选 ``category``）。
        title: 图标题。
        font_path: 中文字体路径；为 None 时经 config 解析。
        colormap: wordcloud 配色方案。

    返回:
        matplotlib Figure。无词 / 字体缺失时返回提示图，不报错。
    """
    font = _resolve_font(font_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    freq = _freq_from_word_freq(word_freq)
    _render_wordcloud_on_axis(ax, freq, font, colormap)
    _set_title(ax, title or "情绪词云", font)
    fig.tight_layout()
    return fig


def make_wordcloud_grid(
    word_freq: pd.DataFrame,
    font_path: str | Path | None = None,
    colormap: str = "viridis",
) -> matplotlib.figure.Figure:
    """生成四类情绪词云（2×2 子图，matplotlib Figure）。

    参数:
        word_freq: 词频表，含 ``word/count/category`` 列。
        font_path: 中文字体路径。
        colormap: 配色方案。

    返回:
        matplotlib Figure；四类中任一类无词时该子图显示提示文字。
    """
    font = _resolve_font(font_path)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for ax, (category, label) in zip(axes, _CATEGORY_LABELS.items()):
        if word_freq is not None and not word_freq.empty and "category" in word_freq.columns:
            sub = word_freq[word_freq["category"] == category]
        else:
            sub = pd.DataFrame()
        freq = _freq_from_word_freq(sub)
        _render_wordcloud_on_axis(ax, freq, font, colormap)
        _set_title(ax, label, font)

    fig.tight_layout()
    return fig


def make_monthly_trend(
    monthly: pd.DataFrame,
    event_date: str | None = None,
) -> go.Figure:
    """生成月度情绪占比趋势图（plotly，双线，可标注事件日）。

    参数:
        monthly: ``aggregate_by_month`` 的输出（month/pos_ratio/neg_ratio/count）。
        event_date: 事件日（YYYY-MM-DD），可标注为虚线。

    返回:
        plotly Figure；无数据时返回带提示的空白图。
    """
    fig = go.Figure()
    if monthly is None or monthly.empty:
        fig.add_annotation(
            text="暂无月度情绪数据", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font=dict(size=14),
        )
        fig.update_layout(title="月度情绪占比趋势", xaxis_title="月份", yaxis_title="占比")
        return fig

    fig.add_trace(go.Scatter(
        x=monthly["month"], y=monthly["pos_ratio"],
        mode="lines+markers", name="正向占比",
    ))
    fig.add_trace(go.Scatter(
        x=monthly["month"], y=monthly["neg_ratio"],
        mode="lines+markers", name="负向占比",
    ))

    if event_date:
        try:
            event_month = pd.to_datetime(event_date, errors="coerce").strftime("%Y-%m")
            fig.add_vline(
                x=event_month, line_dash="dash", line_color="red",
                annotation_text="事件日", annotation_position="top left",
            )
        except (TypeError, ValueError):
            logger.warning("事件日无法解析，跳过标注：%s", event_date)

    fig.update_layout(
        title="月度情绪占比趋势",
        xaxis_title="月份",
        yaxis_title="占比",
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def make_event_window_bar(result: EventWindowResult) -> go.Figure:
    """生成事件前后情绪占比柱状图（plotly，标注样本量）。

    参数:
        result: analysis.event_window_compare 的返回值。

    返回:
        plotly Figure；结果不可用时返回带提示的空白图。
    """
    fig = go.Figure()
    if result is None or not result.valid:
        fig.add_annotation(
            text="事件前后对比不可用（事件日无法解析或无数据）",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(title="事件前后情绪对比")
        return fig

    categories = ["事件前", "事件后"]
    pos = [
        0.0 if result.before_pos_mean is None else result.before_pos_mean,
        0.0 if result.after_pos_mean is None else result.after_pos_mean,
    ]
    neg = [
        0.0 if result.before_neg_mean is None else result.before_neg_mean,
        0.0 if result.after_neg_mean is None else result.after_neg_mean,
    ]
    before_note = f"n={result.before_count}"
    after_note = f"n={result.after_count}"

    fig.add_trace(go.Bar(
        name="正向占比", x=categories, y=pos,
        text=[before_note, after_note], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="负向占比", x=categories, y=neg,
        text=[before_note, after_note], textposition="outside",
    ))
    fig.update_layout(
        title=f"事件前后情绪占比对比（窗口 {result.window_days} 天）",
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def make_price_pe_dual_axis(market: pd.DataFrame) -> go.Figure:
    """生成股价 / PE 双轴走势图（plotly）。

    参数:
        market: 行情 DataFrame（date/close/pe）。

    返回:
        plotly Figure。无 PE 时只出股价；无数据时返回带提示的空白图。
    """
    fig = go.Figure()
    if market is None or market.empty:
        fig.add_annotation(
            text="暂无行情数据", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font=dict(size=14),
        )
        fig.update_layout(title="股价 / PE 走势")
        return fig

    has_pe = "pe" in market.columns and pd.to_numeric(
        market["pe"], errors="coerce"
    ).notna().any()

    fig.add_trace(go.Scatter(
        x=market["date"], y=market["close"], mode="lines", name="收盘价",
    ))
    if has_pe:
        fig.add_trace(go.Scatter(
            x=market["date"], y=market["pe"], mode="lines", name="PE", yaxis="y2",
        ))
        fig.update_layout(
            yaxis2=dict(title="PE", overlaying="y", side="right"),
        )
    else:
        fig.add_annotation(
            text="当前数据源不含 PE，仅展示股价", xref="paper", yref="paper",
            x=0.98, y=0.02, showarrow=False, align="right", font=dict(size=10, color="gray"),
        )

    fig.update_layout(
        title="股价 / PE 走势",
        xaxis_title="日期",
        yaxis_title="收盘价",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def make_correlation_scatter(
    x: Any,
    y: Any,
    result: CorrelationResult | None = None,
    fit: LinearFitResult | None = None,
    x_label: str = "情绪正向占比",
    y_label: str = "收盘价",
    dates: Any = None,
) -> go.Figure:
    """生成相关性散点图（plotly，可带拟合线、R²、n、warning/caution 图注）。

    参数:
        x: 自变量（情绪正向占比），建议来自 analysis.prepare_xy 的对齐结果。
        y: 因变量（行情指标），与 x 等长。
        result: correlate 的返回值（提供 R²/n/warning/caution 图注）。
        fit: linear_fit 的返回值（仅当 ``fit.fit_ok=True`` 时画拟合线）。
        x_label / y_label: 坐标轴标题。
        dates: 可选日期序列，用于悬停显示。

    返回:
        plotly Figure。
    """
    xs = pd.Series(x).reset_index(drop=True).astype(float)
    ys = pd.Series(y).reset_index(drop=True).astype(float)

    hover_text = None
    if dates is not None:
        hover_text = [str(d) for d in dates]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers", name="样本",
        text=hover_text, hovertemplate="%{text}<br>x=%{x}<br>y=%{y}<extra></extra>"
        if hover_text is not None else None,
    ))

    # 拟合线：仅 n>=30（fit_ok=True）才画
    if fit is not None and fit.valid and fit.fit_ok and fit.slope is not None:
        line_x = np.linspace(xs.min(), xs.max(), 50)
        line_y = fit.slope * line_x + fit.intercept
        fig.add_trace(go.Scatter(
            x=line_x, y=line_y, mode="lines", name="拟合线",
            line=dict(color="red", dash="dash"),
        ))

    # 图注：R²、n、warning、caution（强制带上）
    annotation_parts: list[str] = []
    if result is not None:
        if result.r_squared is not None:
            annotation_parts.append(f"R²={result.r_squared:.4f}")
        annotation_parts.append(f"n={result.n}")
        if result.warning:
            annotation_parts.append(result.warning)
        if result.caution:
            annotation_parts.append(result.caution)
    if annotation_parts:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.98,
            text="<br>".join(annotation_parts), showarrow=False,
            align="left", bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray", borderwidth=1,
        )

    fig.update_layout(
        title=f"情绪占比 vs {y_label}",
        xaxis_title=x_label,
        yaxis_title=y_label,
    )
    return fig


def make_market_emotion_dual_axis(
    market: pd.DataFrame,
    emotion_daily: pd.DataFrame,
    event_date: str = "",
    target: str = "pe",
) -> go.Figure:
    """生成「行情指标 + 情绪占比」双轴图（plotly，左轴 PE/股价，右轴正/负占比）。

    对应原报告图 4.8 的三线图：左轴为市盈率（或股价），右轴为正向/负向占比。

    参数:
        market: 行情 DataFrame（date/close/pe）。
        emotion_daily: aggregate_daily 输出（date/pos_ratio/neg_ratio/n）。
        event_date: 可选事件日（虚线标注）。
        target: 左轴指标（"pe" 或 "close"）；当 target="pe" 且行情无有效 PE 时自动降级为 close。

    返回:
        plotly Figure；无情绪或行情数据时返回带提示的空白图。
    """
    fig = go.Figure()

    if emotion_daily is None or emotion_daily.empty:
        fig.add_annotation(
            text="暂无情绪数据", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font=dict(size=14),
        )
        fig.update_layout(title="行情与情绪占比走势")
        return fig

    if market is None or market.empty:
        fig.add_annotation(
            text="暂无行情数据", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font=dict(size=14),
        )
        fig.update_layout(title="行情与情绪占比走势")
        return fig

    use_target = target
    degraded_note = ""
    if target == "pe":
        has_pe = "pe" in market.columns and pd.to_numeric(
            market["pe"], errors="coerce"
        ).notna().any()
        if not has_pe:
            use_target = "close"
            degraded_note = "当前数据源不含 PE，已降级为股价"

    mkt_df = pd.DataFrame(
        {
            "_x": pd.to_datetime(market["date"], errors="coerce"),
            "_y": pd.to_numeric(market[use_target], errors="coerce"),
        }
    ).dropna()
    emo_df = pd.DataFrame(
        {
            "_x": pd.to_datetime(emotion_daily["date"], errors="coerce"),
            "_pos": pd.to_numeric(emotion_daily["pos_ratio"], errors="coerce"),
            "_neg": pd.to_numeric(emotion_daily["neg_ratio"], errors="coerce"),
        }
    ).dropna()

    left_label = "PE" if use_target == "pe" else "收盘价"
    fig.add_trace(go.Scatter(
        x=mkt_df["_x"], y=mkt_df["_y"], mode="lines", name=left_label,
    ))
    fig.add_trace(go.Scatter(
        x=emo_df["_x"], y=emo_df["_pos"], mode="lines+markers",
        name="正向占比", yaxis="y2",
    ))
    fig.add_trace(go.Scatter(
        x=emo_df["_x"], y=emo_df["_neg"], mode="lines+markers",
        name="负向占比", yaxis="y2",
    ))

    if event_date:
        try:
            event_dt = pd.to_datetime(event_date, errors="coerce")
            fig.add_vline(
                x=event_dt, line_dash="dash", line_color="red",
                annotation_text="事件日", annotation_position="top left",
            )
        except (TypeError, ValueError):
            logger.warning("事件日无法解析，跳过标注：%s", event_date)

    if degraded_note:
        fig.add_annotation(
            text=degraded_note, xref="paper", yref="paper",
            x=0.98, y=0.02, showarrow=False, align="right",
            font=dict(size=10, color="gray"),
        )

    fig.update_layout(
        title=f"{left_label}与情绪占比走势",
        xaxis_title="日期",
        yaxis_title=left_label,
        yaxis2=dict(title="占比", overlaying="y", side="right", range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig
