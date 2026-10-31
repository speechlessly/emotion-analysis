# -*- coding: utf-8 -*-
"""规则化结论草稿生成模块（阶段 4）。

约定：
- 输入为一个汇总 dict，输出 Markdown 文本。
- 每条结论都必须带具体数值；缺失的数据**如实写「未提供 / 未计算」**，绝不编造。
- ``R² < 0.1`` 时强制写入「解释力有限，不宜作因果推断」。
- ``n < 30`` 时强制写入「样本量不足」。
- 建议部分使用「可考虑 / 建议关注」等审慎措辞。
"""
from __future__ import annotations

from typing import Any

import config
from analysis import LOW_R2_THRESHOLD, MIN_CORRELATION_N

# 与 analysis.py 保持一致的阈值（供结论措辞使用）
_MIN_N: int = MIN_CORRELATION_N
_LOW_R2: float = LOW_R2_THRESHOLD

_CAUTION_TEXT: str = "解释力有限，不宜作因果推断"


def _fmt(value: Any, digits: int = 4) -> str:
    """把数值格式化为指定小数位；None 或不可解析时返回「未计算」。"""
    if value is None:
        return "未计算"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_pct(value: Any, digits: int = 2) -> str:
    """把 0~1 占比格式化为百分数；None 返回「未计算」。"""
    if value is None:
        return "未计算"
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def _join_words(words: Any, limit: int = 8) -> str:
    """把词列表拼接为字符串；空返回「无」。"""
    if not words:
        return "无"
    items = [str(w) for w in words][:limit]
    return "、".join(items)


def _correlation_lines(title: str, result: Any) -> list[str]:
    """生成一段相关性结论（含 r/R²/p/n 数值，以及 warning/caution 强制措辞）。"""
    lines: list[str] = []
    if result is None or not getattr(result, "valid", False):
        # 修复：无效时优先展示 result.message（如「日期交集为空」的精确双方日期范围），
        # 无 message 再退回通用说明，避免含糊的「数据缺失或交集为空」。
        msg = getattr(result, "message", "") if result is not None else ""
        lines.append(f"- **{title}**：未计算" + (f"（{msg}）" if msg else "（数据缺失或交集为空）。"))
        return lines

    r = getattr(result, "r", None)
    r_squared = getattr(result, "r_squared", None)
    p = getattr(result, "p", None)
    n = getattr(result, "n", 0)

    line = (
        f"- **{title}**：Pearson r={_fmt(r)}，R²={_fmt(r_squared)}，"
        f"p={_fmt(p)}，样本量 n={n}。"
    )
    lines.append(line)

    if n < _MIN_N:
        lines.append(f"  - 样本量不足（n={n}<{_MIN_N}），相关性仅供参考。")
    # R² < 阈值时强制写 caution（不依赖 result.caution 是否已填）
    if r_squared is not None and float(r_squared) < _LOW_R2:
        lines.append(f"  - {_CAUTION_TEXT}。")
    return lines


def generate_report(data: dict[str, Any]) -> str:
    """根据汇总 dict 生成 Markdown 结论草稿。

    参数:
        data: 汇总结果 dict，常见键（缺省则对应段落如实写「未提供」）：
            company / code / market / data_origin / mode / n_comments /
            date_start / date_end / n_days / pos_ratio_mean / neg_ratio_mean /
            top_pos_words / top_neg_words / event_window / corr_close / corr_pe /
            has_market / has_pe / event_date / window_days

    返回:
        Markdown 文本。
    """
    company = str(data.get("company") or "（未指定）")
    code = str(data.get("code") or "")
    market = str(data.get("market") or "")
    market_display = config.market_display_name(market) if market else ""

    lines: list[str] = []
    lines.append("# 上市公司投资者互动文本情绪监测报告（结论草稿）")
    lines.append("")
    lines.append("> 本报告由规则模板自动生成，结论仅供课程演示参考，不构成投资建议。")
    lines.append("")

    # ------------------------------------------------------------------ 一、数据概况
    lines.append("## 一、数据概况")
    lines.append("")
    company_desc = company
    if code:
        company_desc += f"（代码 {code}"
        company_desc += f"，{market_display}" if market_display else ""
        company_desc += "）"
    lines.append(f"- 分析公司：{company_desc}")
    lines.append(f"- 数据来源：{str(data.get('data_origin') or '未提供')}")
    lines.append(f"- 分析模式：{str(data.get('mode') or '未提供')}")
    lines.append(f"- 评论样本量：{str(data.get('n_comments') or '未提供')} 条")

    date_start = data.get("date_start")
    date_end = data.get("date_end")
    n_days = data.get("n_days")
    if date_start and date_end:
        lines.append(f"- 时间跨度：{date_start} 至 {date_end}" + (f"（共 {n_days} 天）" if n_days else ""))
    else:
        lines.append("- 时间跨度：未提供")
    lines.append("")

    # ------------------------------------------------------------------ 二、情绪总体
    lines.append("## 二、情绪总体")
    lines.append("")
    lines.append(f"- 日均正向占比：{_fmt_pct(data.get('pos_ratio_mean'))}")
    lines.append(f"- 日均负向占比：{_fmt_pct(data.get('neg_ratio_mean'))}")
    lines.append(f"- 高频正向情绪词：{_join_words(data.get('top_pos_words'))}")
    lines.append(f"- 高频负向情绪词：{_join_words(data.get('top_neg_words'))}")
    lines.append("")

    # ------------------------------------------------------------------ 三、事件前后对比
    lines.append("## 三、事件前后情绪对比")
    lines.append("")
    event_window = data.get("event_window")
    if event_window is None or not getattr(event_window, "valid", False):
        lines.append("- 未提供有效事件日，未计算事件前后对比。")
    else:
        before_count = getattr(event_window, "before_count", 0)
        after_count = getattr(event_window, "after_count", 0)
        lines.append(
            f"- 事件日：{getattr(event_window, 'event_date', '未提供')}，"
            f"窗口 {getattr(event_window, 'window_days', '未提供')} 天。"
        )
        lines.append(
            f"- 事件前（n={before_count}）：正向 {_fmt_pct(getattr(event_window, 'before_pos_mean', None))}，"
            f"负向 {_fmt_pct(getattr(event_window, 'before_neg_mean', None))}。"
        )
        lines.append(
            f"- 事件后（n={after_count}）：正向 {_fmt_pct(getattr(event_window, 'after_pos_mean', None))}，"
            f"负向 {_fmt_pct(getattr(event_window, 'after_neg_mean', None))}。"
        )
        if before_count == 0 or after_count == 0:
            lines.append("- 事件前/后存在样本量为 0 的一侧，差值未参与计算。")
        else:
            pos_diff = getattr(event_window, "pos_diff", None)
            neg_diff = getattr(event_window, "neg_diff", None)
            if pos_diff is not None:
                lines.append(f"- 正向占比差值（后-前）：{_fmt_pct(pos_diff)}。")
            if neg_diff is not None:
                lines.append(f"- 负向占比差值（后-前）：{_fmt_pct(neg_diff)}。")
    lines.append("")

    # ------------------------------------------------------------------ 四、相关性
    lines.append("## 四、情绪与行情相关性")
    lines.append("")

    has_market = bool(data.get("has_market"))
    if not has_market:
        lines.append("- 未提供行情数据，未计算相关性。建议关注：如希望分析情绪与股价/PE 的关系，")
        lines.append("  可考虑提供行情数据或安装 akshare。")
    else:
        corr_close = data.get("corr_close")
        lines.extend(_correlation_lines("情绪-股价相关性", corr_close))

        has_pe = bool(data.get("has_pe"))
        if not has_pe:
            lines.append("- **情绪-PE 相关性**：当前数据源不含 PE，未计算情绪-PE 相关性。")
        else:
            lines.extend(_correlation_lines("情绪-PE 相关性", data.get("corr_pe")))
    lines.append("")

    # ------------------------------------------------------------------ 五、结论与建议
    lines.append("## 五、结论与建议")
    lines.append("")
    lines.append("- 本报告所有数值均由规则模板基于当前数据计算，未对缺失数据做任何推断或补造。")

    pos_mean = data.get("pos_ratio_mean")
    neg_mean = data.get("neg_ratio_mean")
    if pos_mean is not None and neg_mean is not None:
        if float(pos_mean) > float(neg_mean):
            lines.append(
                f"- 样本期内正向占比均值（{_fmt_pct(pos_mean)}）高于负向（{_fmt_pct(neg_mean)}），"
                "建议关注市场对该公司的整体情绪偏正面。"
            )
        elif float(pos_mean) < float(neg_mean):
            lines.append(
                f"- 样本期内负向占比均值（{_fmt_pct(neg_mean)}）高于正向（{_fmt_pct(pos_mean)}），"
                "建议关注该公司互动文本中的负面情绪来源。"
            )
        else:
            lines.append(
                f"- 样本期内正负向占比均值持平（均为 {_fmt_pct(pos_mean)}），"
                "情绪方向不明确，建议关注后续变化。"
            )

    if event_window is not None and getattr(event_window, "valid", False):
        pos_diff = getattr(event_window, "pos_diff", None)
        if pos_diff is not None:
            direction = "上升" if float(pos_diff) > 0 else "下降"
            lines.append(
                f"- 事件后正向占比较事件前{direction} {_fmt_pct(abs(pos_diff))}，"
                "建议关注事件对投资者情绪的短期影响。"
            )

    corr_close = data.get("corr_close")
    if has_market and corr_close is not None and getattr(corr_close, "valid", False):
        r_squared = getattr(corr_close, "r_squared", None)
        if r_squared is not None and float(r_squared) < _LOW_R2:
            lines.append(f"- 情绪-股价相关性的 {_CAUTION_TEXT}；可考虑结合其他指标综合分析。")
        else:
            lines.append(
                f"- 情绪-股价相关性 R²={_fmt(r_squared)}，建议关注情绪与股价的联动方向。"
            )

    lines.append("- 建议关注：如需更稳健的结论，可考虑扩充样本量、补充行情数据后再行评估。")
    lines.append("")
    return "\n".join(lines)
