# -*- coding: utf-8 -*-
"""统计与建模模块（阶段 3）。

统计口径（必须遵守）：
- 相关性输入：情绪按日聚合（date, pos_ratio, neg_ratio, n）× 行情（date, close, pe）
  按 date 内连接；交集为空返回空结果 + 提示，不报错。
- 相关性用 ``scipy.stats.pearsonr``（不手算），输出 Pearson r、p 值、n、R²=r²。
- n < 30 → 结果加 ``warning``；R² < 0.1 → 结果加 ``caution``（图注与结论都强制带）。
- 行情有 close 无 pe → PE 相关性返回 None（界面隐藏 PE 图），股价相关性正常算。
- 线性回归只做单变量（情绪占比 → 股价 / 情绪占比 → PE），n ≥ 30 才出拟合线与 R²。
- 不做：多元回归、格兰杰因果、滞后相关（课程项目不必要且易被质疑）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
from scipy import stats

from cleaner import parse_datetime

logger = logging.getLogger(__name__)

# 相关性最小样本量阈值（n < 该值时给出 warning）
MIN_CORRELATION_N: int = 30
# R² 低于该值时给出 caution（"解释力有限，不宜作因果推断"）
LOW_R2_THRESHOLD: float = 0.1


def _normalize_date_series(series: pd.Series) -> pd.Series:
    """把日期列解析并归一化到“日”（支持中文日期），失败位置为 NaT，保持原索引。"""
    values = [parse_datetime(value) for value in series]
    result = pd.Series(values, index=series.index)
    return result.dt.normalize()


@dataclass
class CorrelationResult:
    """一次相关性计算的结果。

    字段:
        target: 目标指标（"close" 或 "pe"）。
        valid: 是否成功计算。
        r: Pearson 相关系数。
        p: p 值。
        n: 内连接后的样本量。
        r_squared: 决定系数 R² = r²。
        warning: 样本量不足等提示（n<30）。
        caution: 解释力不足提示（R²<0.1）。
        message: 供界面展示的说明（失败时给出原因）。
    """

    target: str
    valid: bool
    r: float | None = None
    p: float | None = None
    n: int = 0
    r_squared: float | None = None
    warning: str = ""
    caution: str = ""
    message: str = ""


@dataclass
class EventWindowResult:
    """事件前后对比结果。

    字段:
        event_date: 事件日。
        window_days: 窗口天数。
        valid: 事件日是否可解析。
        before_count / after_count: 前/后窗口样本量。
        before_pos_mean / after_pos_mean / before_neg_mean / after_neg_mean: 前后窗口正负占比均值。
        pos_diff / neg_diff: 后-前差值（仅两侧样本量均 >0 时计算，否则为 None）。
        message: 说明（如某侧样本量为 0）。
    """

    event_date: str
    window_days: int
    valid: bool
    before_count: int = 0
    after_count: int = 0
    before_pos_mean: float | None = None
    after_pos_mean: float | None = None
    before_neg_mean: float | None = None
    after_neg_mean: float | None = None
    pos_diff: float | None = None
    neg_diff: float | None = None
    message: str = ""


@dataclass
class LinearFitResult:
    """单变量线性回归结果。

    字段:
        valid: 是否成功计算（有效样本 >=2）。
        n: 有效样本量。
        slope / intercept: 斜率与截距。
        r / r_squared / p: 相关系数、决定系数、p 值。
        fit_ok: 是否可出拟合线与回归结论（n >= 30）。
        message: 说明（n<30 时提示仅出散点图）。
    """

    valid: bool
    n: int = 0
    slope: float | None = None
    intercept: float | None = None
    r: float | None = None
    r_squared: float | None = None
    p: float | None = None
    fit_ok: bool = False
    message: str = ""


@dataclass
class GroupedCorrelationResult:
    """可选分组相关性结果（事件前/事件后两组）。

    字段:
        event_date: 事件日。
        valid: 是否成功执行。
        before / after: 两组各自的相关性结果（dict，含 r/r_squared/p/n/warning）。
        note: 固定提示「分组样本量可能很小，结果仅供参考」。
    """

    event_date: str
    valid: bool
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    note: str = "分组样本量可能很小，结果仅供参考"


# ---------------------------------------------------------------------------
# 1. 情绪按日聚合
# ---------------------------------------------------------------------------
def aggregate_daily(
    df: pd.DataFrame,
    date_col: str = "date",
    pos_col: str = "pos_ratio",
    neg_col: str = "neg_ratio",
) -> pd.DataFrame:
    """把情绪数据按日聚合，返回 ``date, pos_ratio, neg_ratio, n``。

    参数:
        df: 含日期列与正负占比列的数据。
        date_col: 日期列名。
        pos_col / neg_col: 正负占比列名。

    返回:
        按 date 聚合的 DataFrame（pos_ratio/neg_ratio 为日均，n 为当日样本数）。
        输入为空时返回空表（含上述四列）。
    """
    columns = ["date", "pos_ratio", "neg_ratio", "n"]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    work = df.copy()
    # 统一归一化到“日”粒度（去掉时分秒），中文日期与标准日期均可解析
    work["_dt"] = _normalize_date_series(work[date_col])
    work = work.dropna(subset=["_dt"])
    work["date"] = work["_dt"].dt.strftime("%Y-%m-%d")

    pos = (
        pd.to_numeric(work[pos_col], errors="coerce")
        if pos_col in work.columns
        else pd.Series([0.0] * len(work))
    )
    neg = (
        pd.to_numeric(work[neg_col], errors="coerce")
        if neg_col in work.columns
        else pd.Series([0.0] * len(work))
    )

    agg = pd.DataFrame(
        {
            "date": work["date"],
            "pos_ratio": pos.fillna(0.0),
            "neg_ratio": neg.fillna(0.0),
        }
    )
    grouped = (
        agg.groupby("date")
        .agg(
            pos_ratio=("pos_ratio", "mean"),
            neg_ratio=("neg_ratio", "mean"),
            n=("pos_ratio", "size"),
        )
        .reset_index()
    )
    return grouped


# ---------------------------------------------------------------------------
# 2. 相关性
# ---------------------------------------------------------------------------
def _normalize_market_for_join(market: pd.DataFrame) -> pd.DataFrame:
    """把行情表日期归一化到“日”并输出 ``YYYY-MM-DD`` 字符串，便于内连接。"""
    mkt = market.copy()
    mkt["_dt"] = _normalize_date_series(mkt["date"])
    mkt = mkt.dropna(subset=["_dt"])
    mkt["date"] = mkt["_dt"].dt.strftime("%Y-%m-%d")
    return mkt


def prepare_xy(
    emotion_daily: pd.DataFrame,
    market: pd.DataFrame,
    target: str = "close",
    emotion_col: str = "pos_ratio",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """统一做情绪与行情的按日内连接与对齐，返回 ``(x, y, dates)``。

    ``correlate`` 与 ``linear_fit`` 的调用方均基于本函数，保证散点与拟合线使用
    完全一致的样本。两边日期都 ``pd.to_datetime`` 后 ``.dt.normalize()`` 到“日”。

    参数:
        emotion_daily: aggregate_daily 的输出（date, pos_ratio, neg_ratio, n）。
        market: 行情 DataFrame（date, close, pe）。
        target: 目标指标（"close" 或 "pe"）。
        emotion_col: 用作自变量的情绪列（默认 "pos_ratio"；也可传 "neg_ratio"
            以分析负向情绪与行情的相关性）。

    返回:
        ``(x, y, dates)`` 三元组：x 为所选情绪占比、y 为行情指标、dates 为对应日期
        （已归一化到日），三者等长。当输入为空、target 缺失或交集为空时返回空序列。
    """
    empty_x = pd.Series(dtype="float64")
    empty_y = pd.Series(dtype="float64")
    empty_dates = pd.Series(dtype="datetime64[ns]")

    if emotion_daily is None or emotion_daily.empty:
        return empty_x, empty_y, empty_dates
    if market is None or market.empty:
        return empty_x, empty_y, empty_dates

    emo = emotion_daily.copy()
    emo["_dt"] = _normalize_date_series(emo["date"])
    emo = emo.dropna(subset=["_dt"])
    # 情绪列可能在调用方传入时为缺失，此时回退到 pos_ratio（保持向后兼容，避免 KeyError）。
    if emotion_col not in emo.columns:
        emotion_col = "pos_ratio"
    emo[emotion_col] = pd.to_numeric(emo[emotion_col], errors="coerce")
    emo = emo.dropna(subset=[emotion_col])

    mkt = market.copy()
    mkt["_dt"] = _normalize_date_series(mkt["date"])
    mkt = mkt.dropna(subset=["_dt"])
    if target not in mkt.columns:
        return empty_x, empty_y, empty_dates
    mkt[target] = pd.to_numeric(mkt[target], errors="coerce")

    merged = emo.merge(mkt[["_dt", target]], on="_dt", how="inner")
    merged = merged.dropna(subset=[target])

    x = merged[emotion_col].astype(float).reset_index(drop=True)
    y = merged[target].astype(float).reset_index(drop=True)
    dates = merged["_dt"].reset_index(drop=True)
    return x, y, dates


def _market_has_target(market: pd.DataFrame, target: str) -> bool:
    """判断行情表是否含指定指标且至少有一个有效值。

    用于区分「该指标完全缺失（如有 close 无 pe 时的 pe）」与「按日期交集为空」：
    前者相关性返回 None（界面隐藏该图），后者返回 valid=False（有提示）。
    """
    if market is None or market.empty:
        return False
    mkt = market.copy()
    if target not in mkt.columns:
        return False
    return bool(pd.to_numeric(mkt[target], errors="coerce").notna().any())


# 新增：把日期列归一化后输出 min~max 描述串，供「交集为空」提示写明双方日期范围。
def _date_range_desc(df: pd.DataFrame, date_col: str = "date") -> str:
    """返回 DataFrame 日期列的 ``min~max`` 描述串（YYYY-MM-DD）；无法解析时返回「未知」。

    用于相关性按日期交集为空时，把情绪与行情双方的日期范围写进 message，便于用户自查。
    """
    if df is None or df.empty or date_col not in df.columns:
        return "未知"
    dt = _normalize_date_series(df[date_col]).dropna()
    if dt.empty:
        return "未知"
    return f"{dt.min().strftime('%Y-%m-%d')}~{dt.max().strftime('%Y-%m-%d')}"


def correlate(
    emotion_daily: pd.DataFrame,
    market: pd.DataFrame,
    target: str = "close",
    min_n: int = MIN_CORRELATION_N,
    emotion_col: str = "pos_ratio",
) -> CorrelationResult | None:
    """计算情绪占比与行情指标的相关性（Pearson r、p、n、R²）。

    参数:
        emotion_daily: aggregate_daily 的输出（date, pos_ratio, neg_ratio, n）。
        market: 行情 DataFrame（date, close, pe）。
        target: 目标指标（"close" 或 "pe"）。
        min_n: 样本量阈值，n 低于该值给出 warning。
        emotion_col: 用作自变量的情绪列（默认 "pos_ratio"；也可传 "neg_ratio"
            以分析负向情绪与行情的相关性）。

    返回:
        CorrelationResult；当 target 指标在行情中完全缺失（如有 close 无 pe 时的
        pe）返回 None，由界面隐藏该指标的相关性图，而不是报错。

    语义:
        - 数据能算（哪怕 n<30）→ ``valid=True``，n<30 仅附加 ``warning``，r/R²/p 照常返回。
        - 仅当情绪/行情为空，或按日期交集为空 → ``valid=False``。
    """
    if emotion_daily is None or emotion_daily.empty:
        return CorrelationResult(target=target, valid=False, message="情绪数据为空，无法计算相关性")
    if market is None or market.empty:
        return CorrelationResult(target=target, valid=False, message="行情数据为空，无法计算相关性")

    if not _market_has_target(market, target):
        # 该指标完全缺失（如有 close 无 pe 时的 pe）→ 返回 None，界面隐藏对应图
        logger.info("行情不含列 %s 或该指标无有效数据，返回 None（界面隐藏对应相关性）", target)
        return None

    x, y, _dates = prepare_xy(emotion_daily, market, target, emotion_col=emotion_col)
    n = int(len(x))
    if n == 0:
        # 修复：把「交集为空」的 message 精确化为双方日期范围，便于用户自查为何没交集。
        emo_range = _date_range_desc(emotion_daily)
        mkt_range = _date_range_desc(market)
        return CorrelationResult(
            target=target,
            valid=False,
            message=f"情绪与行情日期交集为空（情绪 {emo_range}，行情 {mkt_range}）",
        )

    r, p = stats.pearsonr(x, y)
    r_squared = float(r * r)

    warning = ""
    caution = ""
    if n < min_n:
        warning = f"样本量不足（n={n}<{min_n}），相关性仅供参考"
    if r_squared < LOW_R2_THRESHOLD:
        caution = "解释力有限，不宜作因果推断"

    return CorrelationResult(
        target=target,
        valid=True,
        r=float(r),
        p=float(p),
        n=n,
        r_squared=r_squared,
        warning=warning,
        caution=caution,
    )


# ---------------------------------------------------------------------------
# 3. 事件前后对比
# ---------------------------------------------------------------------------
def event_window_compare(
    df: pd.DataFrame,
    event_date: str,
    window_days: int = 90,
    date_col: str = "date",
) -> EventWindowResult:
    """对比事件日前后的情绪占比均值（返回前后均值、样本量、差值）。

    窗口定义：
    - 前窗口：``[event_date - window_days, event_date)``（不含事件日当天）。
    - 后窗口：``[event_date, event_date + window_days]``（含事件日当天）。

    参数:
        df: 含日期列与 pos_ratio/neg_ratio 列的数据。
        event_date: 事件日（YYYY-MM-DD）。
        window_days: 窗口天数。
        date_col: 日期列名。

    返回:
        EventWindowResult。某侧样本量为 0 时该侧均值与差值不参与计算，并在 message 中标注。
    """
    event = pd.to_datetime(event_date, errors="coerce")
    if pd.isna(event):
        return EventWindowResult(
            event_date=event_date, window_days=window_days, valid=False, message="事件日无法解析"
        )

    work = df.copy()
    work["_dt"] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=["_dt"])

    before = work[
        (work["_dt"] < event) & (work["_dt"] >= event - pd.Timedelta(days=window_days))
    ]
    after = work[
        (work["_dt"] >= event) & (work["_dt"] <= event + pd.Timedelta(days=window_days))
    ]

    def _mean(segment: pd.DataFrame, column: str) -> float | None:
        if segment.empty or column not in segment.columns:
            return None
        return float(pd.to_numeric(segment[column], errors="coerce").mean())

    before_count = int(len(before))
    after_count = int(len(after))
    before_pos = _mean(before, "pos_ratio")
    after_pos = _mean(after, "pos_ratio")
    before_neg = _mean(before, "neg_ratio")
    after_neg = _mean(after, "neg_ratio")

    messages: list[str] = []
    if before_count == 0:
        messages.append("事件前窗口样本量为 0，不参与差值计算")
    if after_count == 0:
        messages.append("事件后窗口样本量为 0，不参与差值计算")

    pos_diff: float | None = None
    neg_diff: float | None = None
    if before_count > 0 and after_count > 0 and before_pos is not None and after_pos is not None:
        pos_diff = after_pos - before_pos
    if before_count > 0 and after_count > 0 and before_neg is not None and after_neg is not None:
        neg_diff = after_neg - before_neg

    return EventWindowResult(
        event_date=event_date,
        window_days=window_days,
        valid=True,
        before_count=before_count,
        after_count=after_count,
        before_pos_mean=before_pos,
        after_pos_mean=after_pos,
        before_neg_mean=before_neg,
        after_neg_mean=after_neg,
        pos_diff=pos_diff,
        neg_diff=neg_diff,
        message="；".join(messages),
    )


# ---------------------------------------------------------------------------
# 4. 单变量线性回归
# ---------------------------------------------------------------------------
def linear_fit(
    x: Any,
    y: Any,
    min_n: int = MIN_CORRELATION_N,
) -> LinearFitResult:
    """单变量线性回归（情绪占比 → 行情指标）。

    参数:
        x: 自变量序列（如情绪正向占比），建议来自 :func:`prepare_xy` 的对齐结果，
           与相关性使用同一份样本。
        y: 因变量序列（如收盘价 / PE）。
        min_n: 样本量阈值，n >= min_n 才出拟合线与回归结论。

    返回:
        LinearFitResult。语义：有效样本 >=2 即 ``valid=True`` 并照常返回
        slope/intercept/r/R²/p；仅当 n<min_n 时 ``fit_ok=False``（界面只出散点图、
        不出拟合线与回归结论），``valid`` 仍为 True。
    """
    xs = pd.to_numeric(pd.Series(x), errors="coerce")
    ys = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = xs.notna() & ys.notna()
    xs = xs[mask].astype(float)
    ys = ys[mask].astype(float)
    n = int(len(xs))

    if n < 2:
        return LinearFitResult(valid=False, n=n, message="有效样本不足 2 个，无法回归")

    result = stats.linregress(xs, ys)
    fit_ok = n >= min_n
    message = "" if fit_ok else f"样本量不足（n={n}<{min_n}），仅出散点图，不出回归结论"
    return LinearFitResult(
        valid=True,
        n=n,
        slope=float(result.slope),
        intercept=float(result.intercept),
        r=float(result.rvalue),
        r_squared=float(result.rvalue**2),
        p=float(result.pvalue),
        fit_ok=fit_ok,
        message=message,
    )


# ---------------------------------------------------------------------------
# 5. 可选分组相关性
# ---------------------------------------------------------------------------
def optional_grouped_correlate(
    emotion_daily: pd.DataFrame,
    market: pd.DataFrame,
    event_date: str,
    target: str = "close",
    min_n: int = MIN_CORRELATION_N,
) -> GroupedCorrelationResult:
    """可选分组相关性：按事件日分「事件前 / 事件后」两组分别算 r、R²、n。

    默认关闭（由界面开关控制）；开启时结果固定带 note「分组样本量可能很小，结果仅供参考」。

    参数:
        emotion_daily: aggregate_daily 的输出。
        market: 行情 DataFrame（date, close, pe）。
        event_date: 事件日（YYYY-MM-DD）。
        target: 目标指标（"close" 或 "pe"）。
        min_n: 样本量阈值。

    返回:
        GroupedCorrelationResult。
    """
    event = pd.to_datetime(event_date, errors="coerce")
    if pd.isna(event):
        return GroupedCorrelationResult(event_date=event_date, valid=False, note="事件日无法解析")

    if emotion_daily is None or emotion_daily.empty or market is None or market.empty:
        return GroupedCorrelationResult(
            event_date=event_date, valid=False, note="情绪或行情数据为空"
        )

    mkt = _normalize_market_for_join(market)
    if target not in mkt.columns:
        return GroupedCorrelationResult(
            event_date=event_date, valid=False, note=f"行情不含列 {target}"
        )
    mkt[target] = pd.to_numeric(mkt[target], errors="coerce")

    merged = emotion_daily.merge(mkt[["date", target]], on="date", how="inner")
    merged["_dt"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = merged.dropna(subset=["_dt", target, "pos_ratio"])

    before = merged[merged["_dt"] < event]
    after = merged[merged["_dt"] >= event]

    def _corr(segment: pd.DataFrame) -> dict[str, Any]:
        if segment.empty or len(segment) < 2:
            return {
                "r": None,
                "r_squared": None,
                "p": None,
                "n": int(len(segment)),
                "warning": "样本不足，无法计算",
            }
        r, p = stats.pearsonr(segment["pos_ratio"].astype(float), segment[target].astype(float))
        n = int(len(segment))
        warning = ""
        if n < min_n:
            warning = f"样本量不足（n={n}<{min_n}），仅供参考"
        return {
            "r": float(r),
            "r_squared": float(r * r),
            "p": float(p),
            "n": n,
            "warning": warning,
        }

    return GroupedCorrelationResult(
        event_date=event_date,
        valid=True,
        before=_corr(before),
        after=_corr(after),
    )
