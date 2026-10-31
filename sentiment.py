# -*- coding: utf-8 -*-
"""情绪分析模块。

支持两种模式：
A. 原始模式（company/date/question/answer）：
   加载词典 → match_emotions 匹配 → compute_ratios 计算正负占比。
B. 已分析模式（用户上传、已跑过词典匹配的表）：
   跳过匹配，直接 parse_emotion_columns 拆分四类情绪词列、parse_ratio_columns 校验占比列，
   词云使用这四列词频，不对 answer 重新分词、不重算占比。

词典读取约定：四个词典 sheet（年报负面/年报正面/社媒负面/社媒正面）按
``header=None`` 读第一列并 ``strip()``，自动跳过「说明」页。
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

import config
from cleaner import EMOTION_WORD_COLUMNS, RATIO_COLUMNS, find_column, parse_datetime

logger = logging.getLogger(__name__)


class LexiconError(Exception):
    """词典加载异常，携带可读的中文原因。"""


# 词典 sheet 名 → 内部英文键
LEXICON_SHEET_MAP: dict[str, str] = {
    "年报负面": "neg_annual",
    "年报正面": "pos_annual",
    "社媒负面": "neg_social",
    "社媒正面": "pos_social",
}

# 情绪词单元格的分隔符（已分析模式下拆分四列用，统一转为顿号后再拆）
_EMOTION_SEPARATORS: tuple[str, ...] = ("、", "，", ",", ";", "；", "|", "/", " ")


def _split_words(value: Any) -> list[str]:
    """把情绪词单元格按分隔符拆成词列表；空值返回空列表；已是列表/元组则原样返回。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(word).strip() for word in value if str(word).strip()]
    if isinstance(value, float) and pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "null"):
        return []
    for sep in _EMOTION_SEPARATORS:
        text = text.replace(sep, "、")
    return [word.strip() for word in text.split("、") if word.strip()]


def clip_ratio(value: Any) -> float:
    """把占比值 clip 到 [0,1]（占比约定为 0~1 的小数）。

    参数:
        value: 任意数值（可能为字符串 / 缺失值）。

    返回:
        [0,1] 内的浮点数；无法解析或缺失返回 0.0；越界值收敛到边界。
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(number):
        return 0.0
    return max(0.0, min(1.0, number))


# ---------------------------------------------------------------------------
# 词典加载
# ---------------------------------------------------------------------------
def load_lexicon(path: str | Path | None = None) -> dict[str, set[str]]:
    """加载情绪词典（四个 sheet，header=None 读第一列并 strip，跳过「说明」页）。

    参数:
        path: 词典 Excel 路径；为 None 时使用 config.DEFAULT_DICT_PATH。

    返回:
        形如 ``{"neg_annual": {...}, "pos_annual": {...}, ...}`` 的词集合字典。

    异常:
        LexiconError: 词典文件不存在/无法读取，或四个 sheet 全部缺失或为空。
    """
    path = Path(path) if path else config.DEFAULT_DICT_PATH
    if not path.exists():
        raise LexiconError(f"情绪词典不存在：{path}")

    try:
        xls = pd.ExcelFile(path)
    except Exception as exc:  # noqa: BLE001 —— 明确记录并抛出
        raise LexiconError(f"无法打开情绪词典 {path}：{exc}") from exc

    lexicon: dict[str, set[str]] = {}
    for sheet_name, key in LEXICON_SHEET_MAP.items():
        if sheet_name not in xls.sheet_names:
            logger.warning("词典缺少 sheet「%s」，该词典词集置空", sheet_name)
            lexicon[key] = set()
            continue

        raw = xls.parse(sheet_name, header=None)
        words: set[str] = set()
        if raw is not None and not raw.empty and raw.shape[1] >= 1:
            for value in raw.iloc[:, 0]:
                word = str(value).strip() if value is not None else ""
                if word and word.lower() != "nan":
                    words.add(word)
        lexicon[key] = words
        logger.info("已加载词典 sheet「%s」：%d 词", sheet_name, len(words))

    if not any(lexicon.values()):
        raise LexiconError(f"情绪词典四个 sheet 均缺失或为空：{path}")

    return lexicon


# ---------------------------------------------------------------------------
# 原始模式
# ---------------------------------------------------------------------------
def match_emotions(text: str, lexicon: dict[str, set[str]]) -> dict[str, list[str]]:
    """在单条文本中匹配四类情绪词，返回每类命中的词列表（含重复，表示词频）。

    参数:
        text: 待分析文本。
        lexicon: load_lexicon 的返回值。

    返回:
        形如 ``{"neg_annual": [...], "pos_annual": [...], ...}`` 的字典。
        空文本或无法命中时各键对应空列表。
    """
    hits: dict[str, list[str]] = {key: [] for key in LEXICON_SHEET_MAP.values()}
    if not text or not isinstance(text, str):
        return hits
    text = text.strip()
    if not text:
        return hits

    for key, words in lexicon.items():
        for word in words:
            count = text.count(word)
            if count > 0:
                hits[key].extend([word] * count)
    return hits


def compute_ratios(text: str, lexicon: dict[str, set[str]]) -> tuple[float, float]:
    """原始模式：计算正负占比（命中词字符长度 / 文本总长度，并 clip 到 [0,1]）。

    正向 = 年报正面 + 社媒正面；负向 = 年报负面 + 社媒负面。
    分子为该向命中词的总字符长度（重复命中按词频累计，如「感谢」「支持」共 4 字符）；
    分母为该条文本的总字符长度 ``len(text)``。
    文本长度 <= 5（如「您好」等无实质内容的废话）直接返回 ``(0.0, 0.0)``，不参与计算。

    参数:
        text: 待分析文本。
        lexicon: load_lexicon 的返回值。

    返回:
        ``(pos_ratio, neg_ratio)``。
    """
    if not text or not isinstance(text, str):
        return 0.0, 0.0
    text = text.strip()
    total_len = len(text)
    if total_len <= 5:
        return 0.0, 0.0

    hits = match_emotions(text, lexicon)
    pos_words = hits["pos_annual"] + hits["pos_social"]
    neg_words = hits["neg_annual"] + hits["neg_social"]
    pos_len = sum(len(word) for word in pos_words)
    neg_len = sum(len(word) for word in neg_words)
    return clip_ratio(pos_len / total_len), clip_ratio(neg_len / total_len)


# ---------------------------------------------------------------------------
# 已分析模式
# ---------------------------------------------------------------------------
def auto_map_emotion_columns(df: pd.DataFrame) -> dict[str, str]:
    """自动匹配四类情绪词列，返回 ``{标准列: 源列}``（匹配不到的键不出现）。"""
    column_map: dict[str, str] = {}
    for standard, aliases in EMOTION_WORD_COLUMNS.items():
        found = find_column(df, aliases)
        if found is not None:
            column_map[standard] = found
    return column_map


def auto_map_ratio_columns(df: pd.DataFrame) -> dict[str, str]:
    """自动匹配正负占比列，返回 ``{标准列: 源列}``（匹配不到的键不出现）。"""
    column_map: dict[str, str] = {}
    for standard, aliases in RATIO_COLUMNS.items():
        found = find_column(df, aliases)
        if found is not None:
            column_map[standard] = found
    return column_map


def parse_emotion_columns(
    df: pd.DataFrame,
    column_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """已分析模式：把四类情绪词列按分隔符拆成词列表（空值→空列表）。

    参数:
        df: 含四类情绪词列（中文列名）的 DataFrame。
        column_map: 显式映射 ``{标准列: 源列}``（由界面手动映射传入）；
                    为 None 时自动匹配。显式映射传入后会过滤掉不在 df 中的源列。

    返回:
        列名为 ``neg_annual_words/pos_annual_words/neg_social_words/pos_social_words``、
        每格为 ``list[str]`` 的 DataFrame（与输入行对齐）。

    异常:
        ValueError: 显式映射为空且自动匹配也找不到任何一类情绪词列。
    """
    if column_map is None:
        column_map = auto_map_emotion_columns(df)
    # 过滤掉不在 df 中的源列（手动映射可能指向不存在的列）
    column_map = {std: src for std, src in column_map.items() if src in df.columns}

    if not column_map:
        raise ValueError(
            f"未找到四类情绪词列，请确认上传表包含类似列名："
            f"{list(EMOTION_WORD_COLUMNS.values())}"
        )

    result = pd.DataFrame(index=df.index)
    for standard, source in column_map.items():
        result[standard] = df[source].apply(_split_words)
    for standard in EMOTION_WORD_COLUMNS:
        if standard not in result.columns:
            result[standard] = [[] for _ in range(len(df))]
    return result


def parse_ratio_columns(
    df: pd.DataFrame,
    column_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """已分析模式：读取正负占比列并用 clip_ratio 校验/归一化到 [0,1]。

    参数:
        df: 含正负占比列（中文列名）的 DataFrame。
        column_map: 显式映射 ``{标准列: 源列}``（由界面手动映射传入）；
                    为 None 时自动匹配。

    返回:
        列 ``pos_ratio/neg_ratio``（float）的 DataFrame；缺失列以 0.0 填充。
    """
    if column_map is None:
        column_map = auto_map_ratio_columns(df)
    column_map = {std: src for std, src in column_map.items() if src in df.columns}

    result = pd.DataFrame(index=df.index)
    for standard in RATIO_COLUMNS:
        source = column_map.get(standard)
        if source is None:
            result[standard] = 0.0
        else:
            result[standard] = df[source].apply(clip_ratio)
    return result


# 新增：由四类情绪词「词数」推算正负占比（无原文长度分母，仅作趋势参考）。
def derive_ratios_from_word_lists(words_df: pd.DataFrame) -> pd.DataFrame:
    """由四类情绪词命中词数推算正负占比。

    已分析模式下，若上传表没有提供占比列，``parse_ratio_columns`` 会把正负占比都补成 0，
    进而导致后续相关性统计得 nan。本函数按四类词的词数推算一个正负倾向占比作为兜底。

    公式：
        pos = pos_annual 词数 + pos_social 词数
        neg = neg_annual 词数 + neg_social 词数
        total = pos + neg（total <= 0 时按 1 处理，避免除零）
        pos_ratio = pos/total、neg_ratio = neg/total，均 clip 到 [0,1]

    注意：这里没有原文长度作分母，仅反映正/负命中词之间的相对强弱，**仅作趋势参考**。

    参数:
        words_df: ``parse_emotion_columns`` 的返回值（列名为四类标准词列，每格为 ``list[str]``）。

    返回:
        列 ``pos_ratio/neg_ratio``（float）的 DataFrame，index 与 words_df 对齐。
    """
    pos = (
        words_df["pos_annual_words"].apply(len)
        + words_df["pos_social_words"].apply(len)
    )
    neg = (
        words_df["neg_annual_words"].apply(len)
        + words_df["neg_social_words"].apply(len)
    )
    total = (pos + neg).clip(lower=1)
    result = pd.DataFrame(index=words_df.index)
    result["pos_ratio"] = (pos / total).clip(0.0, 1.0)
    result["neg_ratio"] = (neg / total).clip(0.0, 1.0)
    return result


def emotion_word_frequency(
    word_lists: dict[str, list[str]] | pd.DataFrame,
) -> pd.DataFrame:
    """把四类情绪词列表统计为词频表（供词云使用）。

    参数:
        word_lists: 形如 ``{"neg_annual": [...], ...}`` 的字典，
                    或 ``parse_emotion_columns`` 的返回值。

    返回:
        列 ``word/count/category`` 的 DataFrame，按 count 降序；无词时返回空表。
    """
    if isinstance(word_lists, pd.DataFrame):
        data: dict[str, list[str]] = {}
        for standard in EMOTION_WORD_COLUMNS:
            if standard in word_lists.columns:
                flat: list[str] = []
                for cell in word_lists[standard]:
                    flat.extend(cell or [])
                data[standard] = flat
    else:
        data = word_lists

    rows: list[dict[str, Any]] = []
    for category, words in data.items():
        for word, count in Counter(words).items():
            rows.append({"word": word, "count": count, "category": category})

    if not rows:
        return pd.DataFrame(columns=["word", "count", "category"])
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def word_frequency_from_detail(detail: pd.DataFrame) -> pd.DataFrame:
    """从情绪明细表统计四类词频（供词云使用）。

    明细表由 ``build_emotion_detail_*`` 产出，含英文标准列
    ``neg_annual_words/pos_annual_words/neg_social_words/pos_social_words``，
    值为「、」分隔的字符串（也可能是列表）。本函数拆分后统计词频，
    不对「回答信息」重新分词、不重算占比。

    参数:
        detail: 情绪明细表。

    返回:
        列 ``word/count/category`` 的 DataFrame；无词或无词列时返回空表（不抛异常）。
    """
    if detail is None or detail.empty:
        return pd.DataFrame(columns=["word", "count", "category"])

    parsed = pd.DataFrame(index=detail.index)
    found_any = False
    for standard in EMOTION_WORD_COLUMNS:
        if standard in detail.columns:
            parsed[standard] = detail[standard].apply(_split_words)
            found_any = True
        else:
            parsed[standard] = [[] for _ in range(len(detail))]

    if not found_any:
        logger.warning("情绪明细表不含任何四类情绪词列，词频表为空")
        return pd.DataFrame(columns=["word", "count", "category"])

    return emotion_word_frequency(parsed)


def word_frequency_from_texts(
    texts: list[str] | tuple[str, ...] | pd.Series,
    lexicon: dict[str, set[str]],
    keys: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """对文本列表用指定词典键做全遍历匹配，统计词频（供「按来源」词云使用）。

    与报告方法论一致：社媒词云用「社媒词典键」匹配投资者提问文本，
    年报词云用「年报词典键」匹配年报原文，各自统计命中词频。

    参数:
        texts: 待匹配的文本序列。
        lexicon: ``load_lexicon`` 的返回值。
        keys: 词典内部键列表，如 ``("pos_social", "neg_social")`` 或
              ``("pos_annual", "neg_annual")``。

    返回:
        列 ``word/count/category`` 的 DataFrame，category 为带 ``_words`` 后缀的
        标准列名（如 ``pos_social_words``），与 ``make_wordcloud_grid`` 的
        ``_CATEGORY_LABELS`` 键一致；无命中时返回空表（不抛异常）。
    """
    counters: dict[str, Counter] = {key: Counter() for key in keys}
    for text in texts:
        if text is None:
            continue
        text = str(text).strip()
        if not text:
            continue
        for key in keys:
            for word in lexicon.get(key, ()):
                count = text.count(word)
                if count > 0:
                    counters[key][word] += count

    rows: list[dict[str, Any]] = []
    for key in keys:
        for word, count in counters[key].items():
            rows.append({"word": word, "count": count, "category": f"{key}_words"})

    if not rows:
        return pd.DataFrame(columns=["word", "count", "category"])
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 两种模式共用：时间聚合与事件窗口对比
# ---------------------------------------------------------------------------
def aggregate_by_month(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """按月份聚合正负占比均值与样本数（两种模式共用）。

    参数:
        df: 含日期列与 ``pos_ratio/neg_ratio`` 列的数据。
        date_col: 日期列名。

    返回:
        列 ``month/pos_ratio/neg_ratio/count`` 的 DataFrame，按 month 升序。
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["month", "pos_ratio", "neg_ratio", "count"])

    work = df.copy()
    # 复用 cleaner.parse_datetime 以支持中文日期（如 "2019年02月21日 08:27"），
    # 与 analysis.aggregate_daily 的日期归一化口径保持一致。
    work["_dt"] = pd.Series(
        [parse_datetime(value) for value in work[date_col]], index=work.index
    )
    work = work.dropna(subset=["_dt"])
    work["month"] = work["_dt"].dt.to_period("M").astype(str)

    pos = work["pos_ratio"] if "pos_ratio" in work else pd.Series([0.0] * len(work))
    neg = work["neg_ratio"] if "neg_ratio" in work else pd.Series([0.0] * len(work))

    agg = pd.DataFrame(
        {
            "month": work["month"],
            "pos_ratio": pd.to_numeric(pos, errors="coerce").fillna(0.0),
            "neg_ratio": pd.to_numeric(neg, errors="coerce").fillna(0.0),
        }
    )
    grouped = (
        agg.groupby("month")
        .agg(pos_ratio=("pos_ratio", "mean"), neg_ratio=("neg_ratio", "mean"), count=("pos_ratio", "size"))
        .reset_index()
    )
    return grouped


def event_window_compare(
    df: pd.DataFrame,
    event_date: str,
    before_days: int = 90,
    after_days: int = 90,
    date_col: str = "date",
) -> dict[str, Any]:
    """对比事件日前后的正负占比均值（两种模式共用）。

    参数:
        df: 含日期列与 ``pos_ratio/neg_ratio`` 列的数据。
        event_date: 事件日（YYYY-MM-DD）。
        before_days: 事件前窗口天数（含事件日之前，不含事件日当天）。
        after_days: 事件后窗口天数（含事件日当天）。
        date_col: 日期列名。

    返回:
        dict，含 ``event_date/valid/before_count/after_count`` 与前后窗口的
        ``before_pos_ratio/after_pos_ratio/before_neg_ratio/after_neg_ratio``；
        事件日无法解析时 ``valid=False`` 并附 ``message``。
    """
    event = pd.to_datetime(event_date, errors="coerce")
    if pd.isna(event):
        return {"event_date": event_date, "valid": False, "message": "事件日无法解析"}

    work = df.copy()
    # 支持中文日期，与 aggregate_by_month / analysis 模块保持一致口径。
    work["_dt"] = pd.Series(
        [parse_datetime(value) for value in work[date_col]], index=work.index
    )
    work = work.dropna(subset=["_dt"])

    before = work[
        (work["_dt"] < event) & (work["_dt"] >= event - pd.Timedelta(days=before_days))
    ]
    after = work[
        (work["_dt"] >= event) & (work["_dt"] <= event + pd.Timedelta(days=after_days))
    ]

    def _mean(segment: pd.DataFrame, column: str) -> float:
        if segment.empty or column not in segment:
            return 0.0
        return float(pd.to_numeric(segment[column], errors="coerce").mean())

    return {
        "event_date": event_date,
        "valid": True,
        "before_count": int(len(before)),
        "after_count": int(len(after)),
        "before_pos_ratio": _mean(before, "pos_ratio"),
        "after_pos_ratio": _mean(after, "pos_ratio"),
        "before_neg_ratio": _mean(before, "neg_ratio"),
        "after_neg_ratio": _mean(after, "neg_ratio"),
    }
