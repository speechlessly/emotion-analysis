# -*- coding: utf-8 -*-
"""数据清洗模块。

职责：
1. 字段映射：把任意列名映射到统一 schema（company/date/question/answer）。
2. 中文日期解析：支持 "2019年02月21日 08:27" 等格式。
3. 去重、空值处理。
4. detect_mode：识别数据是「原始模式」（question 提问列）还是
   「已分析模式」（四类情绪词列已存在）。

两种模式约定：
- 原始模式：company / date / question / answer，但**只用 question 提问列**匹配词典，
  answer 回答列绝不参与情绪计算。
- 已分析模式：用户已跑过词典匹配，表内已有四类情绪词列与正负占比列，跳过匹配。
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Any, Iterable

import pandas as pd

import config

logger = logging.getLogger(__name__)

# 数据模式标识（detect_mode 的返回值）
MODE_RAW = "原始模式"          # 有 question 提问列
MODE_ANALYZED = "已分析模式"   # 有四类情绪词列
MODE_AMBIGUOUS = "需用户选择"  # 两者都有，界面应让用户手动选择
MODE_UNKNOWN = "无法识别"      # 两者都没有

# 统一评论 schema 字段别名：标准英文列 → 常见源列名
COMMENT_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "company": ("company", "公司", "公司名称", "证券简称", "公司简称"),
    "date": ("date", "时间", "日期", "提问时间", "回答时间", "发布时间"),
    "question": ("question", "提问", "问题", "提问信息", "投资者提问"),
    "answer": ("answer", "回答", "回答信息", "回复", "公司回复", "答复"),
}

# 已分析模式的四类情绪词列：标准英文列 → 常见中文列名
# 修复：补充用户真实表头（如「年报负面情绪评级」「正面情绪评级」等），提升自动匹配命中率。
EMOTION_WORD_COLUMNS: dict[str, tuple[str, ...]] = {
    "neg_annual_words": ("年报负面情绪词", "年报负面词", "年报负面", "年报负面情绪评级"),
    "pos_annual_words": (
        "年报正面情绪词", "年报正面词", "年报正面",
        "年报正面情绪评级", "正面情绪评级", "正面情绪",
    ),
    "neg_social_words": (
        "社媒负面情绪词", "社媒负面词", "社媒负面",
        "社媒负面情绪评级", "负面情绪评级", "负面情绪",
    ),
    "pos_social_words": ("社媒正面情绪词", "社媒正面词", "社媒正面", "社媒正面情绪评级"),
}

# 已分析模式的正负占比列：标准英文列 → 常见中文列名
RATIO_COLUMNS: dict[str, tuple[str, ...]] = {
    "pos_ratio": ("正向占比", "正面占比", "积极占比", "正向比例"),
    "neg_ratio": ("负向占比", "负面占比", "消极占比", "负向比例"),
}

# 中文日期正则："2019年02月21日 08:27" → "2019-02-21 08:27"
_CN_DATE_RE = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")


class CleanerError(Exception):
    """清洗相关异常，携带可读的中文原因。"""


# 常见零宽/隐藏字符（从网页或 Excel 复制列名时可能混入）
# 覆盖：零宽空格、零宽不连字、零宽连字、BOM、词连接符，另由下方 \s+ 去除全角空格/不间断空格。
_ZERO_WIDTH_CHARS: str = "\u200b\u200c\u200d\ufeff\u2060"


def _contains_chinese(text: str) -> bool:
    """判断字符串是否含中文（用于限制「包含匹配」仅对中文列名生效，避免英文短词误匹配）。"""
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _normalize_column_name(name: Any) -> str:
    """规范化列名：去除零宽字符与所有 Unicode 空白（含全角空格、不间断空格）。"""
    text = str(name)
    for ch in _ZERO_WIDTH_CHARS:
        text = text.replace(ch, "")
    return re.sub(r"\s+", "", text)


def find_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    """在 df 列中按别名查找匹配列，返回第一个命中的原始列名；找不到返回 None。

    三级容错匹配：
    1) 精确匹配（strip 后完全相等）；
    2) 规范化匹配（去零宽字符与所有空白，兼容全角空格/不间断空格等隐藏空白）；
    3) 包含匹配（仅对中文别名启用：别名是列名的子串，兼容近似列名）。

    参数:
        df: 目标 DataFrame。
        aliases: 候选列名元组。

    返回:
        命中的原始列名，或 None。
    """
    if df is None:
        return None

    # 1) 精确匹配
    for column in df.columns:
        if str(column).strip() in aliases:
            return column

    # 2) 规范化匹配（去所有空白，兼容隐藏空格）
    norm_aliases = {_normalize_column_name(a) for a in aliases}
    for column in df.columns:
        if _normalize_column_name(column) in norm_aliases:
            return column

    # 3) 包含匹配（仅中文别名，避免 "date" in "update" 之类误匹配）。
    #    修复：只允许「别名是列名的子串」（na in c，列名更具体，如"年报正面情绪评级"包含别名"正面情绪"），
    #    去掉「列名是别名的子串」（c in na，如列名"正面情绪评级"被别名"社媒正面情绪评级"反向包含），
    #    避免笼统列名被多个标准字段的具体别名重复映射产生冲突。
    for column in df.columns:
        c = _normalize_column_name(column)
        if not c or not _contains_chinese(c):
            continue
        for alias in aliases:
            na = _normalize_column_name(alias)
            if len(na) >= 2 and _contains_chinese(na) and na in c:
                return column
    return None


def has_emotion_columns(df: pd.DataFrame) -> bool:
    """判断 df 是否至少包含一类情绪词列（用于识别已分析模式）。"""
    for aliases in EMOTION_WORD_COLUMNS.values():
        if find_column(df, aliases) is not None:
            return True
    return False


# 新增：判断某列是否「真正有内容」（修复 question 列存在但全空被误判为有效特征的问题）。
def _column_has_content(df: pd.DataFrame, col: Any) -> bool:
    """判断某列是否含有有效文本内容（用于模式识别的「有效特征」判定）。

    规则：col 为 None 或不在 df 中返回 False；否则列内去除首尾空白后，
    至少存在一个非空、且不是字面量占位符（``nan``/``none``/``null``，忽略大小写）的值才返回 True。

    参数:
        df: 目标 DataFrame。
        col: 列名（可能为 None）。

    返回:
        bool。
    """
    if col is None or col not in df.columns:
        return False
    series = df[col].fillna("").astype(str).str.strip()
    lowered = series.str.lower()
    valid = lowered.ne("") & ~lowered.isin({"nan", "none", "null"})
    return bool(valid.any())


def detect_mode_detail(df: pd.DataFrame) -> dict[str, str]:
    """识别数据模式，并给出中文判定理由（用于界面展示「模式识别依据」）。

    判定规则（answer「回答」列**绝不参与**模式判定与情绪计算）：
    - 有四类情绪词列之一 → 已分析特征；
    - 有 question「提问」列 → 原始特征；
    - question 列与情绪词列同时存在 → ``MODE_AMBIGUOUS``（界面让用户选）；
    - 只有原始特征 → ``MODE_RAW``；只有已分析特征 → ``MODE_ANALYZED``；
    - 都没有 → ``MODE_UNKNOWN``。

    参数:
        df: 原始 DataFrame。

    返回:
        形如 ``{"mode": 模式常量, "reason": 中文判定理由}`` 的字典。
    """
    if df is None or df.empty:
        return {"mode": MODE_UNKNOWN, "reason": "数据为空，无法识别数据模式"}

    question_col = find_column(df, COMMENT_COLUMN_ALIASES["question"])

    # 命中到的情绪词列（标准名 → 原始列名）
    emotion_hits: dict[str, str] = {}
    for standard, aliases in EMOTION_WORD_COLUMNS.items():
        found = find_column(df, aliases)
        if found is not None:
            emotion_hits[standard] = found

    has_analyzed = bool(emotion_hits)
    # 修复：question 列「存在但全空」不应视为有效原始特征，需进一步校验列内是否有内容。
    has_question = question_col is not None and _column_has_content(df, question_col)
    hit_desc = "、".join(f"「{name}」" for name in emotion_hits.values())

    if has_question and has_analyzed:
        return {
            "mode": MODE_AMBIGUOUS,
            "reason": (
                f"同时检测到提问信息列「{question_col}」与情绪词列（{hit_desc}），"
                "两种模式特征并存，需用户手动选择处理方式"
            ),
        }

    if has_analyzed:
        if len(emotion_hits) == len(EMOTION_WORD_COLUMNS):
            label = "四类情绪词列"
        else:
            label = f"情绪词列（命中 {len(emotion_hits)}/{len(EMOTION_WORD_COLUMNS)} 类）"
        extra = "，无 question 列" if not has_question else ""
        return {
            "mode": MODE_ANALYZED,
            "reason": f"检测到{label}（{hit_desc}）{extra} → 已分析模式（跳过词典匹配）",
        }

    if has_question:
        return {
            "mode": MODE_RAW,
            "reason": (
                f"检测到提问文本列「{question_col}」（question），未检测到情绪词列"
                " → 原始模式（需加载词典匹配）"
            ),
        }

    return {
        "mode": MODE_UNKNOWN,
        "reason": "未检测到提问列，也未检测到四类情绪词列，无法识别数据模式（回答列不参与分析）",
    }


def detect_mode(df: pd.DataFrame) -> str:
    """识别数据属于哪种模式（返回模式标识常量）。

    参数:
        df: 原始 DataFrame。

    返回:
        模式标识常量（见模块顶部 MODE_*）。判定规则与理由见 :func:`detect_mode_detail`。
    """
    return detect_mode_detail(df)["mode"]


def finalize_mode(
    df: pd.DataFrame,
    mode_preview: str | None = None,
    detected_mode: str | None = None,
) -> tuple[str, str]:
    """对数据模式做最终判定（含防御式复核），返回 ``(模式, 理由)``。

    用途：界面层在得到 ``detect_mode_detail`` 的初判后调用本函数，防止
    「原始模式数据被误判为已分析模式」这类致命问题——例如在线爬取返回
    ``company/date/question/answer`` 时，即便初判异常，也绝不进入已分析分支。
    注意：answer「回答」列绝不参与模式判定与情绪计算。

    规则：
    1. 初判为「需用户选择」且界面已给出手动选择时，采用界面选择。
    2. 初判为「已分析模式」但数据实际不含四类情绪词列：
       - 若含 question 提问列 → 修正为「原始模式」；
       - 否则 → 修正为「无法识别」（界面应提示并停止）。
    3. 其余情况维持初判。

    参数:
        df: 原始 DataFrame。
        mode_preview: 界面层在「需用户选择」时给出的手动选择（可为 None）。
        detected_mode: 显式指定的初判模式；默认 None 时自动调用
            :func:`detect_mode_detail` 判定。此参数主要用于单元测试注入
            「误判」场景以验证防御分支，正常业务调用无需传入。

    返回:
        ``(最终模式常量, 中文理由)``。
    """
    detail = detect_mode_detail(df)
    mode = detected_mode if detected_mode is not None else detail["mode"]
    reason = detail["reason"]

    if mode == MODE_AMBIGUOUS and mode_preview in (MODE_RAW, MODE_ANALYZED):
        return mode_preview, (
            f"数据同时含提问列与情绪词列，已由用户在界面手动选择为：{mode_preview}"
        )

    # 修复：初判为「需用户选择」但界面未给出手动选择（mode_preview 非 RAW/ANALYZED）时，
    # 不再落空导致误入原始模式，而是默认按「已分析模式」处理（含四类情绪词列时）。
    if mode == MODE_AMBIGUOUS:
        if has_emotion_columns(df):
            return MODE_ANALYZED, "数据同时含提问列与四类情绪词列，默认按已分析模式处理"
        return MODE_RAW, (
            "数据同时含提问列与情绪词列特征，但未识别到四类情绪词列，默认按原始模式处理"
        )

    if mode == MODE_ANALYZED and not has_emotion_columns(df):
        has_question = find_column(df, COMMENT_COLUMN_ALIASES["question"]) is not None
        if has_question:
            return MODE_RAW, (
                "初判为已分析模式但数据不含四类情绪词列，"
                "检测到提问列，已修正为原始模式"
            )
        return MODE_UNKNOWN, (
            "初判为已分析模式但数据不含四类情绪词列，也无法识别提问列"
        )

    return mode, reason


def map_columns(
    df: pd.DataFrame,
    mapping: dict[str, str] | None = None,
    auto: bool = True,
) -> pd.DataFrame:
    """把源列名映射/重命名为统一标准列名（company/date/question/answer）。

    参数:
        df: 原始 DataFrame。
        mapping: 显式映射 {标准列: 源列}；为 None 时按别名自动识别。
        auto: mapping 为 None 时是否启用别名自动识别（默认启用）。

    返回:
        重命名并仅保留标准列的 DataFrame；未映射到的标准列不会出现在结果中。
    """
    if df is None or df.empty:
        return df

    # mapping 为 None 或空 dict 时，只要允许自动识别，就按别名自动匹配。
    # （修复：空 dict 不再直接返回未重命名的原表，避免调用方误以为已完成映射）
    if auto and not mapping:
        mapping = {}
        for standard, aliases in COMMENT_COLUMN_ALIASES.items():
            found = find_column(df, aliases)
            if found is not None:
                mapping[standard] = found

    if not mapping:
        return df

    rename = {src: dst for dst, src in mapping.items() if src in df.columns}
    result = df.rename(columns=rename)
    keep = [col for col in result.columns if col in mapping]
    return result[keep].copy()


def parse_datetime(value: Any) -> pd.Timestamp:
    """把单个值解析为 pandas 时间戳，支持中文日期；失败返回 NaT。

    支持格式示例：
    - "2019年02月21日 08:27" / "2019年2月21日"
    - "2019-02-21 08:27" / "2019/2/21"
    - 已是 datetime/Timestamp 直接返回。

    参数:
        value: 任意待解析值。

    返回:
        pd.Timestamp；无法解析时为 pd.NaT。
    """
    if value is None:
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, (datetime, date)):
        return pd.Timestamp(value)

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return pd.NaT

    # "2019年02月21日 08:27" → "2019-02-21 08:27"
    text = _CN_DATE_RE.sub(
        lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}", text
    )
    parsed = pd.to_datetime(text, errors="coerce")
    return pd.NaT if pd.isna(parsed) else pd.Timestamp(parsed)


def parse_date_series(series: Iterable[Any]) -> pd.Series:
    """批量解析日期列，返回解析后的 Series（无法解析的位置为 NaT）。"""
    return pd.Series([parse_datetime(value) for value in series])


def drop_duplicates(
    df: pd.DataFrame, subset: list[str] | None = None
) -> pd.DataFrame:
    """按指定列去重（保留首条，重建索引）。"""
    if df is None or df.empty:
        return df
    return df.drop_duplicates(subset=subset).reset_index(drop=True)


def fill_missing_text(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """把指定文本列的空值填充为空字符串（同时清掉 pandas 转出的 "nan" 字面量）。"""
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = result[column].fillna("").astype(str)
            result[column] = result[column].replace("nan", "")
    return result


def clean_raw_comments(
    df: pd.DataFrame,
    mapping: dict[str, str] | None = None,
    company_default: str = "",
    drop_na_date: bool = True,
) -> pd.DataFrame:
    """原始模式完整清洗：映射列 → 解析日期 → 空值填充 → 去重。

    参数:
        df: 原始 DataFrame。
        mapping: 显式列映射 {标准列: 源列}；为 None 时按别名自动识别。
        company_default: company 列为空时填充的默认公司名（可为空）。
        drop_na_date: 是否丢弃日期无法解析的行（默认 True）。

    返回:
        统一列 ``company/date/question/answer`` 的清洗后 DataFrame。
    """
    mapped = map_columns(df, mapping=mapping)
    if mapped.empty:
        return mapped

    # 保证四个标准列都存在（缺失的补空列，避免后续 KeyError）
    for column in config.COMMENT_COLUMNS:
        if column not in mapped.columns:
            mapped[column] = ""

    mapped = fill_missing_text(mapped, ["company", "question", "answer"])
    if company_default:
        mapped["company"] = mapped["company"].replace("", company_default)

    # 日期解析：无法解析的位置为 NaT
    mapped["date"] = parse_date_series(mapped["date"])
    if drop_na_date:
        before = len(mapped)
        mapped = mapped.dropna(subset=["date"]).reset_index(drop=True)
        logger.info("日期解析后丢弃无法解析的行 %d 条", before - len(mapped))

    return drop_duplicates(mapped, subset=["date", "question", "answer"])


def read_excel_smart(
    source: Any, sheet_name: str | int | None = None
) -> tuple[pd.DataFrame, str]:
    """读取 Excel 并自动选择一个能被识别出数据模式的 sheet。

    背景：用户上传的 Excel 可能把「年度汇总」等无法识别的 sheet 放在第一个位置，
    直接 ``pd.read_excel`` 读第一个 sheet 会读到无情绪词列的汇总表。本函数遍历所有
    sheet，优先返回第一个能被 :func:`detect_mode` 识别的 sheet。

    策略：
    1. 显式指定 ``sheet_name`` 时直接读取该 sheet。
    2. 否则先读第一个 sheet：若 ``detect_mode != MODE_UNKNOWN`` 直接返回。
    3. 第一个 sheet 无法识别时，依次尝试其余 sheet，返回第一个可识别的 sheet。
    4. 全部都无法识别时返回第一个 sheet 的内容，由上层统一提示「无法识别」。

    参数:
        source: 文件路径、文件对象或 bytes（传给 ``pd.ExcelFile``）。
        sheet_name: 可选，显式指定要读取的 sheet 名或序号。

    返回:
        ``(DataFrame, 实际读取的 sheet 名)``。

    异常:
        CleanerError: 文件无法打开，或工作簿内没有任何 sheet。
    """
    try:
        xls = pd.ExcelFile(source)
    except Exception as exc:  # noqa: BLE001 —— 明确记录并抛出
        raise CleanerError(f"无法打开 Excel 文件：{exc}") from exc

    sheet_names: list[str] = [str(name) for name in xls.sheet_names]
    if not sheet_names:
        raise CleanerError("Excel 文件中没有任何工作表")

    if sheet_name is not None:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        return df, str(sheet_name)

    first = pd.read_excel(xls, sheet_name=sheet_names[0])
    if detect_mode(first) != MODE_UNKNOWN:
        return first, sheet_names[0]

    for name in sheet_names[1:]:
        candidate = pd.read_excel(xls, sheet_name=name)
        if detect_mode(candidate) != MODE_UNKNOWN:
            logger.info(
                "上传 Excel 默认 sheet「%s」无法识别，已自动切换为 sheet「%s」",
                sheet_names[0], name,
            )
            return candidate, name

    logger.warning("上传 Excel 所有 sheet 均无法识别出数据模式，返回第一个 sheet 供上层提示")
    return first, sheet_names[0]
