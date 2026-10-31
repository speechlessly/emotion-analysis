# -*- coding: utf-8 -*-
"""Streamlit 主界面（阶段 5）。

把数据获取 → 清洗 → 情绪分析 → 可视化 → 结论草稿 → 导出串成一体化交互式仪表盘。

安全约定：
- Cookie 为选填项：仅在「在线爬取」模式下、勾选「尝试在线爬取」且填写 Cookie 时才发起请求；
  Cookie 明文不进 session_state、不写日志、不写入导出文件（session_state 仅存「是否已填」布尔）。
- 在线爬取失败自动回退本地缓存，并在界面提示回退原因。
- 所有图表用 st.pyplot / st.plotly_chart 展示 visualization 返回的 Figure，不重复绘图。
"""
from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

import config
import cleaner
import data_source
import exporter
import market_data
import sentiment
from analysis import (
    aggregate_daily,
    correlate,
    event_window_compare,
    linear_fit,
    prepare_xy,
)
from cleaner import (
    COMMENT_COLUMN_ALIASES,
    MODE_AMBIGUOUS,
    MODE_ANALYZED,
    MODE_RAW,
    MODE_UNKNOWN,
    detect_mode_detail,
    find_column,
)
from data_source import (
    MODE_CACHE,
    MODE_DEMO,
    MODE_ONLINE,
    MODE_UPLOAD,
    ORIGIN_CACHE,
    ORIGIN_DEMO,
    ORIGIN_ONLINE,
    ORIGIN_UPLOAD,
    get_company_options,
    get_data,
    resolve_company,
)
from report_generator import generate_report
from visualization import (
    make_correlation_scatter,
    make_event_window_bar,
    make_market_emotion_dual_axis,
    make_monthly_trend,
    make_wordcloud,
    make_wordcloud_grid,
)

logger = logging.getLogger(__name__)

config.setup_logging()

st.set_page_config(page_title="投资者互动情绪监测仪表盘", page_icon="📊", layout="wide")


# ---------------------------------------------------------------------------
# 缓存函数（key 含 company/source 或词典路径，不含 Cookie）
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_lexicon_cached(path: str) -> dict[str, set[str]]:
    """缓存词典加载（key 为词典路径）。"""
    return sentiment.load_lexicon(path)


@st.cache_data(show_spinner=False)
def compute_raw_emotion(texts: tuple[str, ...], lexicon_path: str) -> pd.DataFrame:
    """原始模式：对文本列表做词典匹配与占比计算，结果缓存。"""
    lexicon = load_lexicon_cached(lexicon_path)
    rows: list[dict[str, Any]] = []
    for text in texts:
        pos, neg = sentiment.compute_ratios(text, lexicon)
        hits = sentiment.match_emotions(text, lexicon)
        rows.append(
            {
                "pos_ratio": pos,
                "neg_ratio": neg,
                "neg_annual_words": "、".join(hits["neg_annual"]),
                "pos_annual_words": "、".join(hits["pos_annual"]),
                "neg_social_words": "、".join(hits["neg_social"]),
                "pos_social_words": "、".join(hits["pos_social"]),
            }
        )
    return pd.DataFrame(rows)


def _read_uploaded_df(uploaded_file: Any) -> tuple[pd.DataFrame, str]:
    """读取上传文件（Excel/CSV），返回 ``(DataFrame, sheet名)``；CSV 的 sheet 名为空。

    Excel 会自动选择一个能被识别出数据模式的 sheet（跳过无法识别的汇总 sheet）。
    """
    name = getattr(uploaded_file, "name", "") or ""
    if name.lower().endswith((".csv", ".tsv")):
        return pd.read_csv(uploaded_file, encoding="utf-8-sig"), ""
    return cleaner.read_excel_smart(uploaded_file)


def read_annual_report_text(uploaded_file: Any) -> str:
    """读取上传的年报文章原文，支持 .txt / .docx / .pdf。

    读取失败或格式不支持时返回空字符串（上层据此提示「未上传 / 无法解析」）。
    """
    name = getattr(uploaded_file, "name", "") or ""
    lower = name.lower()
    try:
        if lower.endswith(".txt"):
            return uploaded_file.getvalue().decode("utf-8", errors="ignore")

        # 先统一读取为 bytes 再用 BytesIO 包装：部分解析库（zipfile/pdfplumber）
        # 要求 file-like 对象具备 seekable() 等方法，直接传上传对象在部分环境会失败。
        import io

        data = (
            uploaded_file.getvalue()
            if hasattr(uploaded_file, "getvalue")
            else uploaded_file.read()
        )

        if lower.endswith(".docx"):
            import docx

            doc = docx.Document(io.BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs if p.text and p.text.strip())
        if lower.endswith(".pdf"):
            import pdfplumber

            parts: list[str] = []
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    if text:
                        parts.append(text)
            return "\n".join(parts)
    except Exception as exc:  # noqa: BLE001 —— 界面提示，不抛异常
        logger.warning("读取年报文章失败：%s", exc)
        return ""
    return ""


def _report_image_or_none(relative_dir: str, filename: str) -> Path | None:
    """返回 assets 下报告图片路径；不存在时返回 None（界面提示而不报错）。"""
    path = config.ASSET_DIR / relative_dir / filename
    return path if path.exists() else None


def render_report_corr_demo() -> None:
    """演示模式：嵌入报告图 4.8/4.9/4.10 与报告结论卡（与报告演示口径一致）。"""
    rd = "report_corr"
    fig48 = _report_image_or_none(rd, "PE与情绪走势图_图4.8.png")
    fig49 = _report_image_or_none(rd, "正向情绪PE散点_图4.9.png")
    fig410 = _report_image_or_none(rd, "负向情绪PE散点_图4.10.png")

    st.info(
        "**报告结论（原文）**：正向情绪与市盈率相关性较弱（图 4.9，R²=0.0518，解释力约 5.18%）；"
        "负向情绪与市盈率相关性较大（图 4.10，剔除无关值后 R²=0.9895），"
        "且负向情绪发生时点对应的 PE 均低于 2021-2024 年均市盈率 26.72，"
        "反映投资者对市场的信心相对不足、盈利预期较为保守。"
    )

    if fig48 is not None:
        st.markdown("###### 图 4.8　2021-2024 海康威视市盈率及情绪波动图（报告原图）")
        st.image(str(fig48), use_container_width=True)
    else:
        st.warning("未找到报告图 4.8（assets/report_corr/PE与情绪走势图_图4.8.png）。")

    c1, c2 = st.columns(2)
    with c1:
        if fig49 is not None:
            st.markdown("###### 图 4.9　PE 与正向情绪相关性（报告原图，R²=0.0518）")
            st.image(str(fig49), use_container_width=True)
        else:
            st.warning("未找到报告图 4.9。")
    with c2:
        if fig410 is not None:
            st.markdown("###### 图 4.10　PE 与负向情绪相关性（报告原图，剔除后 R²=0.9895）")
            st.image(str(fig410), use_container_width=True)
        else:
            st.warning("未找到报告图 4.10。")

    with st.expander("口径说明：报告数字与下方复算结果的关系", expanded=False):
        st.markdown(
            "报告图 4.10 的 R²=0.9895 为「剔除无关值后」的筛选结果；"
            "下方复算是课程管线对**全样本**（未剔除无关值）的初始数据计算，属于进一步处理前的基准口径。"
            "在实际分析中，可根据爬取到的数据做进一步清洗处理后，再上传文件重新分析：\n\n"
            "- 正向-PE 不显著，与报告图 4.9「解释力较低」的结论方向一致；\n"
            "- 负向-PE 呈显著负相关（p<0.05），与报告「负向情绪与市盈率相关性较大」的核心结论方向一致。"
        )


def render_corr_metrics(result: Any, label: str) -> None:
    """用 metric / caption / warning / info 展示相关性统计指标（不藏在图注里）。

    参数:
        result: ``analysis.CorrelationResult`` 或 None。
        label: 指标中文名（如「情绪-股价」「情绪-PE」）。

    展示规则:
        - n、r、R² 用 ``st.metric``；p 值用 ``st.caption``。
        - n<30 → ``result.warning`` 用 ``st.warning``。
        - R²<0.1 → ``result.caution`` 用 ``st.info``。
    """
    if result is None:
        return
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{label} · 样本量 n", result.n)
    c2.metric(
        f"{label} · Pearson r",
        f"{result.r:.4f}" if result.r is not None else "—",
    )
    c3.metric(
        f"{label} · R²",
        f"{result.r_squared:.4f}" if result.r_squared is not None else "—",
    )
    if result.p is not None:
        st.caption(f"{label} · p 值 = {result.p:.4f}")
    if result.warning:
        st.warning(result.warning)
    if result.caution:
        st.info(result.caution)


def render_field_mapping(df: pd.DataFrame, mode: str) -> tuple[dict[str, Any], str]:
    """根据数据模式在侧边栏渲染字段映射下拉框，返回 ``(映射字典, 最终模式)``。

    原始模式返回 ``{company/date/question/answer: 源列}``；
    已分析模式返回 ``{date, emotion_columns, ratio_columns}``（含嵌套映射）。
    无法识别模式时给出错误提示并返回空映射（不抛异常）。
    MODE_AMBIGUOUS 时用 radio 让用户选择处理方式，并把选择写入返回值中的 mode。
    """
    mapping: dict[str, Any] = {}
    columns = [str(c) for c in df.columns]

    if mode == MODE_AMBIGUOUS:
        # 修复：已分析模式选项放在前面（默认优先按已分析模式处理），
        # 并把用户选择赋给 mode 随返回值一起带出，供主流程使用。
        choice = st.radio(
            "检测到数据同时含提问列与情绪词列，请选择处理方式",
            [MODE_ANALYZED, MODE_RAW],
            key="mode_choice",
        )
        mode = choice

    if mode == MODE_RAW:
        st.markdown("**字段映射（原始模式）**")
        for standard in ("company", "date", "question", "answer"):
            aliases = COMMENT_COLUMN_ALIASES[standard]
            auto = find_column(df, aliases)
            default_index = columns.index(auto) if auto in columns else 0
            src = st.selectbox(
                f"{standard}", columns, index=default_index, key=f"map_{standard}"
            )
            mapping[standard] = src

    elif mode == MODE_ANALYZED:
        st.markdown("**字段映射（已分析模式）**")
        # 修复（A1/A2/A3）：占位符放在选项末尾；自动匹配到列时用
        # index=columns.index(auto) 精确选中该列；匹配失败时落到占位符并给出 warning；
        # 选中占位符的字段在写入 mapping 前转为 None，从而在传给后端时排除占位符。
        placeholder = "（未自动匹配，请手动选择）"
        options = columns + [placeholder]

        auto_emo = sentiment.auto_map_emotion_columns(df)
        auto_ratio = sentiment.auto_map_ratio_columns(df)

        emo_labels = (
            ("neg_annual_words", "年报负面情绪词"),
            ("pos_annual_words", "年报正面情绪词"),
            ("neg_social_words", "社媒负面情绪词"),
            ("pos_social_words", "社媒正面情绪词"),
        )
        ratio_labels = (("pos_ratio", "正向占比"), ("neg_ratio", "负向占比"))

        # 自动匹配失败时给出明确提示（A2）
        missing = [
            label for standard, label in emo_labels if auto_emo.get(standard) not in columns
        ] + [
            label for standard, label in ratio_labels if auto_ratio.get(standard) not in columns
        ]
        if missing:
            st.warning(
                "自动匹配失败，请手动选择四类情绪词列，否则词云将为空！"
                + "（未匹配字段：" + "、".join(missing) + "）"
            )

        # 日期列
        auto_date = find_column(df, COMMENT_COLUMN_ALIASES["date"])
        date_choice = st.selectbox(
            "date（时间列）",
            options,
            index=columns.index(auto_date) if auto_date in columns else len(columns),
            key="map_date",
        )
        mapping["date"] = None if date_choice == placeholder else date_choice

        # 四类情绪词列（自动匹配失败时仍可手动指定）
        emo_map: dict[str, str | None] = {}
        st.caption("四类情绪词列")
        for standard, label in emo_labels:
            auto = auto_emo.get(standard)
            choice = st.selectbox(
                label,
                options,
                index=columns.index(auto) if auto in columns else len(columns),
                key=f"map_{standard}",
            )
            emo_map[standard] = None if choice == placeholder else choice
        mapping["emotion_columns"] = emo_map

        # 正负占比列
        ratio_map: dict[str, str | None] = {}
        st.caption("正负占比列")
        for standard, label in ratio_labels:
            auto = auto_ratio.get(standard)
            choice = st.selectbox(
                label,
                options,
                index=columns.index(auto) if auto in columns else len(columns),
                key=f"map_{standard}",
            )
            ratio_map[standard] = None if choice == placeholder else choice
        mapping["ratio_columns"] = ratio_map

    else:
        # MODE_UNKNOWN 等无法识别的情况：友好提示，不抛异常
        st.error(
            "未识别到数据模式，请上传包含【时间+提问】或【时间+四类情绪词】的 Excel 表"
        )

    # 修复：返回值增加最终 mode，供调用方拿到 MODE_AMBIGUOUS 分支的用户选择。
    return mapping, mode


def resolve_mode_for_origin(
    df: pd.DataFrame,
    data_origin: str,
    mode_preview: str | None = None,
) -> tuple[str, str]:
    """按实际数据来源解析最终分析模式（封杀 answer 参与情绪计算）。

    - 在线爬取：强制原始模式，只分析「提问」文本。
    - 本地缓存：含四类情绪词列 → 已分析模式（直接读取占比）；
                纯文本 → 原始模式（只分析提问）。
    - 上传文件：保持原有 detect_mode + 手动选择逻辑。

    参数:
        df: 数据源返回的原始 DataFrame。
        data_origin: 实际数据来源（在线 / 缓存 / 上传）。
        mode_preview: 上传模式下界面初判/手动选择的模式（可为 None）。

    返回:
        ``(模式常量, 中文理由)``。
    """
    if data_origin == ORIGIN_DEMO:
        return cleaner.MODE_ANALYZED, "演示模式：直接读取四类情绪词表（已分析数据），词云展示报告图 4.1~4.4"

    if data_origin == ORIGIN_ONLINE:
        return cleaner.MODE_RAW, "在线爬取结果，强制按原始模式只分析「提问」文本"

    if data_origin == ORIGIN_CACHE:
        if cleaner.has_emotion_columns(df):
            return cleaner.MODE_ANALYZED, "本地缓存含四类情绪词列，按已分析模式直接读取占比"
        if cleaner.find_column(df, cleaner.COMMENT_COLUMN_ALIASES["question"]) is not None:
            return cleaner.MODE_RAW, "本地缓存为纯文本，按原始模式只分析「提问」文本"
        return cleaner.MODE_UNKNOWN, "本地缓存既无四类情绪词列，也无提问列，无法分析"

    # 上传文件：保持原有逻辑
    return cleaner.finalize_mode(df, mode_preview)


def build_emotion_detail_raw(
    df: pd.DataFrame,
    mapping: dict[str, str],
    company_default: str,
    missing_question_message: str = "当前数据缺失提问列，无法分析投资者情绪！",
) -> pd.DataFrame:
    """原始模式：清洗 → 词典匹配 → 组装情绪明细表。

    只接收 question 提问列的文本，**绝不拼接或使用 answer 回答列**。
    question 为空的行直接丢弃；整个表 question 全为空时调用 ``st.error`` + ``st.stop()``。
    找不到目标列或清洗后无有效数据时返回空表（含标准列），由上层提示，不抛 KeyError。
    """
    detail_cols = [
        "date", "pos_ratio", "neg_ratio",
        "neg_annual_words", "pos_annual_words", "neg_social_words", "pos_social_words",
    ]
    cleaned = cleaner.clean_raw_comments(df, mapping=mapping, company_default=company_default)
    if cleaned is None or cleaned.empty or "date" not in cleaned.columns:
        logger.warning("原始模式清洗后无有效数据（缺 date 列或为空），返回空明细表")
        return pd.DataFrame(columns=detail_cols)

    # 原始模式强制只使用 question 列，绝不拼接 answer。
    if "question" not in cleaned.columns:
        st.error(missing_question_message)
        st.stop()

    # 丢弃 question 为空的行（不拿 answer 充数）。
    question_series = cleaned["question"].fillna("").astype(str).str.strip()
    cleaned = cleaned[question_series.ne("")].reset_index(drop=True)

    # 整个表的 question 全为空：无法分析，直接终止。
    if cleaned.empty:
        st.error(missing_question_message)
        st.stop()

    texts: list[str] = []
    for _, row in cleaned.iterrows():
        texts.append(str(row["question"]).strip())

    emotion = compute_raw_emotion(tuple(texts), str(config.DEFAULT_DICT_PATH))
    if emotion is None or emotion.empty or "pos_ratio" not in emotion.columns:
        logger.warning("原始模式情绪匹配结果为空（缺 pos_ratio 列），返回空明细表")
        return pd.DataFrame(columns=detail_cols)

    detail = pd.DataFrame(
        {"date": pd.to_datetime(cleaned["date"], errors="coerce").dt.strftime("%Y-%m-%d").reset_index(drop=True)}
    )
    for col in (
        "pos_ratio",
        "neg_ratio",
        "neg_annual_words",
        "pos_annual_words",
        "neg_social_words",
        "pos_social_words",
    ):
        if col in emotion.columns:
            detail[col] = emotion[col].reset_index(drop=True)
        else:
            detail[col] = ""
    return detail


def build_emotion_detail_analyzed(df: pd.DataFrame, mapping: dict[str, Any]) -> pd.DataFrame:
    """已分析模式：跳过匹配，直接解析四类情绪词列与占比列（支持界面手动映射）。

    列解析失败或日期列缺失时返回空表（含标准列），由上层提示，不抛 ValueError/KeyError。
    """
    detail_cols = [
        "date", "pos_ratio", "neg_ratio",
        "neg_annual_words", "pos_annual_words", "neg_social_words", "pos_social_words",
    ]
    emotion_map = mapping.get("emotion_columns") if mapping else None
    ratio_map = mapping.get("ratio_columns") if mapping else None

    try:
        words_df = sentiment.parse_emotion_columns(df, column_map=emotion_map)
        ratios_df = sentiment.parse_ratio_columns(df, column_map=ratio_map)
    except ValueError as exc:
        logger.warning("已分析模式情绪词/占比列解析失败：%s", exc)
        return pd.DataFrame(columns=detail_cols)

    date_col = (mapping or {}).get("date") or find_column(df, COMMENT_COLUMN_ALIASES["date"])
    if date_col is None or date_col not in df.columns:
        logger.warning("已分析模式未找到日期列，返回空明细表")
        return pd.DataFrame(columns=detail_cols)

    # 修复：上传表未提供占比列时，parse_ratio_columns 会把正负占比都补成 0，
    # 导致后续 pearsonr 得 nan、PE 相关性失效。此时改用四类情绪词的词数占比推算。
    if (
        ratios_df["pos_ratio"].eq(0.0).all()
        and ratios_df["neg_ratio"].eq(0.0).all()
    ):
        logger.info("上传表未提供有效占比列（正负占比均为 0），改用四类情绪词数占比推算")
        ratios_df = sentiment.derive_ratios_from_word_lists(words_df)

    detail = pd.DataFrame({"date": df[date_col].astype(str).reset_index(drop=True)})
    detail["pos_ratio"] = ratios_df["pos_ratio"].reset_index(drop=True)
    detail["neg_ratio"] = ratios_df["neg_ratio"].reset_index(drop=True)
    for standard in (
        "neg_annual_words",
        "pos_annual_words",
        "neg_social_words",
        "pos_social_words",
    ):
        detail[standard] = words_df[standard].apply("、".join).reset_index(drop=True)
    return detail


def build_report_data(s: Any) -> dict[str, Any]:
    """从 session_state 组装报告所需的汇总 dict。"""
    daily = s["daily"]
    word_freq = s["word_freq"]
    top_pos = (
        word_freq[word_freq["category"].isin(["pos_annual_words", "pos_social_words"])]
        .sort_values("count", ascending=False)["word"].drop_duplicates().head(8).tolist()
        if word_freq is not None and not word_freq.empty
        else []
    )
    top_neg = (
        word_freq[word_freq["category"].isin(["neg_annual_words", "neg_social_words"])]
        .sort_values("count", ascending=False)["word"].drop_duplicates().head(8).tolist()
        if word_freq is not None and not word_freq.empty
        else []
    )
    mkt = s.get("mkt_result")
    return {
        "company": s.get("company"),
        "code": s.get("code"),
        "market": s.get("market"),
        "data_origin": s.get("data_origin"),
        "mode": s.get("mode"),
        "n_comments": len(s.get("raw_df", pd.DataFrame())),
        "date_start": daily["date"].min() if daily is not None and not daily.empty else None,
        "date_end": daily["date"].max() if daily is not None and not daily.empty else None,
        "n_days": len(daily) if daily is not None else None,
        "pos_ratio_mean": float(daily["pos_ratio"].mean()) if daily is not None and not daily.empty else None,
        "neg_ratio_mean": float(daily["neg_ratio"].mean()) if daily is not None and not daily.empty else None,
        "top_pos_words": top_pos,
        "top_neg_words": top_neg,
        "event_window": s.get("event_window"),
        "corr_close": s.get("corr_close"),
        "corr_pe": s.get("corr_pe"),
        "has_market": bool(mkt is not None and not mkt.empty),
        "has_pe": bool(mkt is not None and getattr(mkt, "has_pe", False)),
    }


# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("参数设置")

    # 年报文章输入（可选）：上传文件或粘贴原文，用于生成年报正/负面词云。
    annual_file = None
    annual_text_pasted = ""
    # 行情表输入（可选）：无网络 / 无内置样例时，上传含 date/close/pe 的行情表补充 PE。
    market_file = None

    source_mode = st.radio(
        "数据源模式",
        [MODE_DEMO, MODE_ONLINE, MODE_CACHE, MODE_UPLOAD],
        index=0,
        help=(
            "演示模式：读 data/demo 四类情绪词表，词云直接展示报告图 4.1~4.4；"
            "在线爬取 / 上传文件：提问 × 社媒词典、年报 × 年报词典，按报告方式处理。"
        ),
    )

    load_platform = st.checkbox("尝试加载平台公司列表", value=False)
    company_options = get_company_options(include_platform=load_platform)
    selected = st.selectbox("选择公司（预设/缓存）", [""] + company_options)
    manual = st.text_input("或手动输入公司名", value=selected or "")
    company = (manual or selected).strip()

    uploaded_file = None
    mode_preview = None
    mapping: dict[str, str] = {}

    if source_mode == MODE_UPLOAD:
        uploaded_file = st.file_uploader(
            "上传数据文件（Excel/CSV）", type=["xlsx", "xls", "csv"]
        )
        if uploaded_file is not None:
            try:
                uploaded_df, uploaded_sheet = _read_uploaded_df(uploaded_file)
                mode_detail_preview = detect_mode_detail(uploaded_df)
                mode_preview = mode_detail_preview["mode"]
                st.info(
                    f"识别到数据类型：{mode_preview}　"
                    f"（依据：{mode_detail_preview['reason']}）"
                )
                if uploaded_sheet:
                    st.caption(f"已自动读取 Sheet「{uploaded_sheet}」")
                # 修复：同时接收 render_field_mapping 返回的最终 mode
                # （MODE_AMBIGUOUS 时为用户 radio 选择，否则维持原 mode_preview）。
                mapping, mode_preview = render_field_mapping(uploaded_df, mode_preview)
            except Exception as exc:  # noqa: BLE001 —— 界面提示
                st.error(f"读取上传文件失败：{exc}")

    cookie_str = None
    page_range: tuple[int, int] = (1, 1)
    if source_mode == MODE_ONLINE:
        attempt_online = st.checkbox("尝试在线爬取", value=False)
        if attempt_online:
            cookie_str = st.text_input("Cookie（选填）", type="password", help="仅本次会话使用")
        page_start = st.number_input("起始页", min_value=1, max_value=100, value=1)
        page_end = st.number_input("结束页", min_value=1, max_value=100, value=1)
        if page_end < page_start:
            st.warning("结束页应大于等于起始页")
        page_range = (int(page_start), int(page_end))

    meta = resolve_company(company) if company else {}
    event_date = st.text_input(
        "事件日期", value=meta.get("event_date", ""), help="分拆上市等事件日，YYYY-MM-DD"
    )
    window_days = st.number_input(
        "事件窗口天数", min_value=7, max_value=730, value=config.DEFAULT_WINDOW_DAYS
    )
    # 情绪指标选择：正向 / 负向占比（负向可产出报告核心的「负向情绪 vs PE」分析）
    emotion_metric = st.selectbox("情绪指标", ["正向占比", "负向占比"], index=0)
    emotion_col = "pos_ratio" if emotion_metric == "正向占比" else "neg_ratio"

    # 年报文章（可选）：上传文件或粘贴原文，用于生成年报正/负面词云。
    with st.expander("年报文章（可选，生成年报词云）"):
        annual_file = st.file_uploader(
            "上传年报（.txt/.docx/.pdf）", type=["txt", "docx", "pdf"]
        )
        annual_text_pasted = st.text_area(
            "或直接粘贴年报原文", height=120,
            help="留空且未上传文件时，将无法生成年报词云（仅生成社媒提问词云）。",
        )

    # 行情表（可选）：当 akshare / 内置样例取不到 PE 时，用本表补充 PE。
    with st.expander("行情表（可选，补充 PE）"):
        market_file = st.file_uploader(
            "上传行情表（date/close/pe）", type=["xlsx", "xls", "csv"],
            help="可选。海康威视演示已内置真实 PE，无需上传；爬取其他公司且取不到 PE 时可上传。",
        )

    run_clicked = st.button("运行分析", type="primary", use_container_width=True)
    st.caption("Cookie 仅在「在线爬取」且勾选「尝试在线爬取」时使用；不写日志、不导出。")


# ---------------------------------------------------------------------------
# 运行按钮：执行完整流程，结果写入 session_state
# ---------------------------------------------------------------------------
if run_clicked:
    if not company:
        st.error("请先选择或输入公司名。")
    else:
        meta = resolve_company(company)
        code = meta["code"]
        market_code = meta["market"]
        st.session_state["cookie_filled"] = bool(cookie_str)

        # 严格隔离数据源：仅「上传文件」模式才允许传入上传文件；
        # 在线爬取 / 本地缓存模式强制传 None，杜绝读到残留的上传文件。
        effective_uploaded = uploaded_file if source_mode == MODE_UPLOAD else None

        with st.spinner("正在获取评论数据..."):
            try:
                data_result = get_data(
                    company=company,
                    source_mode=source_mode,
                    uploaded_file=effective_uploaded,
                    cookie_str=cookie_str,
                    page_range=page_range,
                    code=code,
                    market=market_code,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"数据获取失败：{exc}")
                st.stop()

        raw_df = data_result.df
        if raw_df is None or raw_df.empty:
            st.error("获取到的数据为空，请检查数据源或上传文件。")
            st.stop()

        # 按实际数据来源解析最终模式：
        # - 在线爬取强制原始模式（只分析「提问」文本）；
        # - 本地缓存优先识别已分析模式（有四类情绪词列直接读占比），纯文本才走原始；
        # - 上传文件保持 detect_mode + 手动选择逻辑。
        mode, mode_reason = resolve_mode_for_origin(raw_df, data_result.data_origin, mode_preview)

        # 无法识别数据模式时立即终止，绝不再往下走，避免红屏 KeyError/ValueError。
        if mode == MODE_UNKNOWN:
            st.error(
                "未能识别数据模式：请确保表格包含【时间】以及【提问】或【四类情绪词】列。"
            )
            st.stop()

        with st.spinner("正在清洗与情绪分析..."):
            # 修复：改为明确三分支，避免 MODE_AMBIGUOUS / MODE_UNKNOWN 误落进原始模式
            # （原始模式在 question 全空时会报「缺失提问列」并停止）。
            if mode == MODE_ANALYZED:
                detail = build_emotion_detail_analyzed(raw_df, mapping)
            elif mode == MODE_RAW:
                missing_msg = (
                    "未抓到提问内容，无法分析投资者情绪！"
                    if data_result.data_origin == ORIGIN_ONLINE
                    else "当前数据缺失提问列，无法分析投资者情绪！"
                )
                detail = build_emotion_detail_raw(
                    raw_df, mapping, company, missing_question_message=missing_msg
                )
            else:
                st.error(
                    f"当前数据模式为「{mode}」，既无可用提问内容，也未识别到四类情绪词列。"
                    "请检查上传表是否包含【提问】或【四类情绪词】列。"
                )
                st.stop()

            # 字段映射错误或数据不含目标列时，build_* 会返回空明细表，此时友好提示而非红屏。
            if detail is None or detail.empty or "pos_ratio" not in detail.columns:
                st.error("情绪分析未产出有效数据：请检查字段映射（时间/情绪词/占比列）是否正确。")
                st.stop()

            daily = aggregate_daily(detail[["date", "pos_ratio", "neg_ratio"]])
            monthly = sentiment.aggregate_by_month(detail[["date", "pos_ratio", "neg_ratio"]])
            # 词频直接来自情绪明细表的四类词列，不对「回答信息」重新分词。
            try:
                word_freq = sentiment.word_frequency_from_detail(detail)
            except (ValueError, KeyError) as exc:  # noqa: BLE001 —— 界面友好提示
                logger.exception("从情绪明细表统计词频失败")
                st.error(f"解析情绪词列失败：{exc}，请检查字段映射是否正确。")
                st.stop()

            # 年报词云：读取上传/粘贴的年报原文，用年报词典匹配统计词频。
            # 年报正/负面词云独立于社媒词云，避免用「提问文本命中的年报词」冒充年报原文。
            annual_text = ""
            if annual_file is not None:
                annual_text = read_annual_report_text(annual_file)
            if not annual_text and annual_text_pasted.strip():
                annual_text = annual_text_pasted.strip()

            annual_word_freq = pd.DataFrame(columns=["word", "count", "category"])
            if annual_text:
                lexicon = load_lexicon_cached(str(config.DEFAULT_DICT_PATH))
                annual_word_freq = sentiment.word_frequency_from_texts(
                    [annual_text], lexicon, ["pos_annual", "neg_annual"]
                )

        with st.spinner("正在获取行情并计算相关性..."):
            # 修复：把情绪日期范围透传给行情获取，避免行情窗口写死导致与评论日期无交集。
            # daily 已在上一段「清洗与情绪分析」中算出，此处可直接取其日期 min/max。
            start = daily["date"].min().replace("-", "") if not daily.empty else None
            end = daily["date"].max().replace("-", "") if not daily.empty else None
            mkt_result = market_data.get_market(
                code, market=market_code, start_date=start, end_date=end,
                uploaded_file=market_file,
            )
            corr_close = None
            corr_pe = None
            fit_close = None
            fit_pe = None
            if not mkt_result.empty:
                corr_close = correlate(daily, mkt_result.df, target="close", emotion_col=emotion_col)
                if mkt_result.has_pe:
                    corr_pe = correlate(daily, mkt_result.df, target="pe", emotion_col=emotion_col)
                if corr_close is not None and corr_close.valid:
                    x, y, _ = prepare_xy(daily, mkt_result.df, target="close", emotion_col=emotion_col)
                    fit_close = linear_fit(x, y)
                if corr_pe is not None and corr_pe.valid:
                    xp, yp, _ = prepare_xy(daily, mkt_result.df, target="pe", emotion_col=emotion_col)
                    fit_pe = linear_fit(xp, yp)

        ew = event_window_compare(daily, event_date, window_days) if event_date else None

        st.session_state.update(
            {
                "company": company,
                "code": code,
                "market": market_code,
                "data_origin": data_result.data_origin,
                "data_message": data_result.message,
                "mode": mode,
                "mode_reason": mode_reason,
                "raw_df": raw_df,
                "detail": detail,
                "daily": daily,
                "monthly": monthly,
                "word_freq": word_freq,
                "mkt_result": mkt_result,
                "corr_close": corr_close,
                "corr_pe": corr_pe,
                "fit_close": fit_close,
                "fit_pe": fit_pe,
                "event_window": ew,
                "event_date": event_date,
                "window_days": window_days,
                "emotion_col": emotion_col,
                "annual_text": annual_text,
                "annual_word_freq": annual_word_freq,
            }
        )
        if "report_md" in st.session_state:
            del st.session_state["report_md"]
        st.success(f"分析完成。{data_result.message}")


# ---------------------------------------------------------------------------
# 主区域 Tabs
# ---------------------------------------------------------------------------
has_result = "raw_df" in st.session_state

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["数据预览", "情绪分析", "市场关联", "结论草稿", "导出", "说明"]
)

# ---------------- Tab1 数据预览 ----------------
with tab1:
    if not has_result:
        st.info("请在左侧配置参数并点击「运行分析」。")
    else:
        st.subheader("数据预览")
        st.markdown(f"**数据来源**：`{st.session_state['data_origin']}`")
        st.markdown(f"**识别模式**：`{st.session_state['mode']}`")
        st.markdown(f"**模式识别依据**：{st.session_state.get('mode_reason', '')}")
        if st.session_state.get("data_message"):
            st.info(st.session_state["data_message"])

        raw_df = st.session_state["raw_df"]
        detail = st.session_state["detail"]
        st.markdown(
            f"原始行数 {len(raw_df)}，清洗/解析后有效行数 {len(detail)}，"
            f"去重/丢弃 {max(0, len(raw_df) - len(detail))} 行。"
        )
        st.dataframe(raw_df.head(config.PREVIEW_ROWS), use_container_width=True)

# ---------------- Tab2 情绪分析 ----------------
with tab2:
    if not has_result:
        st.info("请先运行分析。")
    else:
        st.subheader("情绪分析")
        st.plotly_chart(
            make_monthly_trend(st.session_state["monthly"], st.session_state.get("event_date")),
            use_container_width=True,
        )
        # 修复：已分析模式下词频为空时，提示用户检查字段映射，而不是只显示一张空图。
        word_freq = st.session_state["word_freq"]
        if (word_freq is None or word_freq.empty) and st.session_state.get("mode") == MODE_ANALYZED:
            st.info(
                "未从表中解析到情绪词。请检查左侧侧边栏的「四类情绪词列」字段映射是否已手动选中正确的列。"
            )
        # 词云：演示模式直接展示报告图 4.1~4.4；爬取/上传按
        # 「社媒提问 × 社媒词典、年报 × 年报词典」动态生成（与报告逻辑一致）。
        if st.session_state.get("data_origin") == ORIGIN_DEMO:
            report_dir = config.ASSET_DIR / "report_wordclouds"
            st.markdown("#### 词云（报告图 4.1~4.4）")
            c1, c2 = st.columns(2)
            with c1:
                st.image(str(report_dir / "社媒正面词云.png"), caption="图 4.1 社媒正面词云", use_container_width=True)
                st.image(str(report_dir / "年报负面词云.png"), caption="图 4.3 年报负面词云", use_container_width=True)
            with c2:
                st.image(str(report_dir / "年报正面词云.png"), caption="图 4.2 年报正面词云", use_container_width=True)
                st.image(str(report_dir / "社媒负面词云.png"), caption="图 4.4 社媒负面词云", use_container_width=True)
        else:
            social_wf = (
                word_freq
                if word_freq is not None and not word_freq.empty
                else pd.DataFrame(columns=["word", "count", "category"])
            )

            st.markdown("#### 社媒词云（投资者提问）")
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(
                    make_wordcloud(
                        social_wf[social_wf["category"] == "pos_social_words"],
                        title="社媒正面词云", colormap="Reds",
                    ),
                    use_container_width=True,
                )
            with c2:
                st.pyplot(
                    make_wordcloud(
                        social_wf[social_wf["category"] == "neg_social_words"],
                        title="社媒负面词云", colormap="Blues",
                    ),
                    use_container_width=True,
                )

            st.markdown("#### 年报词云（上传年报）")
            annual_wf = st.session_state.get("annual_word_freq")
            if annual_wf is None or annual_wf.empty:
                st.info(
                    "未上传年报文章，无法生成年报词云。可在左侧「年报文章」处上传 "
                    ".txt/.docx/.pdf 或粘贴年报原文；上传后将按年报词典匹配生成年报正/负面词云。"
                )
            else:
                c3, c4 = st.columns(2)
                with c3:
                    st.pyplot(
                        make_wordcloud(
                            annual_wf[annual_wf["category"] == "pos_annual_words"],
                            title="年报正面词云", colormap="Reds",
                        ),
                        use_container_width=True,
                    )
                with c4:
                    st.pyplot(
                        make_wordcloud(
                            annual_wf[annual_wf["category"] == "neg_annual_words"],
                            title="年报负面词云", colormap="Blues",
                        ),
                        use_container_width=True,
                    )
        ew = st.session_state.get("event_window")
        if ew is not None and ew.valid:
            st.plotly_chart(make_event_window_bar(ew), use_container_width=True)
        else:
            st.info("未提供有效事件日，跳过事件前后对比。")

# ---------------- Tab3 市场关联 ----------------
with tab3:
    if not has_result:
        st.info("请先运行分析。")
    else:
        st.subheader("市场关联")
        st.caption(
            "提示：p 值较大（如 p>0.05）表示未达统计显著，R²<0.1 表示解释力有限，"
            "均属课程案例的客观真实情况，无需强行解释。"
        )
        mkt = st.session_state.get("mkt_result")
        if mkt is None or mkt.empty:
            st.warning("行情完全缺失：请提供行情数据（上传或安装 akshare）。")
        else:
            # 演示模式：先展示报告原图与结论（口径与报告一致），动态复算折叠为对照；
            # 其余模式：直接展示动态计算结果（contextlib.nullcontext 保持原行为）。
            is_demo = st.session_state.get("data_origin") == ORIGIN_DEMO
            if is_demo:
                render_report_corr_demo()
                body_ctx = st.expander(
                    "动态复算（课程管线真实计算：全样本、真实 PE，不含人工剔除）", expanded=False
                )
            else:
                body_ctx = contextlib.nullcontext()
            with body_ctx:
                st.markdown(f"行情来源：`{mkt.data_origin}`")
                if not mkt.has_pe:
                    st.info("当前数据源不含 PE，已降级为情绪-股价相关性。")

                daily = st.session_state["daily"]
                emotion_col = st.session_state.get("emotion_col", "pos_ratio")
                emo_label = "正向情绪" if emotion_col == "pos_ratio" else "负向情绪"
                st.plotly_chart(
                    make_market_emotion_dual_axis(
                        mkt.df, daily, st.session_state.get("event_date", ""), target="pe"
                    ),
                    use_container_width=True,
                )

                corr_close = st.session_state.get("corr_close")
                # 修复：改为四分支——valid=False 时把 message（含日期交集为空的双方范围）展示出来。
                if corr_close is None:
                    pass  # 行情整体缺失，上游已有提示
                elif not corr_close.valid:
                    st.warning(f"情绪-股价相关性未计算：{corr_close.message}")
                    st.caption(
                        f"情绪日期范围：{daily['date'].min()} ~ {daily['date'].max()}；"
                        f"行情日期范围：{mkt.df['date'].min()} ~ {mkt.df['date'].max()}"
                    )
                else:
                    render_corr_metrics(corr_close, f"{emo_label}-股价")
                    x, y, _ = prepare_xy(daily, mkt.df, target="close", emotion_col=emotion_col)
                    st.plotly_chart(
                        make_correlation_scatter(
                            x, y, result=corr_close, fit=st.session_state.get("fit_close"),
                            y_label="收盘价",
                        ),
                        use_container_width=True,
                    )

                corr_pe = st.session_state.get("corr_pe")
                # 修复：valid=False 时同样把 message 展示出来，而不是静默不显示。
                if corr_pe is None:
                    if mkt.has_pe:
                        st.info("PE 数据无效，未生成情绪-PE 相关性图。")
                elif not corr_pe.valid:
                    st.warning(f"情绪-PE 相关性未计算：{corr_pe.message}")
                else:
                    render_corr_metrics(corr_pe, f"{emo_label}-PE")
                    xp, yp, _ = prepare_xy(daily, mkt.df, target="pe", emotion_col=emotion_col)
                    st.plotly_chart(
                        make_correlation_scatter(
                            xp, yp, result=corr_pe, fit=st.session_state.get("fit_pe"),
                            y_label="PE",
                        ),
                        use_container_width=True,
                    )

# ---------------- Tab4 结论草稿 ----------------
with tab4:
    if not has_result:
        st.info("请先运行分析。")
    else:
        st.subheader("结论草稿")
        report_data = build_report_data(st.session_state)
        if "report_md" not in st.session_state:
            st.session_state["report_md"] = generate_report(report_data)
        if st.button("重新生成结论"):
            st.session_state["report_md"] = generate_report(report_data)
        edited = st.text_area("结论草稿（可编辑）", st.session_state["report_md"], height=420)
        st.session_state["report_md"] = edited

# ---------------- Tab5 导出 ----------------
with tab5:
    if not has_result:
        st.info("请先运行分析。")
    else:
        st.subheader("导出")
        company = st.session_state.get("company", "output")
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        base = exporter.safe_filename(company)

        report_data = build_report_data(st.session_state)
        report_md = st.session_state.get("report_md") or generate_report(report_data)

        figures: dict[str, Any] = {
            "四类词云": make_wordcloud_grid(st.session_state["word_freq"]),
            "月度趋势": make_monthly_trend(st.session_state["monthly"], st.session_state.get("event_date")),
        }
        ew = st.session_state.get("event_window")
        if ew is not None and ew.valid:
            figures["事件前后"] = make_event_window_bar(ew)
        mkt = st.session_state.get("mkt_result")
        if mkt is not None and not mkt.empty:
            figures["行情情绪双轴"] = make_market_emotion_dual_axis(
                mkt.df, st.session_state["daily"], st.session_state.get("event_date", ""), target="pe"
            )
            corr_close = st.session_state.get("corr_close")
            if corr_close is not None and corr_close.valid:
                x, y, _ = prepare_xy(
                    st.session_state["daily"], mkt.df, target="close",
                    emotion_col=st.session_state.get("emotion_col", "pos_ratio"),
                )
                figures["股价相关性"] = make_correlation_scatter(
                    x, y, result=corr_close, fit=st.session_state.get("fit_close"), y_label="收盘价"
                )

        result_dict = {
            "raw_df": st.session_state["raw_df"],
            "emotion_detail": st.session_state["detail"],
            "daily": st.session_state["daily"],
            "monthly": st.session_state["monthly"],
            "corr_close": st.session_state.get("corr_close"),
            "corr_pe": st.session_state.get("corr_pe"),
            "event_window": st.session_state.get("event_window"),
        }

        c1, c2, c3 = st.columns(3)
        with c1:
            excel_path = exporter.export_excel(result_dict, Path("/tmp") / f"{base}_{ts}.xlsx")
            with open(excel_path, "rb") as fh:
                st.download_button(
                    "下载 Excel", fh, file_name=excel_path.name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        with c2:
            md_path = exporter.export_markdown(report_md, Path("/tmp") / f"{base}_{ts}.md")
            with open(md_path, "rb") as fh:
                st.download_button("下载 Markdown", fh, file_name=md_path.name, mime="text/markdown")
        with c3:
            try:
                word_path = exporter.export_word(report_md, Path("/tmp") / f"{base}_{ts}.docx")
                with open(word_path, "rb") as fh:
                    st.download_button(
                        "下载 Word", fh, file_name=word_path.name,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Word 导出不可用：{exc}")

        st.divider()
        st.markdown("**图表导出**（PNG / HTML，文件名含公司名与时间戳）")
        try:
            fig_dir = exporter.export_figures(figures, Path("/tmp") / f"{base}_{ts}_figures", company=company)
            if fig_dir:
                for p in fig_dir:
                    with open(p, "rb") as fh:
                        st.download_button(
                            f"下载 {Path(p).name}", fh, file_name=Path(p).name,
                            mime="text/html" if p.suffix == ".html" else "image/png",
                        )
            else:
                st.info("无可导出图表")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"图表导出失败：{exc}")

        st.caption("所有导出文件均不含 Cookie 及学号姓名等敏感信息。")


# ---------------- Tab6 说明 ----------------
with tab6:
    st.subheader("合规与说明")
    st.markdown(
        """
### 数据覆盖限制
- 评论数据来自上证 e 互动平台（沪市）。**不承诺「平台全部公司可选」**。
- 海康威视为深市（002415.SZ）股票，其缓存数据系课程演示样例，不代表平台全量覆盖。
- 内置行情 `market_002415.csv` 的 `pe` 为**年度近似值，仅用于课程演示**，非当日真实市盈率。

### Cookie 与合规
- Cookie 为选填项，仅当「在线爬取」且勾选「尝试在线爬取」并填写 Cookie 时才发起请求。
- Cookie 不进日志、不进导出文件、不进 session_state 明文（仅存「是否已填」布尔）。
- 爬取已内置限速/重试/超时，请勿高频访问；所抓取内容版权归平台与发布者所有，仅供学习研究。

### AI 使用说明模板
> 本研究使用生成式 AI 工具辅助完成代码编写、数据清洗脚本生成与部分文档草稿撰写。
> AI 生成内容均已由作者逐项审阅、校验与修正；所有数据分析、结论推导与观点表述由作者独立完成。
> 研究引用的数据均来自公开渠道，不包含 AI 虚构或未经核实的数据。

### 局限
- 情绪占比按命中词次数归一化，同日多条简单平均，未按文本长度加权，仅作趋势参考。
- `n<30` 时相关性仅供参考；`R²<0.1` 时解释力有限，不宜作因果推断。
- 若显示 p 值较大（如 p>0.05），说明在统计学意义上未达到显著，该结果符合课程案例的客观真实情况，无需强行解释。
- 不做多元回归、格兰杰因果、滞后相关。
"""
    )
