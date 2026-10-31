# -*- coding: utf-8 -*-
"""阶段 2 测试用例（清洗 + 情绪分析双模式）。

覆盖点：
1. 原始模式：detect_mode 识别 + match_emotions 匹配 + compute_ratios 计算占比 + clean_raw_comments 清洗。
2. 已分析模式：detect_mode 识别 + parse_emotion_columns 拆分 + parse_ratio_columns 校验 + 词频统计。
3. 空回答：不抛异常，返回空命中、占比为 0。
4. 中文日期解析失败：返回 NaT 且清洗时被丢弃。
5. 词典缺 sheet：缺一个 sheet 不崩溃（该词集置空），四个全缺抛 LexiconError。
6. 占比 clip：>1 视为百分比、越界收敛到 [0,1]、非数值归 0。

运行方式（在项目根目录）：
    pytest -q tests/test_phase2.py
"""
from __future__ import annotations

import pandas as pd
import pytest

import cleaner
import sentiment
from cleaner import (
    MODE_AMBIGUOUS,
    MODE_ANALYZED,
    MODE_RAW,
    MODE_UNKNOWN,
    clean_raw_comments,
    detect_mode,
    detect_mode_detail,
    finalize_mode,
    map_columns,
    parse_datetime,
    read_excel_smart,
)
from sentiment import (
    LexiconError,
    clip_ratio,
    compute_ratios,
    emotion_word_frequency,
    load_lexicon,
    match_emotions,
    parse_emotion_columns,
    parse_ratio_columns,
    word_frequency_from_detail,
)


# ---------------------------------------------------------------------------
# 测试用小型词典（仅含少量词，便于断言命中结果）
# ---------------------------------------------------------------------------
@pytest.fixture
def mini_lexicon() -> dict[str, set[str]]:
    return {
        "neg_annual": {"亏损", "下滑"},
        "pos_annual": {"增长", "盈利"},
        "neg_social": {"爆雷", "套牢"},
        "pos_social": {"利好", "涨停"},
    }


# ---------------------------------------------------------------------------
# 1. 原始模式
# ---------------------------------------------------------------------------
def test_detect_mode_raw() -> None:
    df = pd.DataFrame({"company": ["A"], "date": ["2024-01-01"], "question": ["q"], "answer": ["a"]})
    assert detect_mode(df) == MODE_RAW


def test_match_emotions_raw(mini_lexicon) -> None:
    text = "公司业绩增长，盈利改善，但仍有下滑风险"
    hits = match_emotions(text, mini_lexicon)
    assert "增长" in hits["pos_annual"]
    assert "盈利" in hits["pos_annual"]
    assert "下滑" in hits["neg_annual"]
    assert "利好" not in hits["pos_social"]


def test_compute_ratios_raw(mini_lexicon) -> None:
    # 占比 = 命中词字符长度 / 文本总长度。
    # "增长 盈利 下滑"：正=「增长」+「盈利」共 4 字符，负=「下滑」2 字符，总长 8。
    text = "增长 盈利 下滑"
    pos, neg = compute_ratios(text, mini_lexicon)
    assert pos == pytest.approx(4 / 8)
    assert neg == pytest.approx(2 / 8)


def test_compute_ratios_short_text_returns_zero(mini_lexicon) -> None:
    """长度 <= 5 的废话文本不参与计算，直接返回 (0.0, 0.0)。"""
    assert compute_ratios("感谢关注", mini_lexicon) == (0.0, 0.0)
    assert compute_ratios("您好", mini_lexicon) == (0.0, 0.0)


def test_clean_raw_comments() -> None:
    df = pd.DataFrame(
        {
            "公司": ["海康威视", "海康威视", "海康威视"],
            "时间": ["2019年02月21日 08:27", "2019-02-22 10:00", "不是日期"],
            "提问": ["q1", "q2", "q3"],
            "回答": ["a1", "a2", "a3"],
        }
    )
    out = clean_raw_comments(df)
    # "不是日期" 那行被丢弃，剩余 2 行
    assert len(out) == 2
    assert list(out.columns) == ["company", "date", "question", "answer"]
    assert out.loc[0, "date"].year == 2019
    assert out.loc[0, "date"].month == 2


# ---------------------------------------------------------------------------
# 2. 已分析模式
# ---------------------------------------------------------------------------
def _analyzed_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "时间": ["2024-01-05", "2024-01-06"],
            "回答信息": ["", ""],  # 回答很多为空，但情绪词有值（情绪词可能来自提问）
            "年报负面情绪词": ["亏损、下滑", ""],
            "年报正面情绪词": ["", "增长"],
            "社媒负面情绪词": ["套牢", ""],
            "社媒正面情绪词": ["", "利好、涨停"],
            "正向占比": [0.2, 0.8],
            "负向占比": [0.8, 0.2],
        }
    )


def test_detect_mode_analyzed() -> None:
    assert detect_mode(_analyzed_df()) == MODE_ANALYZED


def test_detect_mode_real_uploaded_table_not_raw() -> None:
    """用户真实上传表（无 question 列、有四类情绪词列）应判为已分析模式。

    该表含「回答信息」列（且部分为空），不应被误判为原始模式或「需用户选择」。
    """
    df = pd.DataFrame(
        {
            "时间": ["2024-01-05", "2024-01-06", "2024-01-07"],
            "回答信息": ["", "感谢关注。", ""],
            "年报负面情绪词": ["亏损", "", "下滑"],
            "年报正面情绪词": ["", "增长", ""],
            "社媒负面情绪词": ["", "套牢", ""],
            "社媒正面情绪词": ["利好", "", "涨停"],
            "正向占比": [0.2, 0.6, 0.3],
            "负向占比": [0.8, 0.4, 0.7],
        }
    )
    mode = detect_mode(df)
    assert mode == MODE_ANALYZED
    assert mode != MODE_RAW
    assert mode != MODE_AMBIGUOUS


def test_detect_mode_ambiguous() -> None:
    df = _analyzed_df()
    df["question"] = ["q1", "q2"]  # 既有 question 又有情绪词列 → 需用户选择
    assert detect_mode(df) == MODE_AMBIGUOUS


def test_detect_mode_detail_real_uploaded_table() -> None:
    """detect_mode_detail 应给出「已分析模式」及其判定理由。

    真实上传表：无 question 列、有四类情绪词列、含「回答信息」列。
    理由须明确点出「无 question 列」，且不得误判为原始模式或需用户选择。
    """
    df = pd.DataFrame(
        {
            "时间": ["2024-01-05", "2024-01-06"],
            "回答信息": ["", "感谢关注。"],
            "年报负面情绪词": ["亏损", ""],
            "年报正面情绪词": ["", "增长"],
            "社媒负面情绪词": ["", "套牢"],
            "社媒正面情绪词": ["利好", ""],
            "正向占比": [0.2, 0.6],
            "负向占比": [0.8, 0.4],
        }
    )
    detail = detect_mode_detail(df)
    assert detail["mode"] == MODE_ANALYZED
    assert "无 question 列" in detail["reason"]
    assert "已分析模式" in detail["reason"]


def test_detect_mode_detail_raw() -> None:
    df = pd.DataFrame(
        {"company": ["A"], "date": ["2024-01-01"], "question": ["q"], "answer": ["a"]}
    )
    detail = detect_mode_detail(df)
    assert detail["mode"] == MODE_RAW
    assert "原始模式" in detail["reason"]


def test_detect_mode_detail_ambiguous() -> None:
    df = _analyzed_df()
    df["question"] = ["q1", "q2"]
    detail = detect_mode_detail(df)
    assert detail["mode"] == MODE_AMBIGUOUS
    assert "手动选择" in detail["reason"]


def test_detect_mode_consistent_with_detail() -> None:
    """detect_mode 与 detect_mode_detail 的模式判定必须一致（单一事实来源）。"""
    for df in (_analyzed_df(), pd.DataFrame({"foo": [1, 2]})):
        assert detect_mode(df) == detect_mode_detail(df)["mode"]


def test_detect_mode_unknown() -> None:
    df = pd.DataFrame({"foo": [1, 2]})
    assert detect_mode(df) == MODE_UNKNOWN


def test_parse_emotion_columns() -> None:
    parsed = parse_emotion_columns(_analyzed_df())
    assert parsed.loc[0, "neg_annual_words"] == ["亏损", "下滑"]
    assert parsed.loc[1, "neg_annual_words"] == []      # 空值 → 空列表
    assert parsed.loc[1, "pos_social_words"] == ["利好", "涨停"]


def test_parse_ratio_columns() -> None:
    ratios = parse_ratio_columns(_analyzed_df())
    assert ratios.loc[0, "pos_ratio"] == pytest.approx(0.2)
    assert ratios.loc[1, "neg_ratio"] == pytest.approx(0.2)


def test_emotion_word_frequency() -> None:
    parsed = parse_emotion_columns(_analyzed_df())
    freq = emotion_word_frequency(parsed)
    # "亏损"、"下滑"、"套牢"、"增长"、"利好"、"涨停" 各出现一次
    assert len(freq) == 6
    assert set(freq["word"]) == {"亏损", "下滑", "套牢", "增长", "利好", "涨停"}
    assert list(freq.columns) == ["word", "count", "category"]


def test_aggregate_by_month_chinese_date() -> None:
    """aggregate_by_month 应支持中文日期（如「2019年02月21日 08:27」）。"""
    df = pd.DataFrame(
        {
            "date": ["2019年02月21日 08:27", "2019年02月22日 10:00", "2019年03月01日"],
            "pos_ratio": [0.2, 0.4, 0.6],
            "neg_ratio": [0.8, 0.6, 0.4],
        }
    )
    out = sentiment.aggregate_by_month(df)
    assert list(out["month"]) == ["2019-02", "2019-03"]
    assert out.loc[0, "pos_ratio"] == pytest.approx(0.3)  # (0.2+0.4)/2
    assert out.loc[0, "count"] == 2


# ---------------------------------------------------------------------------
# 3. 空回答
# ---------------------------------------------------------------------------
def test_match_emotions_empty_answer(mini_lexicon) -> None:
    assert match_emotions("", mini_lexicon) == {
        "neg_annual": [], "pos_annual": [], "neg_social": [], "pos_social": []
    }
    assert match_emotions(None, mini_lexicon)["neg_annual"] == []


def test_compute_ratios_empty_answer(mini_lexicon) -> None:
    assert compute_ratios("", mini_lexicon) == (0.0, 0.0)
    assert compute_ratios(None, mini_lexicon) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# 4. 中文日期解析失败
# ---------------------------------------------------------------------------
def test_parse_datetime_cn() -> None:
    ts = parse_datetime("2019年02月21日 08:27")
    assert (ts.year, ts.month, ts.day, ts.hour, ts.minute) == (2019, 2, 21, 8, 27)


def test_parse_datetime_invalid_returns_nat() -> None:
    assert pd.isna(parse_datetime("不是日期"))
    assert pd.isna(parse_datetime(""))
    assert pd.isna(parse_datetime(None))


# ---------------------------------------------------------------------------
# 5. 词典缺 sheet
# ---------------------------------------------------------------------------
def test_load_lexicon_missing_one_sheet(tmp_path) -> None:
    """缺一个 sheet 不应崩溃，缺失的词集置空，其余正常加载。"""
    path = tmp_path / "dict_missing.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"w": ["亏损", "下滑"]}).to_excel(writer, sheet_name="年报负面", index=False, header=False)
        pd.DataFrame({"w": ["增长"]}).to_excel(writer, sheet_name="年报正面", index=False, header=False)
        pd.DataFrame({"w": ["利好"]}).to_excel(writer, sheet_name="社媒正面", index=False, header=False)
        # 缺「社媒负面」

    lexicon = load_lexicon(path)
    assert lexicon["neg_annual"] == {"亏损", "下滑"}
    assert lexicon["pos_annual"] == {"增长"}
    assert lexicon["neg_social"] == set()          # 缺失 sheet → 空集
    assert lexicon["pos_social"] == {"利好"}


def test_load_lexicon_all_missing_raises(tmp_path) -> None:
    """四个词典 sheet 全缺（仅「说明」页）时应抛 LexiconError。"""
    path = tmp_path / "dict_empty.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"note": ["这是说明页"]}).to_excel(writer, sheet_name="说明", index=False)

    with pytest.raises(LexiconError):
        load_lexicon(path)


def test_load_lexicon_skips_instruction_sheet() -> None:
    """默认词典应自动跳过「说明」页，只返回四个词典键。"""
    lexicon = load_lexicon()  # 使用 data/dictionary/金融领域中文情绪词典.xlsx
    assert set(lexicon.keys()) == {"neg_annual", "pos_annual", "neg_social", "pos_social"}
    assert all(len(words) > 0 for words in lexicon.values())


# ---------------------------------------------------------------------------
# 6. 占比 clip
# ---------------------------------------------------------------------------
def test_clip_ratio() -> None:
    assert clip_ratio(0.5) == pytest.approx(0.5)
    assert clip_ratio(1.5) == pytest.approx(1.0)     # 越界 → 收敛到 1
    assert clip_ratio(-0.2) == pytest.approx(0.0)    # 负值 → 收敛到 0
    assert clip_ratio(85) == pytest.approx(1.0)      # 超过 1 → 封顶 1
    assert clip_ratio("abc") == pytest.approx(0.0)   # 非数值 → 0


# ---------------------------------------------------------------------------
# 7. 智能读取 sheet（跳过无法识别的汇总 sheet）
# ---------------------------------------------------------------------------
def test_read_excel_smart_skips_unrecognized_sheet(tmp_path) -> None:
    """上传 Excel 第一个 sheet 是年度汇总（无法识别），应自动跳到含情绪词列的明细 sheet。"""
    path = tmp_path / "上传数据.xlsx"
    summary = pd.DataFrame({"年份": [2023, 2024], "正向占比": [0.4, 0.5], "负向占比": [0.6, 0.5]})
    detail = _analyzed_df()
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # 年度汇总放在第一个位置（sheet 名按写入顺序排列）
        summary.to_excel(writer, sheet_name="Sheet2", index=False)
        detail.to_excel(writer, sheet_name="Sheet1", index=False)

    df, sheet = read_excel_smart(path)
    assert sheet == "Sheet1"
    assert detect_mode(df) == MODE_ANALYZED
    assert "年报负面情绪词" in df.columns


def test_read_excel_smart_first_sheet_ok(tmp_path) -> None:
    """第一个 sheet 就能识别时，保持原行为直接返回第一个 sheet。"""
    path = tmp_path / "正常.xlsx"
    _analyzed_df().to_excel(path, index=False)
    df, sheet = read_excel_smart(path)
    assert sheet == "Sheet1"  # 默认第一个 sheet 名
    assert detect_mode(df) == MODE_ANALYZED


# ---------------------------------------------------------------------------
# 8. 从情绪明细表统计词频（供词云）
# ---------------------------------------------------------------------------
def test_word_frequency_from_detail() -> None:
    """从情绪明细表（顿号分隔字符串）统计词频，不对 answer 重新分词。"""
    detail = pd.DataFrame(
        {
            "date": ["2024-01-05", "2024-01-06"],
            "pos_ratio": [0.2, 0.8],
            "neg_ratio": [0.8, 0.2],
            "neg_annual_words": ["亏损、下滑", ""],
            "pos_annual_words": ["", "增长"],
            "neg_social_words": ["套牢", ""],
            "pos_social_words": ["", "利好、涨停"],
        }
    )
    freq = word_frequency_from_detail(detail)
    assert set(freq["word"]) == {"亏损", "下滑", "套牢", "增长", "利好", "涨停"}
    assert len(freq) == 6
    assert list(freq.columns) == ["word", "count", "category"]


def test_word_frequency_from_detail_empty() -> None:
    freq = word_frequency_from_detail(pd.DataFrame())
    assert freq.empty
    assert list(freq.columns) == ["word", "count", "category"]


# ---------------------------------------------------------------------------
# 9. 在线爬取数据 → 原始模式（防止误判为已分析模式的防御式复核）
# ---------------------------------------------------------------------------
def _online_crawl_df() -> pd.DataFrame:
    """模拟在线爬取返回的统一 schema（company/date/question/answer），5 行。"""
    return pd.DataFrame(
        {
            "company": ["海康威视"] * 5,
            "date": [
                "2024-01-05 10:30",
                "2024-01-06 14:00",
                "2024-01-07 09:15",
                "2024-01-08 11:00",
                "2024-01-09 16:20",
            ],
            "question": [
                "请问分拆上市进展如何？",
                "公司创新业务有何规划？",
                "一季度经营情况如何？",
                "萤石网络分拆进展？",
                "未来是否有分拆计划？",
            ],
            "answer": [
                "感谢关注，请以公告为准。",
                "如有进展将及时披露。",
                "经营情况详见定期报告。",
                "相关进展请关注公告。",
                "如有计划将及时披露。",
            ],
        }
    )


def test_online_crawl_df_detected_as_raw() -> None:
    """在线爬取返回的 date/question/answer 表必须识别为原始模式，而非已分析模式。"""
    df = _online_crawl_df()
    assert detect_mode(df) == MODE_RAW
    assert detect_mode(df) != MODE_ANALYZED


def test_finalize_mode_online_crawl_keeps_raw() -> None:
    """finalize_mode 对在线爬取数据应维持原始模式，且理由含「原始模式」。"""
    mode, reason = finalize_mode(_online_crawl_df())
    assert mode == MODE_RAW
    assert "原始模式" in reason


def test_finalize_mode_rejects_false_analyzed() -> None:
    """防御：即使初判被错误标为已分析模式，只要无情绪词列、有 question/answer 列，就修正为原始模式。"""
    df = _online_crawl_df()
    mode, reason = finalize_mode(df, detected_mode=MODE_ANALYZED)
    assert mode == MODE_RAW
    assert "修正为原始模式" in reason


def test_finalize_mode_analyzed_keeps_analyzed() -> None:
    """已分析表应维持已分析模式，防御逻辑不得误伤。"""
    mode, _reason = finalize_mode(_analyzed_df())
    assert mode == MODE_ANALYZED


def test_map_columns_empty_mapping_auto_detects() -> None:
    """空 mapping（如在线/缓存模式未做界面映射）时，map_columns 也应自动识别列名。"""
    df = pd.DataFrame(
        {"公司": ["A"], "时间": ["2024-01-01"], "提问": ["q"], "回答": ["a"]}
    )
    out = map_columns(df, mapping={})
    assert set(out.columns) == {"company", "date", "question", "answer"}


def test_online_crawl_raw_pipeline_no_valueerror() -> None:
    """模拟在线爬取数据走原始模式清洗全流程，不得抛出「未找到四类情绪词列」。"""
    df = _online_crawl_df()
    assert detect_mode(df) == MODE_RAW

    # 原始模式：清洗 → 词典匹配 → 计算占比（不应调用 parse_emotion_columns）
    cleaned = clean_raw_comments(df)
    assert len(cleaned) == 5
    assert list(cleaned.columns) == ["company", "date", "question", "answer"]

    lexicon = load_lexicon()
    # 只取 question 提问文本参与情绪计算，绝不使用 answer。
    text = " ".join(cleaned["question"].tolist())
    pos_ratio, neg_ratio = compute_ratios(text, lexicon)
    assert isinstance(pos_ratio, float)
    assert isinstance(neg_ratio, float)
