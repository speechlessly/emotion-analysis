# -*- coding: utf-8 -*-
"""阶段 5 测试用例（导出 exporter.py + 行情情绪双轴图 + 日期覆盖）。

覆盖点：
1. Excel 正常导出（多 sheet）/ 空导出不报错。
2. Markdown / Word 正常导出；Word 缺依赖时提示不报错。
3. 图表导出文件名含时间戳。
4. 导出文件不含 Cookie 字符串。
5. make_market_emotion_dual_axis：无 PE 降级为情绪+股价；无情绪时提示。
6. prepare_xy 日期覆盖事件日（真实数据），不满足则跳过并 warn。

运行方式（项目根目录）：
    pytest -q tests/test_phase5.py
"""
from __future__ import annotations

import builtins
import re
import warnings

import matplotlib
import pandas as pd
import plotly.graph_objects as go
import pytest

import config
import cleaner
from analysis import (
    CorrelationResult,
    EventWindowResult,
    aggregate_daily,
    prepare_xy,
)
from exporter import (
    ExportError,
    export_excel,
    export_figures,
    export_markdown,
    export_word,
)
from visualization import make_market_emotion_dual_axis, make_wordcloud


# ---------------------------------------------------------------------------
# 辅助构造
# ---------------------------------------------------------------------------
def _result_dict() -> dict:
    return {
        "raw_df": pd.DataFrame(
            {"company": ["海康威视"], "date": ["2024-01-01"], "question": ["q"], "answer": ["a"]}
        ),
        "emotion_detail": pd.DataFrame(
            {"date": ["2024-01-01"], "pos_ratio": [0.5], "neg_ratio": [0.5]}
        ),
        "daily": pd.DataFrame(
            {"date": ["2024-01-01"], "pos_ratio": [0.5], "neg_ratio": [0.5], "n": [1]}
        ),
        "monthly": pd.DataFrame(
            {"month": ["2024-01"], "pos_ratio": [0.5], "neg_ratio": [0.5], "count": [1]}
        ),
        "corr_close": CorrelationResult(
            target="close", valid=True, r=0.5, p=0.1, n=30, r_squared=0.25
        ),
        "event_window": EventWindowResult(
            event_date="2024-01-01", window_days=90, valid=True,
            before_count=1, after_count=1,
            before_pos_mean=0.4, after_pos_mean=0.5,
            before_neg_mean=0.6, after_neg_mean=0.5,
            pos_diff=0.1, neg_diff=-0.1,
        ),
    }


# ---------------------------------------------------------------------------
# 1. Excel 导出
# ---------------------------------------------------------------------------
def test_export_excel_normal(tmp_path) -> None:
    path = export_excel(_result_dict(), tmp_path / "out.xlsx")
    assert path.exists()
    xls = pd.ExcelFile(path)
    for sheet in ("原始数据", "情绪明细", "日聚合", "月度聚合", "相关性", "事件前后"):
        assert sheet in xls.sheet_names


def test_export_excel_empty(tmp_path) -> None:
    path = export_excel({}, tmp_path / "empty.xlsx")
    assert path.exists()
    xls = pd.ExcelFile(path)
    assert "说明" in xls.sheet_names  # 空导出时兜底写说明页，不报错


# ---------------------------------------------------------------------------
# 2. Markdown / Word
# ---------------------------------------------------------------------------
def test_export_markdown(tmp_path) -> None:
    path = export_markdown("# 标题\n- 条目", tmp_path / "r.md")
    assert path.exists()
    assert "标题" in path.read_text(encoding="utf-8")


def test_export_word_normal(tmp_path) -> None:
    path = export_word("# 标题\n- 条目一\n普通段落", tmp_path / "r.docx")
    assert path.exists()
    import docx

    doc = docx.Document(path)
    texts = [p.text for p in doc.paragraphs]
    assert "标题" in texts
    assert "条目一" in texts


def test_export_word_empty(tmp_path) -> None:
    path = export_word("", tmp_path / "empty.docx")
    assert path.exists()


def test_export_word_missing_dependency(tmp_path, monkeypatch) -> None:
    """python-docx 缺失时抛 ExportError（可读提示），不崩溃。"""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "docx":
            raise ImportError("No module named 'docx'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ExportError):
        export_word("# x", tmp_path / "x.docx")


# ---------------------------------------------------------------------------
# 3. 图表导出（文件名含时间戳）
# ---------------------------------------------------------------------------
def test_export_figures_filename_timestamp(tmp_path) -> None:
    word_freq = pd.DataFrame([{"word": "增长", "count": 3, "category": "pos_annual_words"}])
    figures = {"词云": make_wordcloud(word_freq)}
    saved = export_figures(figures, tmp_path, company="海康威视")
    assert len(saved) == 1
    name = saved[0].name
    # 文件名含 公司名 + 时间戳（YYYYMMDD_HHMMSS）
    assert "海康威视" in name
    assert re.search(r"\d{8}_\d{6}", name)


def test_export_figures_empty(tmp_path) -> None:
    saved = export_figures({"空": None}, tmp_path)
    assert saved == []


# ---------------------------------------------------------------------------
# 4. 导出文件不含 Cookie
# ---------------------------------------------------------------------------
def test_export_files_no_cookie(tmp_path) -> None:
    result_dict = _result_dict()
    report_md = "# 报告\n- 无敏感信息\n"
    excel_path = export_excel(result_dict, tmp_path / "a.xlsx")
    md_path = export_markdown(report_md, tmp_path / "a.md")
    word_path = export_word(report_md, tmp_path / "a.docx")

    # Markdown 纯文本
    assert "cookie" not in md_path.read_text(encoding="utf-8").lower()

    # Excel：逐 sheet 的字符串表示检查
    xls = pd.ExcelFile(excel_path)
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        assert "cookie" not in df.to_string().lower()

    # Word 段落文本
    import docx

    doc = docx.Document(word_path)
    for p in doc.paragraphs:
        assert "cookie" not in p.text.lower()


# ---------------------------------------------------------------------------
# 5. 行情情绪双轴图（无 PE 降级 / 无情绪提示）
# ---------------------------------------------------------------------------
def test_make_market_emotion_dual_axis_no_pe() -> None:
    """无 PE 时自动降级为情绪+股价，且不报错。"""
    market = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "close": [10.0, 11.0]})
    emotion = pd.DataFrame(
        {"date": ["2024-01-01", "2024-01-02"], "pos_ratio": [0.3, 0.4], "neg_ratio": [0.7, 0.6]}
    )
    fig = make_market_emotion_dual_axis(market, emotion, target="pe")
    assert isinstance(fig, go.Figure)
    # 第一条 trace 名称应为「收盘价」（降级）
    assert fig.data[0].name == "收盘价"
    # 图注应包含「不含 PE」
    texts = " ".join(a.text for a in fig.layout.annotations)
    assert "不含 PE" in texts


def test_make_market_emotion_dual_axis_no_emotion() -> None:
    """无情绪数据时返回提示图，不报错。"""
    market = pd.DataFrame({"date": ["2024-01-01"], "close": [10.0], "pe": [20.0]})
    fig = make_market_emotion_dual_axis(market, pd.DataFrame(), target="pe")
    assert isinstance(fig, go.Figure)
    texts = " ".join(a.text for a in fig.layout.annotations)
    assert "暂无情绪数据" in texts


def test_make_market_emotion_dual_axis_with_pe() -> None:
    """有 PE 时左轴为 PE（三条 trace：PE + 正 + 负）。"""
    market = pd.DataFrame(
        {"date": ["2024-01-01", "2024-01-02"], "close": [10.0, 11.0], "pe": [20.0, 21.0]}
    )
    emotion = pd.DataFrame(
        {"date": ["2024-01-01", "2024-01-02"], "pos_ratio": [0.3, 0.4], "neg_ratio": [0.7, 0.6]}
    )
    fig = make_market_emotion_dual_axis(market, emotion, event_date="2024-01-01", target="pe")
    assert isinstance(fig, go.Figure)
    assert fig.data[0].name == "PE"
    assert len(fig.data) == 3


# ---------------------------------------------------------------------------
# 7. data_source 上传读取：自动跳过无法识别的汇总 sheet
# ---------------------------------------------------------------------------
def test_data_source_read_upload_smart_sheet(tmp_path) -> None:
    """data_source._read_upload 应自动读取可识别的明细 sheet，而非默认第一个汇总 sheet。"""
    import data_source

    path = tmp_path / "上传.xlsx"
    summary = pd.DataFrame({"年份": [2023], "正向占比": [0.4], "负向占比": [0.6]})
    detail = pd.DataFrame(
        {
            "时间": ["2024-01-05"],
            "回答信息": [""],
            "年报负面情绪词": ["亏损"],
            "年报正面情绪词": [""],
            "社媒负面情绪词": [""],
            "社媒正面情绪词": ["利好"],
            "正向占比": [0.3],
            "负向占比": [0.7],
        }
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # 汇总 sheet 放在第一个位置
        summary.to_excel(writer, sheet_name="Sheet2", index=False)
        detail.to_excel(writer, sheet_name="Sheet1", index=False)

    df = data_source._read_upload(str(path))
    assert "年报负面情绪词" in df.columns
    assert "年份" not in df.columns


# ---------------------------------------------------------------------------
# 6. prepare_xy 日期覆盖事件日（真实数据）
# ---------------------------------------------------------------------------
def test_prepare_xy_dates_cover_event() -> None:
    """真实数据：prepare_xy 返回的 dates 应覆盖 2022-12-28（不满足则跳过并 warn）。"""
    cache_path = config.DEMO_DIR / "海康威视.xlsx"
    market_path = config.market_template_path("002415")
    if not (cache_path.exists() and market_path.exists()):
        pytest.skip("演示数据或行情样例不存在，跳过日期覆盖断言")

    # 缓存表可能含多个 sheet（首个可能是年度汇总），用智能读取选中可识别的明细 sheet。
    cache, _ = cleaner.read_excel_smart(cache_path)
    # 缓存表日期列名可能为「date」或「时间」，用 cleaner 的别名匹配兜底，兼容两种缓存结构。
    date_col = cleaner.find_column(cache, cleaner.COMMENT_COLUMN_ALIASES["date"])
    if date_col is None:
        pytest.skip("缓存表无日期列，跳过日期覆盖断言")
    # 用缓存 date 构造情绪日聚合（占比用占位值，仅验证日期交集）
    emotion = pd.DataFrame(
        {"date": cache[date_col].astype(str), "pos_ratio": 0.5, "neg_ratio": 0.5}
    )
    daily = aggregate_daily(emotion)
    market = pd.read_csv(market_path, encoding="utf-8-sig")

    x, y, dates = prepare_xy(daily, market, target="close")
    n = len(dates)
    print(f"[日期覆盖] min={dates.min()}, max={dates.max()}, n={n}")

    event = pd.to_datetime("2022-12-28")
    if n == 0 or not (dates.min() <= event <= dates.max()):
        warnings.warn(
            f"真实数据未覆盖事件日 2022-12-28（min={dates.min()}, max={dates.max()}, n={n}），"
            "日期覆盖断言跳过。"
        )
        pytest.skip("真实数据未覆盖事件日 2022-12-28")
    assert dates.min() <= event <= dates.max()


# ---------------------------------------------------------------------------
# 8. 已分析模式：字段映射排除占位符后，词频不为空（回归「暂无词汇数据」bug）
# ---------------------------------------------------------------------------
def _load_app_module():
    """打桩 streamlit 后导入 app 模块（供 build_emotion_detail_analyzed 测试用）。

    app.py 是 Streamlit 脚本式入口，导入时会执行顶层界面代码；这里用 MagicMock
    打桩所有 st.* 调用（button 返回 False 跳过主流程、tabs 返回 6 个上下文管理器），
    使模块可被安全导入。首次导入后缓存到 sys.modules，后续调用直接复用。
    """
    import sys
    from unittest.mock import MagicMock

    if "app" in sys.modules:
        return sys.modules["app"]

    class _CM:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    stub = MagicMock()
    stub.button.return_value = False
    stub.checkbox.return_value = False
    stub.radio.return_value = "本地缓存"
    stub.selectbox.return_value = ""
    stub.text_input.return_value = ""
    stub.number_input.return_value = 7
    stub.file_uploader.return_value = None
    stub.sidebar.__enter__ = lambda self: self
    stub.sidebar.__exit__ = lambda *a: None
    stub.tabs.return_value = [_CM() for _ in range(6)]
    sys.modules["streamlit"] = stub

    import app  # noqa: PLC0415 —— 延迟导入，需先完成 streamlit 打桩

    return app


def test_build_emotion_detail_analyzed_word_freq() -> None:
    """已分析模式：含「时间 + 四类情绪词」的表经 build_emotion_detail_analyzed 后词频不为空。

    回归问题：字段映射误把占位符（如「（未自动匹配，请手动选择）」）传入后端，
    导致解析出的四类情绪词全为空、词云显示「暂无词汇数据」。此处构造标准表头 +
    正确映射，断言词频统计非空。
    """
    app = _load_app_module()

    df = pd.DataFrame(
        {
            "时间": ["2024-01-05", "2024-01-06", "2024-01-07"],
            "回答信息": ["", "", ""],
            "年报负面情绪词": ["亏损", "", "下滑"],
            "年报正面情绪词": ["", "增长", ""],
            "社媒负面情绪词": ["", "套牢", ""],
            "社媒正面情绪词": ["利好", "", "涨停"],
            "正向占比": [0.06, 0.05, 0.04],
            "负向占比": [0.03, 0.02, 0.01],
        }
    )
    mapping = {
        "date": "时间",
        "emotion_columns": {
            "neg_annual_words": "年报负面情绪词",
            "pos_annual_words": "年报正面情绪词",
            "neg_social_words": "社媒负面情绪词",
            "pos_social_words": "社媒正面情绪词",
        },
        "ratio_columns": {"pos_ratio": "正向占比", "neg_ratio": "负向占比"},
    }

    detail = app.build_emotion_detail_analyzed(df, mapping)
    freq = app.sentiment.word_frequency_from_detail(detail)

    assert not freq.empty
    assert {"word", "count", "category"} <= set(freq.columns)
    # 四类词都应被统计出来，而不是空表
    assert freq["category"].nunique() == 4
    assert set(freq["word"]) == {"亏损", "下滑", "增长", "套牢", "利好", "涨停"}


def test_build_emotion_detail_analyzed_excludes_placeholder() -> None:
    """已分析模式：映射值为占位符（None/非列名）时被排除，不报错、词频仍非空。

    回归 A3：render_field_mapping 已把选中占位符的字段转为 None；此处直接传
    含 None 的映射，验证 build_emotion_detail_analyzed 能正确过滤并解析其余列。
    """
    app = _load_app_module()

    df = pd.DataFrame(
        {
            "时间": ["2024-01-05"],
            "年报负面情绪词": ["亏损"],
            "年报正面情绪词": ["增长"],
            "社媒负面情绪词": ["套牢"],
            "社媒正面情绪词": ["利好"],
        }
    )
    # pos_social_words 映射为 None（模拟用户未选中该列），其余正常。
    mapping = {
        "date": "时间",
        "emotion_columns": {
            "neg_annual_words": "年报负面情绪词",
            "pos_annual_words": "年报正面情绪词",
            "neg_social_words": "社媒负面情绪词",
            "pos_social_words": None,
        },
        "ratio_columns": {"pos_ratio": None, "neg_ratio": None},
    }

    detail = app.build_emotion_detail_analyzed(df, mapping)
    freq = app.sentiment.word_frequency_from_detail(detail)

    assert not freq.empty
    # pos_social_words 被排除后，其余三类仍正常统计
    assert set(freq["category"]) == {"neg_annual_words", "pos_annual_words", "neg_social_words"}
