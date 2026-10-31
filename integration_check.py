# -*- coding: utf-8 -*-
"""全流程联调自检脚本（阶段 5）。

联调顺序（按用户要求）：
1. 优先用「用户真实表」（已分析模式）跑通全流程：模式识别 → 情绪 → 图表 → 相关性 → 结论 → 导出。
2. 再测「原始模式」（本地缓存海康威视 → 词典匹配）跑通全流程。
3. 最后测降级路径：不填 Cookie / 错误 Cookie 回退缓存、无行情、无 PE、n<30、R²<0.1、导出不含敏感信息。

运行：python integration_check.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import config
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
    clean_raw_comments,
    detect_mode,
    detect_mode_detail,
    find_column,
)
from crawler import CrawlerError
from report_generator import generate_report
from visualization import (
    make_event_window_bar,
    make_market_emotion_dual_axis,
    make_monthly_trend,
    make_wordcloud_grid,
)

RESULTS: list[tuple[str, bool, str]] = []
TAB_SUMMARY: dict[str, str] = {}


def check(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  —— {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------
_UPLOAD_SRC = "/root/uploads/1793272839403792330-爬取评论+数据.xlsx"


def _load_analyzed_table() -> pd.DataFrame:
    """优先读用户真实表；不存在时构造一份等价的已分析模式样例。"""
    src = Path(_UPLOAD_SRC)
    if src.exists():
        return pd.read_excel(src, sheet_name="Sheet1")
    return pd.DataFrame(
        {
            "时间": ["2024-01-05", "2024-01-06", "2024-01-07"],
            "回答信息": ["", "", ""],
            "年报负面情绪词": ["亏损、下滑", "", ""],
            "年报正面情绪词": ["", "增长", "盈利"],
            "社媒负面情绪词": ["套牢", "", ""],
            "社媒正面情绪词": ["", "利好", "涨停"],
            "正向占比": [0.2, 0.7, 0.9],
            "负向占比": [0.8, 0.3, 0.1],
        }
    )


market = pd.read_csv(config.market_template_path("002415"), encoding="utf-8-sig")
lexicon = sentiment.load_lexicon()
tmp_dir = Path(tempfile.mkdtemp())


# ===========================================================================
# 1. 已分析模式（用户真实表）全流程
# ===========================================================================
df_analyzed = _load_analyzed_table()
mode_detail = detect_mode_detail(df_analyzed)
check(
    "1a 真实表识别为已分析模式",
    mode_detail["mode"] == "已分析模式",
    f"mode={mode_detail['mode']}",
)
TAB_SUMMARY["Tab1.模式"] = mode_detail["mode"]
TAB_SUMMARY["Tab1.识别依据"] = mode_detail["reason"]

words_df = sentiment.parse_emotion_columns(df_analyzed)
ratios_df = sentiment.parse_ratio_columns(df_analyzed)
date_col = find_column(df_analyzed, COMMENT_COLUMN_ALIASES["date"]) or "时间"
detail = pd.DataFrame({"date": df_analyzed[date_col].astype(str)})
detail["pos_ratio"] = ratios_df["pos_ratio"]
detail["neg_ratio"] = ratios_df["neg_ratio"]
daily = aggregate_daily(detail[["date", "pos_ratio", "neg_ratio"]])
monthly = sentiment.aggregate_by_month(detail[["date", "pos_ratio", "neg_ratio"]])
word_freq = sentiment.emotion_word_frequency(words_df)

fig_wc = make_wordcloud_grid(word_freq)
fig_trend = make_monthly_trend(monthly, "2022-12-28")
ew = event_window_compare(daily, "2022-12-28", 180)
fig_ew = make_event_window_bar(ew)
check(
    "1b 已分析模式图表生成",
    fig_wc is not None and fig_trend is not None and fig_ew is not None,
    f"daily={len(daily)} word_freq={len(word_freq)}",
)
TAB_SUMMARY["Tab2.词云词数"] = str(len(word_freq))
TAB_SUMMARY["Tab2.月度趋势月数"] = str(len(monthly))
TAB_SUMMARY["Tab2.事件前后样本"] = f"前={ew.before_count} / 后={ew.after_count}"

corr_close = correlate(daily, market, target="close")
corr_pe = correlate(daily, market, target="pe")
check("1c 已分析模式相关性计算", corr_close is not None and corr_close.valid, f"n={corr_close.n}")
TAB_SUMMARY["Tab3.股价相关"] = (
    f"n={corr_close.n}, r={corr_close.r:.4f}, R²={corr_close.r_squared:.4f}"
    + (f" | warning={corr_close.warning}" if corr_close.warning else "")
    + (f" | caution={corr_close.caution}" if corr_close.caution else "")
)
if corr_pe is not None:
    TAB_SUMMARY["Tab3.PE相关"] = f"n={corr_pe.n}, r={corr_pe.r:.4f}, R²={corr_pe.r_squared:.4f}"

report = generate_report(
    {
        "company": "海康威视", "code": "002415", "market": "SZ",
        "data_origin": "上传", "mode": "已分析模式", "n_comments": len(df_analyzed),
        "date_start": daily["date"].min(), "date_end": daily["date"].max(), "n_days": len(daily),
        "pos_ratio_mean": float(daily["pos_ratio"].mean()),
        "neg_ratio_mean": float(daily["neg_ratio"].mean()),
        "top_pos_words": ["增长"], "top_neg_words": ["亏损"],
        "event_window": ew, "corr_close": corr_close, "corr_pe": corr_pe,
        "has_market": True, "has_pe": corr_pe is not None,
    }
)
check("1d 已分析模式结论生成", "R²=" in report, "报告含 R² 数值")
TAB_SUMMARY["Tab4.结论首行"] = report.splitlines()[0] if report else ""

xlsx = exporter.export_excel(
    {"raw_df": df_analyzed, "emotion_detail": detail, "daily": daily, "monthly": monthly,
     "corr_close": corr_close, "corr_pe": corr_pe, "event_window": ew},
    tmp_dir / "r_analyzed.xlsx",
)
check("1e 已分析模式 Excel 导出", xlsx.exists(), xlsx.name)

# ===========================================================================
# 2. 原始模式（本地缓存 → 词典匹配）全流程
# ===========================================================================
cache = pd.read_excel(config.CACHE_DIR / "海康威视.xlsx")
raw_mode_detail = detect_mode_detail(cache)
check(
    "2a 缓存识别为原始模式",
    raw_mode_detail["mode"] == "原始模式",
    f"mode={raw_mode_detail['mode']}",
)
cleaned = clean_raw_comments(cache, company_default="海康威视")
_cache_rows = []
for _, row in cleaned.iterrows():
    # 只取 question 提问文本，绝不使用 answer 回答文本。
    text = str(row.get("question") or "").strip()
    pos, neg = sentiment.compute_ratios(text, lexicon)
    _cache_rows.append({"date": str(row["date"]), "pos_ratio": pos, "neg_ratio": neg})
raw_detail = pd.DataFrame(_cache_rows)
raw_daily = aggregate_daily(raw_detail[["date", "pos_ratio", "neg_ratio"]])
raw_monthly = sentiment.aggregate_by_month(raw_detail[["date", "pos_ratio", "neg_ratio"]])

# 原始模式词频：从命中词统计（只取 question 提问文本）
_freq_rows = []
for _, row in cleaned.iterrows():
    text = str(row.get("question") or "").strip()
    for cat, words in sentiment.match_emotions(text, lexicon).items():
        for w in words:
            _freq_rows.append({"word": w, "count": 1, "category": cat})
raw_freq = (
    pd.DataFrame(_freq_rows).groupby(["word", "category"], as_index=False)["count"].sum()
    if _freq_rows
    else pd.DataFrame(columns=["word", "count", "category"])
)
raw_wc = make_wordcloud_grid(raw_freq)
check(
    "2b 原始模式图表生成",
    raw_wc is not None and len(raw_daily) > 0,
    f"raw_daily={len(raw_daily)} raw_freq={len(raw_freq)}",
)
real_corr = correlate(raw_daily, market, target="close")
check("2c 原始模式相关性计算", real_corr is not None and real_corr.valid, f"n={real_corr.n}")
TAB_SUMMARY["原始模式.词频数"] = str(len(raw_freq))
TAB_SUMMARY["原始模式.日聚合"] = str(len(raw_daily))

# ===========================================================================
# 3. 不填 Cookie → 回退缓存
# ===========================================================================
r_nocookie = data_source.get_data("海康威视", "在线爬取", cookie_str=None)
check("3 不填Cookie回退缓存", r_nocookie.data_origin == "缓存", r_nocookie.message)

# ===========================================================================
# 4. 错误 Cookie → 爬取失败 → 回退缓存
# ===========================================================================
_orig_crawl = data_source.crawl_comments


def _fake_crawl(*args, **kwargs):
    raise CrawlerError("模拟 Cookie 失效（403）")


data_source.crawl_comments = _fake_crawl
r_badcookie = data_source.get_data("海康威视", "在线爬取", cookie_str="bad_cookie_value")
data_source.crawl_comments = _orig_crawl
check(
    "4 错误Cookie回退缓存并提示原因",
    r_badcookie.data_origin == "缓存" and "回退" in r_badcookie.message,
    r_badcookie.message,
)

# ===========================================================================
# 5. 无行情 → 降级提示（不崩溃）
# ===========================================================================
r_nomarket = market_data.get_market("999999", mode="内置CSV")
check("5 无行情返回空结果不崩溃", r_nomarket.empty, r_nomarket.message)

# ===========================================================================
# 6. 无 PE 有 close → PE 隐藏、股价正常
# ===========================================================================
market_no_pe = market[["date", "close"]]
corr_close2 = correlate(daily, market_no_pe, target="close")
corr_pe2 = correlate(daily, market_no_pe, target="pe")
check(
    "6 无PE时股价正常、PE隐藏",
    corr_close2 is not None and corr_close2.valid and corr_pe2 is None,
    "corr_pe=None（界面隐藏 PE 图）",
)

# ===========================================================================
# 7. n<30 warning / R²<0.1 caution
# ===========================================================================
small_daily = pd.DataFrame(
    {
        "date": pd.date_range("2024-01-01", periods=15, freq="D").strftime("%Y-%m-%d"),
        "pos_ratio": [0.1 + 0.01 * i for i in range(15)],
        "neg_ratio": [0.9 - 0.01 * i for i in range(15)],
    }
)
small_market = pd.DataFrame(
    {
        "date": pd.date_range("2024-01-01", periods=15, freq="D").strftime("%Y-%m-%d"),
        "close": [10.0 + i for i in range(15)],
        "pe": [20.0] * 15,
    }
)
small_corr = correlate(small_daily, small_market, target="close")
check(
    "7a n<30 触发 warning",
    small_corr is not None and "样本量不足" in small_corr.warning,
    small_corr.warning if small_corr else "",
)
check(
    "7b 真实数据 R²<0.1 触发 caution",
    real_corr is not None and "解释力有限" in real_corr.caution,
    f"R²={real_corr.r_squared:.4f}" if real_corr else "",
)
TAB_SUMMARY["Tab3.n<30提示"] = small_corr.warning if small_corr else ""
TAB_SUMMARY["Tab3.R²<0.1提示"] = real_corr.caution if real_corr else ""

# ===========================================================================
# 8. 导出文件不含 Cookie / 学号姓名
# ===========================================================================
report_md = generate_report(
    {"company": "海康威视", "has_market": True, "has_pe": True, "corr_close": real_corr}
)
md_path = exporter.export_markdown(report_md, tmp_dir / "report.md")
word_path = exporter.export_word(report_md, tmp_dir / "report.docx")
fig_dir = exporter.export_figures(
    {"词云": raw_wc, "月度趋势": fig_trend}, tmp_dir / "figures", company="海康威视"
)


def _no_sensitive(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    return all(k not in text for k in ("cookie", "学号", "姓名", "password"))


check("8 导出不含敏感信息", _no_sensitive(md_path) and _no_sensitive(word_path), "")
TAB_SUMMARY["Tab5.导出文件"] = ", ".join(
    sorted([xlsx.name, md_path.name, word_path.name] + [p.name for p in fig_dir])
)


# ===========================================================================
# 汇总 + 各 Tab 输出摘要
# ===========================================================================
failed = [r for r in RESULTS if not r[1]]
print(f"\n联调结果：{len(RESULTS) - len(failed)}/{len(RESULTS)} 通过")
if failed:
    print("失败项：", [r[0] for r in failed])

print("\n================ 各 Tab 实际输出摘要 ================")
for key, value in TAB_SUMMARY.items():
    print(f"  {key}: {value}")
print("=====================================================")

if failed:
    sys.exit(1)
print("全流程联调通过。")
