# -*- coding: utf-8 -*-
"""导出模块（阶段 5）。

提供 Excel / 图表 / Markdown / Word 四种导出。所有导出均不写入 Cookie、不写入学号姓名等敏感信息。
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# 文件名中不允许出现的字符（Windows 兼容）
_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]+')


class ExportError(Exception):
    """导出异常，携带可读的中文原因。"""


def _timestamp() -> str:
    """返回当前时间戳（用于文件名，形如 20260101_123456）。"""
    return time.strftime("%Y%m%d_%H%M%S")


def safe_filename(name: str) -> str:
    """把任意名称转为安全文件名（去除非法字符、去首尾空白）。"""
    cleaned = _INVALID_FILENAME_CHARS.sub("", str(name)).strip()
    return cleaned or "output"


# ---------------------------------------------------------------------------
# 1. Excel（多 sheet）
# ---------------------------------------------------------------------------
def _corr_record(target: str, result: Any) -> dict[str, Any]:
    """把相关性结果对象转为一行 dict（无结果返回 None）。"""
    if result is None or not getattr(result, "valid", False):
        return None
    return {
        "指标": target,
        "Pearson_r": getattr(result, "r", None),
        "R2": getattr(result, "r_squared", None),
        "p值": getattr(result, "p", None),
        "样本量n": getattr(result, "n", None),
        "提示": "；".join(
            [getattr(result, "warning", ""), getattr(result, "caution", "")]
        ).strip("；"),
    }


def export_excel(result_dict: dict[str, Any], path: str | Path) -> Path:
    """导出多 sheet Excel：原始数据、情绪明细、日聚合、月度聚合、相关性、事件前后。

    参数:
        result_dict: 汇总结果 dict，可选键：
            raw_df / emotion_detail / daily / monthly /
            corr_close / corr_pe / event_window。
        path: 输出路径（.xlsx）。

    返回:
        输出文件路径。

    说明:
        不含 Cookie；各 sheet 仅在对应数据存在时写入。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sheets: list[tuple[str, pd.DataFrame]] = []

    def _add(name: str, df: Any) -> None:
        if df is not None and not df.empty:
            sheets.append((name, df.copy()))

    _add("原始数据", result_dict.get("raw_df"))
    _add("情绪明细", result_dict.get("emotion_detail"))
    _add("日聚合", result_dict.get("daily"))
    _add("月度聚合", result_dict.get("monthly"))

    corr_rows = [
        rec
        for rec in (
            _corr_record("情绪-股价", result_dict.get("corr_close")),
            _corr_record("情绪-PE", result_dict.get("corr_pe")),
        )
        if rec is not None
    ]
    if corr_rows:
        sheets.append(("相关性", pd.DataFrame(corr_rows)))

    ew = result_dict.get("event_window")
    if ew is not None and getattr(ew, "valid", False):
        ew_df = pd.DataFrame(
            [
                {
                    "事件日": getattr(ew, "event_date", ""),
                    "窗口天数": getattr(ew, "window_days", ""),
                    "前窗口样本量": getattr(ew, "before_count", 0),
                    "后窗口样本量": getattr(ew, "after_count", 0),
                    "前窗口正向均值": getattr(ew, "before_pos_mean", None),
                    "后窗口正向均值": getattr(ew, "after_pos_mean", None),
                    "前窗口负向均值": getattr(ew, "before_neg_mean", None),
                    "后窗口负向均值": getattr(ew, "after_neg_mean", None),
                    "正向差值": getattr(ew, "pos_diff", None),
                    "负向差值": getattr(ew, "neg_diff", None),
                }
            ]
        )
        sheets.append(("事件前后", ew_df))

    # 保证至少一个 sheet（空导出时写说明页，不报错）
    if not sheets:
        sheets.append(("说明", pd.DataFrame({"说明": ["无可用数据导出"]})))

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info("已导出 Excel：%s（%d 个 sheet）", path, len(sheets))
    return path


# ---------------------------------------------------------------------------
# 2. 图表（词云 PNG + plotly HTML）
# ---------------------------------------------------------------------------
def export_figures(
    figures_dict: dict[str, Any],
    out_dir: str | Path,
    company: str = "",
) -> list[Path]:
    """导出图表：matplotlib Figure → PNG，plotly Figure → HTML。

    参数:
        figures_dict: ``{图名: Figure}``（matplotlib / plotly）。
        out_dir: 输出目录。
        company: 公司名（用于文件名前缀）。

    返回:
        已保存的文件路径列表。None 或无法识别的 Figure 会被跳过。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = _timestamp()
    prefix = f"{safe_filename(company)}_{ts}" if company else ts

    saved: list[Path] = []
    for name, fig in figures_dict.items():
        if fig is None:
            continue
        key = safe_filename(name)
        # matplotlib Figure（有 savefig 方法）
        if hasattr(fig, "savefig"):
            fpath = out_dir / f"{prefix}_{key}.png"
            fig.savefig(fpath, dpi=150, bbox_inches="tight")
            saved.append(fpath)
        # plotly Figure（有 write_html 方法）
        elif hasattr(fig, "write_html"):
            fpath = out_dir / f"{prefix}_{key}.html"
            fig.write_html(fpath)
            saved.append(fpath)
        else:
            logger.warning("跳过无法识别的图对象：%s", name)

    logger.info("已导出图表 %d 个到 %s", len(saved), out_dir)
    return saved


# ---------------------------------------------------------------------------
# 3. Markdown
# ---------------------------------------------------------------------------
def export_markdown(report_md: str, path: str | Path) -> Path:
    """导出 Markdown 文本（UTF-8 编码）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = report_md if report_md else ""
    path.write_text(content, encoding="utf-8")
    logger.info("已导出 Markdown：%s", path)
    return path


# ---------------------------------------------------------------------------
# 4. Word（可选依赖 python-docx）
# ---------------------------------------------------------------------------
def export_word(report_md: str, path: str | Path) -> Path:
    """把 Markdown 结论草稿导出为 Word（依赖 python-docx）。

    参数:
        report_md: Markdown 文本。
        path: 输出路径（.docx）。

    返回:
        输出文件路径。

    异常:
        ExportError: 未安装 python-docx 或写入失败（携带可读原因，调用方提示即可，不崩溃）。
    """
    try:
        import docx
    except ImportError as exc:  # noqa: BLE001 —— 可选依赖，明确提示
        raise ExportError(
            "未安装 python-docx，无法导出 Word。请执行 `pip install python-docx`。"
        ) from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = docx.Document()

    # 设置默认字体，保证中文正常显示
    try:
        style = document.styles["Normal"]
        style.font.name = "宋体"
        style._element.rPr.rFonts.set(
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia", "宋体"
        )
    except Exception as exc:  # noqa: BLE001 —— 字体设置失败不影响导出
        logger.debug("设置 Word 默认字体失败：%s", exc)

    for line in (report_md or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("## "):
            document.add_heading(stripped[3:], level=1)
        elif stripped.startswith("# "):
            document.add_heading(stripped[2:], level=0)
        elif stripped.startswith("- "):
            document.add_paragraph(stripped[2:], style="List Bullet")
        elif stripped.startswith("> "):
            document.add_paragraph(stripped[2:], style="Intense Quote")
        else:
            document.add_paragraph(stripped)

    document.save(path)
    logger.info("已导出 Word：%s", path)
    return path
