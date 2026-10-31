# -*- coding: utf-8 -*-
"""行情数据获取模块。

五级回退顺序（每级失败均记录日志后降级，绝不崩溃）：
    ① akshare.stock_zh_a_hist（东方财富，收盘价 + 尝试 PE）
    ② akshare.stock_zh_a_daily（新浪，仅收盘价）
    ③ 腾讯历史 K 线（仅收盘价，无 PE）
    ④ data/templates/market_{code}.csv（内置样例，date/close/pe）
    ⑤ 用户上传行情表（界面下拉映射列名 → date/close/pe）

当最终数据只有 close、没有 pe 时，返回 ``has_pe=False``，由相关性模块降级为
「情绪-股价相关性」，并在界面注明“当前数据源不含 PE”，而不是报错。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)

# 行情来源标识
ORIGIN_AKHIST = "akshare-东方财富"
ORIGIN_AKDAILY = "akshare-新浪"
ORIGIN_TENCENT = "腾讯"
ORIGIN_BUILTIN = "内置CSV"
ORIGIN_UPLOAD = "上传"

# 获取模式
MODE_AUTO = "自动"          # 按五级回退依次尝试
MODE_AKSHARE = "akshare"    # 仅尝试 akshare 通道（①②）
MODE_BUILTIN = "内置CSV"    # 仅读内置样例
MODE_UPLOAD = "上传"        # 仅读上传文件

# 行情表统一列名
MARKET_COLUMNS: tuple[str, ...] = ("date", "close", "pe")

# 各通道 → 统一列名的别名映射（界面/文件里常见的中英文列名）
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "date": ("date", "日期", "时间", "trade_date"),
    "close": ("close", "收盘", "收盘价"),
    "pe": ("pe", "PE", "市盈率", "市盈率(pe)", "市盈率（pe）"),
}


class MarketDataError(Exception):
    """行情获取异常，携带可读的中文原因。"""


@dataclass
class MarketResult:
    """一次行情获取的结果。

    属性:
        df: 统一列 ``date, close, pe`` 的行情数据（pe 可能整列为 NaN）。
        data_origin: 行情来源。
        has_pe: 是否含有效 PE 数据（False 时相关性模块降级为股价相关性）。
        message: 供界面展示的说明文字。
    """

    df: pd.DataFrame
    data_origin: str
    has_pe: bool
    message: str = ""

    @property
    def empty(self) -> bool:
        """行情是否完全缺失。"""
        return self.df is None or self.df.empty


def _empty_result(message: str) -> MarketResult:
    """构造“行情完全缺失”的空结果（不抛异常，供界面隐藏相关性模块）。"""
    return MarketResult(
        df=pd.DataFrame(columns=list(MARKET_COLUMNS)),
        data_origin="",
        has_pe=False,
        message=message,
    )


def _code_prefix(code: str) -> str:
    """由股票代码推断交易所前缀（用于新浪/腾讯等需要前缀的接口）。"""
    if code.startswith(("6", "9")):
        return "sh"
    if code.startswith(("0", "2", "3")):
        return "sz"
    if code.startswith(("4", "8")):
        return "bj"
    return ""


def build_symbol(code: str, market: str = "") -> str:
    """拼接行情接口所需的带前缀股票代码（如 ``sz002415``）。

    规则（代码推断优先，market 仅作一致性校验）：
    1. 先按 code 开头推断前缀：6/9→sh、0/2/3→sz、4/8→bj。
    2. 若显式 market 非空，转前缀后与推断结果比对；
       冲突时以代码推断为准，并记录 warning。
    3. market 为空直接使用推断结果。
    4. 4/8 开头的北交所代码推断为 bj 时记 info（akshare/腾讯接口支持有限，
       由上层在失败时正常降级，不视为异常）。

    参数:
        code: 股票代码（如 "002415"）。
        market: 市场代码（SZ/SH/BJ）。

    返回:
        形如 ``sz002415`` / ``sh600519`` 的字符串；code 为空时返回空串。
    """
    code = str(code or "").strip().zfill(6)
    if not code:
        return ""

    inferred_prefix = _code_prefix(code)

    # 北交所代码：接口支持有限，仅记 info，失败时由上层正常降级
    if inferred_prefix == "bj":
        logger.info(
            "代码 %s 推断为北交所前缀 bj，akshare/腾讯接口支持有限，失败时将正常降级", code
        )

    if not market:
        return f"{inferred_prefix}{code}"

    market_prefix = config.market_prefix(market)
    if market_prefix and market_prefix != inferred_prefix:
        logger.warning(
            "code=%s 与 market=%s 冲突，已按代码推断为 %s",
            code,
            market,
            inferred_prefix,
        )
    return f"{inferred_prefix}{code}"


def _normalize_market_df(
    df: pd.DataFrame, date_col: str = "date", close_col: str = "close", pe_col: str = "pe"
) -> tuple[pd.DataFrame, bool]:
    """把行情 DataFrame 归一化为统一列 date/close/pe，并返回是否含有效 PE。

    参数:
        df: 原始行情 DataFrame。
        date_col / close_col / pe_col: 源列名。

    返回:
        (归一化后的 df, has_pe)。df 已按日期排序、去重并丢弃 close 为空的行。
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=list(MARKET_COLUMNS)), False

    out = pd.DataFrame()
    out["date"] = df[date_col].astype(str).str.strip()
    out["close"] = pd.to_numeric(df[close_col], errors="coerce")
    out["pe"] = pd.to_numeric(df[pe_col], errors="coerce") if pe_col in df.columns else float("nan")

    out = out.dropna(subset=["close"])          # 无收盘价的行无效
    out = out[out["date"] != ""].reset_index(drop=True)
    out = out.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    has_pe = bool(out["pe"].notna().any()) if "pe" in out.columns else False
    return out, has_pe


def _map_uploaded_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """按列名别名把上传行情表映射为 date/close/pe。

    参数:
        df: 用户上传的原始 DataFrame。

    返回:
        (归一化后的 df, has_pe)。

    异常:
        MarketDataError: 缺少必要的 date 或 close 列。
    """
    mapping: dict[str, str] = {}
    for target, aliases in _COLUMN_ALIASES.items():
        for column in df.columns:
            if str(column).strip() in aliases:
                mapping[target] = column
                break
    if "date" not in mapping or "close" not in mapping:
        raise MarketDataError(
            f"上传行情表缺少必要列，至少需要 date/日期 与 close/收盘 两列。当前列：{list(df.columns)}"
        )
    return _normalize_market_df(
        df, date_col=mapping["date"], close_col=mapping["close"], pe_col=mapping.get("pe", "")
    )


# ---------------------------------------------------------------------------
# 各级数据通道（每个函数失败时应抛异常，由上层捕获并降级）
# ---------------------------------------------------------------------------
# 修复：新增可选 start_date/end_date 参数，未传时回落 config 默认区间（向后兼容）。
def _fetch_akshare_hist(
    code: str, start_date: str | None = None, end_date: str | None = None
) -> tuple[pd.DataFrame, bool]:
    """① akshare.stock_zh_a_hist（东方财富）：收盘价 + 显式获取历史 PE。

    在拿到日线收盘价后，显式调用 ``ak.stock_a_indicator_lg(symbol=code)`` 获取历史
    PE（乐咕乐股指标接口），并统一日期格式后按日期左连接合并；PE 接口失败或返回空时
    捕获异常、记录日志并降级为仅收盘价，绝不让程序崩溃。

    参数:
        code: 股票代码（如 "002415"）。
        start_date / end_date: 行情起止日期（YYYYMMDD）；为 None 时回落 config 默认区间。
    """
    if not config.AKSHARE_ENABLED:
        raise MarketDataError("akshare 通道已关闭")
    import akshare as ak  # 延迟导入，避免未安装时阻塞

    start = start_date or config.MARKET_START
    end = end_date or config.MARKET_END
    hist = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start,
        end_date=end,
        adjust="qfq",
    )
    if hist is None or hist.empty:
        raise MarketDataError("stock_zh_a_hist 返回空数据")

    df = hist.rename(columns={"日期": "date", "收盘": "close"})[["date", "close"]].copy()
    # 统一日期格式为 YYYY-MM-DD，保证与 PE 指标表按日期对齐
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date"])

    # 显式获取历史 PE 并按日期合并（接口失败 / 返回空仅降级，不崩溃）
    try:
        # 优先新版 stock_value_em（东方财富个股估值：含 PE(TTM)）；
        # 旧版 stock_a_indicator_lg 已在 akshare 1.x 移除，保留 hasattr 兜底以防版本差异。
        indicator = None
        try:
            indicator = ak.stock_value_em(symbol=code)
            if indicator is not None and not indicator.empty:
                indicator = indicator.rename(columns={"数据日期": "trade_date", "PE(TTM)": "pe"})
        except Exception as exc:  # noqa: BLE001 —— 新接口失败则尝试旧接口
            logger.debug("stock_value_em 获取 PE 失败：%s", exc)
        if indicator is None and hasattr(ak, "stock_a_indicator_lg"):
            try:
                indicator = ak.stock_a_indicator_lg(symbol=code)
            except Exception as exc:  # noqa: BLE001
                logger.debug("stock_a_indicator_lg 获取 PE 失败：%s", exc)

        if (
            indicator is not None
            and not indicator.empty
            and "trade_date" in indicator.columns
            and "pe" in indicator.columns
        ):
            pe_map = indicator[["trade_date", "pe"]].copy()
            # trade_date 可能为 "20210104"（旧接口）或 "2018-01-02"（新接口），
            # 统一用通用解析（不指定 format），再格式化为 YYYY-MM-DD 便于按日期对齐。
            pe_map["date"] = (
                pd.to_datetime(pe_map["trade_date"].astype(str), errors="coerce")
                .dt.strftime("%Y-%m-%d")
            )
            pe_map = pe_map.dropna(subset=["date"])[["date", "pe"]]
            df = df.merge(pe_map, on="date", how="left")
            logger.info("已为代码 %s 合并历史 PE（共 %d 条）", code, int(pe_map["pe"].notna().sum()))
        else:
            logger.warning("PE 接口返回空或无 pe/trade_date 列，PE 置空")
            df["pe"] = float("nan")
    except Exception as exc:  # noqa: BLE001 —— PE 非必需，失败仅记录并降级
        logger.warning("获取 PE 失败：%s，将降级为仅收盘价", exc)
        df["pe"] = float("nan")

    if "pe" not in df.columns:
        df["pe"] = float("nan")

    return _normalize_market_df(df)


# 修复：新增可选 start_date/end_date 参数，未传时回落 config 默认区间（向后兼容）。
def _fetch_akshare_daily(
    code: str,
    market: str = "",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """② akshare.stock_zh_a_daily（新浪）：仅收盘价。"""
    if not config.AKSHARE_ENABLED:
        raise MarketDataError("akshare 通道已关闭")
    import akshare as ak

    start = start_date or config.MARKET_START
    end = end_date or config.MARKET_END
    symbol = build_symbol(code, market)
    daily = ak.stock_zh_a_daily(
        symbol=symbol,
        start_date=start,
        end_date=end,
        adjust="qfq",
    )
    if daily is None or daily.empty:
        raise MarketDataError("stock_zh_a_daily 返回空数据")
    return daily.rename(columns={"date": "date", "close": "close"})[["date", "close"]]


# 修复：新增可选 start_date/end_date 参数，透传到腾讯 url 参数（逗号分隔）。
def _fetch_tencent_kline(
    code: str,
    market: str = "",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """③ 腾讯历史 K 线：仅收盘价（无 PE）。"""
    start = start_date or config.MARKET_START
    end = end_date or config.MARKET_END
    symbol = build_symbol(code, market)
    url = config.TENCENT_KLINE_URL
    params = {"param": f"{symbol},day,{start},{end},1000,qfq"}
    response = requests.get(url, params=params, timeout=config.MARKET_TIMEOUT)
    response.raise_for_status()
    payload = response.json()

    # 结构：data[symbol]["qfqday"] 或 ["day"]，每项 [date, open, close, high, low, volume, ...]
    node = payload.get("data", {}).get(symbol, {})
    rows = node.get("qfqday") or node.get("day") or []
    if not rows:
        raise MarketDataError("腾讯历史 K 线返回空数据")

    records: list[dict[str, Any]] = []
    for item in rows:
        try:
            records.append({"date": str(item[0]), "close": float(item[2])})
        except (IndexError, ValueError, TypeError) as exc:
            logger.debug("跳过无法解析的腾讯K线条目：%s（%s）", item, exc)
    if not records:
        raise MarketDataError("腾讯历史 K 线解析后无有效数据")
    return pd.DataFrame(records)


def _read_builtin_csv(code: str) -> pd.DataFrame:
    """④ 内置行情样例 data/templates/market_{code}.csv。

    读取后原样保留 date/close/pe 列（PE 列不做裁剪，交由 _normalize_market_df 归一化），
    确保带 PE 的内置样例能被下游识别并绘制 PE 相关性图。
    """
    path = config.market_template_path(code)
    if not path.exists():
        raise MarketDataError(
            f"未找到该公司内置行情样例（{path.name}），请上传行情表或安装 akshare。"
        )
    df = pd.read_csv(path, encoding="utf-8-sig")  # 兼容带 BOM 的 CSV
    # 修复：读到 PE 列时确保其为数值类型并原样保留，避免被误判为「无 PE」而丢图。
    if "pe" in df.columns:
        df["pe"] = pd.to_numeric(df["pe"], errors="coerce")
    return df


def _read_upload_market(uploaded_file: Any) -> pd.DataFrame:
    """⑤ 读取用户上传的行情表（Excel/CSV）。"""
    if uploaded_file is None:
        raise MarketDataError("未提供上传行情表")
    if isinstance(uploaded_file, (str, Path)):
        name = str(uploaded_file)
        source: Any = uploaded_file
    else:
        name = getattr(uploaded_file, "name", "")
        source = uploaded_file

    if name.lower().endswith((".csv", ".tsv")):
        return pd.read_csv(source, encoding="utf-8-sig")
    return pd.read_excel(source)


def _try_level(
    origin: str, loader: Callable[[], Any]
) -> MarketResult | None:
    """执行一级数据通道，失败时记录日志并返回 None（供上层继续降级）。

    参数:
        origin: 该通道的来源标识。
        loader: 返回 ``pd.DataFrame`` 或 ``(df, has_pe)`` 的可调用对象。

    返回:
        成功时返回 MarketResult；失败时返回 None。
    """
    try:
        loaded = loader()
        if isinstance(loaded, tuple):
            df, has_pe = loaded
        else:
            df, has_pe = _normalize_market_df(loaded)

        df, _ = _normalize_market_df(df)  # 二次归一化，确保列/排序一致
        if df.empty:
            raise MarketDataError("该通道解析后无有效数据")

        note = "" if has_pe else "当前数据源不含 PE，将降级为情绪-股价相关性。"
        return MarketResult(df=df, data_origin=origin, has_pe=has_pe, message=note)
    except Exception as exc:  # noqa: BLE001 —— 降级边界，明确记录
        logger.warning("行情通道[%s]失败，降级到下一级：%s", origin, exc)
        return None


# 新增：网络接口返回无 PE 时，尝试读取本地内置 CSV 样例（仅当含 PE 时采用）。
def _try_builtin_with_pe(code: str) -> MarketResult | None:
    """网络接口返回无 PE 时，尝试读取本地内置 CSV 样例（仅当含 PE 时采用）。

    返回含 PE 的 MarketResult；本地样例不存在、读取失败或无 PE 时返回 None。
    """
    path = config.market_template_path(code)
    if not path.exists():
        return None
    try:
        candidate = _read_builtin_csv(code)
        _, norm_has_pe = _normalize_market_df(candidate)
        if not norm_has_pe:
            return None
        logger.info("网络接口无 PE，已自动切换为本地内置 CSV 行情（%s）", path.name)
        return _try_level(ORIGIN_BUILTIN, lambda: candidate)
    except Exception as exc:  # noqa: BLE001 —— 降级边界，明确记录
        logger.warning("读取本地内置 CSV 行情失败：%s", exc)
        return None


def _try_pe_fallback(code: str, uploaded_file: Any = None) -> MarketResult | None:
    """网络接口返回无 PE 时，依次尝试「内置 CSV → 用户上传行情表」补 PE。

    返回含 PE 的 MarketResult；两者都无 PE / 不可用则返回 None。
    """
    builtin = _try_builtin_with_pe(code)
    if builtin is not None:
        return builtin
    if uploaded_file is not None:
        try:
            upload = _try_level(ORIGIN_UPLOAD, lambda: _read_upload_market(uploaded_file))
            if upload is not None and upload.has_pe:
                logger.info("网络接口无 PE，已改用用户上传行情表（含 PE）")
                return upload
        except Exception as exc:  # noqa: BLE001 —— 降级边界，明确记录
            logger.warning("读取用户上传行情表失败：%s", exc)
    return None


# 修复：新增可选 start_date/end_date 参数（YYYYMMDD），为 None 时回落 config 默认区间，
# 并把日期区间透传给 ①/②/③ 各行情通道，避免写死区间导致与评论日期无交集。
def get_market(
    code: str,
    mode: str = MODE_AUTO,
    uploaded_file: Any = None,
    market: str = "",
    start_date: str | None = None,
    end_date: str | None = None,
) -> MarketResult:
    """按五级回退顺序获取行情数据。

    参数:
        code: 股票代码（如 "002415"）。
        mode: 获取模式（自动 / akshare / 内置CSV / 上传）。
        uploaded_file: 上传的行情文件（仅上传模式或自动模式最后一级使用）。
        market: 市场代码（SZ/SH/BJ），用于拼接新浪/腾讯接口的前缀；为空则按代码推断。
        start_date / end_date: 行情起止日期（YYYYMMDD 字符串）；为 None 时回落
            config.MARKET_START / config.MARKET_END（保持向后兼容，现有调用方无需改动）。

    返回:
        MarketResult。行情完全缺失时返回空结果并带提示，不抛异常。
    """
    if not code:
        return _empty_result("未提供股票代码，无法获取行情数据。")

    # 日期参数解析：未传则回落 config 默认区间
    start = start_date if start_date else config.MARKET_START
    end = end_date if end_date else config.MARKET_END

    # ① akshare 东方财富（收盘价 + 尝试 PE）
    if mode in (MODE_AUTO, MODE_AKSHARE):
        result = _try_level(ORIGIN_AKHIST, lambda: _fetch_akshare_hist(code, start, end))
        if result:
            if result.has_pe:
                return result
            # 修复（B2）：网络无 PE 时，若本地内置 CSV 含 PE 则切换，避免 PE 图缺失。
            builtin = _try_pe_fallback(code, uploaded_file)
            return builtin if builtin else result
        # ② akshare 新浪（仅收盘价）
        result = _try_level(ORIGIN_AKDAILY, lambda: _fetch_akshare_daily(code, market, start, end))
        if result:
            if result.has_pe:
                return result
            builtin = _try_pe_fallback(code, uploaded_file)
            return builtin if builtin else result
        # ③ 腾讯历史 K 线（仅收盘价）
        result = _try_level(ORIGIN_TENCENT, lambda: _fetch_tencent_kline(code, market, start, end))
        if result:
            builtin = _try_pe_fallback(code, uploaded_file)
            return builtin if builtin else result

    # ④ 内置 CSV 样例
    if mode in (MODE_AUTO, MODE_BUILTIN):
        result = _try_level(ORIGIN_BUILTIN, lambda: _read_builtin_csv(code))
        if result:
            return result

    # ⑤ 用户上传
    if mode in (MODE_AUTO, MODE_UPLOAD):
        result = _try_level(ORIGIN_UPLOAD, lambda: _read_upload_market(uploaded_file))
        if result:
            return result

    # 全部通道失败：给出明确可读的提示（演示切换公司时，缺内置样例是常见场景）
    builtin_path = config.market_template_path(code)
    if not builtin_path.exists():
        message = (
            f"未找到该公司内置行情样例（{builtin_path.name}），"
            "请上传行情表或安装 akshare。"
        )
    else:
        message = "行情完全缺失：请提供行情数据（上传或安装 akshare）。"
    return _empty_result(message)
