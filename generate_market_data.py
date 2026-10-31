# -*- coding: utf-8 -*-
"""批量生成内置行情样例数据（课堂演示保障脚本）。

用途：
    一次性读取 companies.json 中的所有公司，调用 akshare 抓取日度行情（收盘价）
    与历史指标（PE），合并成标准列 ``date,close,pe``，并保存为
    ``data/templates/market_{code}.csv``，供仪表盘「内置 CSV」通道直接使用。

运行（在项目根目录）：
    python generate_market_data.py

说明：
    - 复用 ``market_data._fetch_akshare_hist``（收盘价 + ``stock_a_indicator_lg`` 历史 PE 合并）。
    - 逐家抓取、异常捕获、进度打印、请求限速（``time.sleep``），单家失败不影响其余公司。
    - 未安装 akshare 时给出明确提示并退出（不静默失败）。
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import config
from market_data import MarketDataError, _fetch_akshare_hist

logger = logging.getLogger(__name__)


def load_companies(path: Path | None = None) -> list[dict[str, str]]:
    """读取 companies.json 中的公司列表，返回含 name/code/market 的条目列表。"""
    path = Path(path) if path else config.COMPANIES_JSON
    if not path.exists():
        raise FileNotFoundError(f"公司列表文件不存在：{path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return list(data.get("companies", []))


def _check_akshare() -> None:
    """校验 akshare 可用性；未安装时抛出带中文提示的异常。"""
    try:
        import akshare  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("未安装 akshare，请先执行：pip install akshare") from exc


def main() -> int:
    config.setup_logging()
    _check_akshare()

    companies = load_companies()
    if not companies:
        logger.error("companies.json 中没有任何公司条目")
        return 1

    config.TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    fail_count = 0
    total = len(companies)

    print(f"共需生成 {total} 家公司的行情数据：")
    for idx, item in enumerate(companies, start=1):
        name = str(item.get("name") or "")
        code = str(item.get("code") or "").strip()
        if not code:
            logger.warning("[%d/%d] 公司「%s」缺少 code，跳过", idx, total, name)
            fail_count += 1
            continue

        print(f"[{idx}/{total}] 抓取 {name}（{code}）...", end=" ", flush=True)
        try:
            df, has_pe = _fetch_akshare_hist(code)
            if df is None or df.empty:
                raise MarketDataError("抓取结果为空")
            # 统一输出标准列 date,close,pe（保留 PE 为 NaN 的行，供降级展示）
            out = df[["date", "close", "pe"]].copy()
            out_path = config.market_template_path(code)
            out.to_csv(out_path, index=False, encoding="utf-8-sig")
            ok_count += 1
            print(f"OK：{len(out)} 行" + ("，含 PE" if has_pe else "，无 PE"))
        except Exception as exc:  # noqa: BLE001 —— 单家失败不中断，继续下一家
            fail_count += 1
            logger.warning("[%d/%d] 抓取 %s（%s）失败：%s", idx, total, name, code, exc)
            print(f"失败：{exc}")

        # 请求限速：避免高频访问被接口限制
        if idx < total:
            time.sleep(config.REQUEST_DELAY)

    print("\n========== 生成结果汇总 ==========")
    print(f"成功：{ok_count} 家")
    print(f"失败/跳过：{fail_count} 家")
    print(f"输出目录：{config.TEMPLATE_DIR}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
