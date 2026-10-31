# -*- coding: utf-8 -*-
"""全局配置中心。

集中管理项目路径、中文字体路径、网络请求参数、情绪阈值、默认公司与缓存目录等。
所有可变参数一律从这里读取，禁止在各业务模块中散落硬编码。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
# 项目根目录（config.py 所在目录）
BASE_DIR: Final[Path] = Path(__file__).resolve().parent

# 数据目录与子目录
DATA_DIR: Final[Path] = BASE_DIR / "data"
CACHE_DIR: Final[Path] = DATA_DIR / "cache"          # 本地缓存目录（在线爬取落盘）
DEMO_DIR: Final[Path] = DATA_DIR / "demo"            # 演示数据目录（已分析的四类情绪词表）
TEMPLATE_DIR: Final[Path] = DATA_DIR / "templates"    # 上传模板目录
LOG_DIR: Final[Path] = BASE_DIR / "logs"              # 日志目录

# 静态资源目录
ASSET_DIR: Final[Path] = BASE_DIR / "assets"
FONT_DIR: Final[Path] = ASSET_DIR / "fonts"           # 中文字体目录

# 中文字体路径（词云 / matplotlib 中文显示用）。
# 说明：项目不随包分发字体，请将 SimHei.ttf / msyh.ttc 等中文字体放入 assets/fonts 目录，
# 或在此处直接指定系统中已安装字体的绝对路径。若路径不存在，词云模块会给出明确报错，
# 而不会静默回退为乱码。
FONT_PATH: Final[Path] = FONT_DIR / "SimHei.ttf"

# 预设公司列表 JSON 文件
COMPANIES_JSON: Final[Path] = BASE_DIR / "companies.json"

# 日志文件
LOG_FILE: Final[Path] = LOG_DIR / "app.log"

# ---------------------------------------------------------------------------
# 网络请求配置（爬虫限速 / 重试 / 超时）
# ---------------------------------------------------------------------------
REQUEST_TIMEOUT: Final[int] = 15          # 单次请求超时（秒）
REQUEST_RETRIES: Final[int] = 3           # 单页失败后的最大重试次数
REQUEST_DELAY: Final[float] = 1.5         # 每页之间的限速间隔（秒）
REQUEST_BACKOFF: Final[float] = 2.0       # 重试退避系数（指数退避）

# 通用 User-Agent（可在界面/配置覆盖，不硬编码到爬虫内部）
USER_AGENT: Final[str] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
)

# 爬取目标站点（上证 e 互动平台，样例数据来源站点）
BASE_URL: Final[str] = "https://sns.sseinfo.com"
SEARCH_URL: Final[str] = BASE_URL + "/searchFullText.do"
POST_URL: Final[str] = BASE_URL + "/qasearchFullText.do"

# ---------------------------------------------------------------------------
# 行情数据配置（market_data.py 五级回退）
# ---------------------------------------------------------------------------
AKSHARE_ENABLED: Final[bool] = True       # 是否启用 akshare 行情通道
MARKET_TIMEOUT: Final[int] = 30           # 行情接口单次请求超时（秒）
MARKET_START: Final[str] = "20210101"     # akshare 历史行情起始日期
MARKET_END: Final[str] = "20241231"       # akshare 历史行情结束日期

# 腾讯历史K线接口（第③级备用通道，仅提供收盘价，无 PE）
TENCENT_KLINE_URL: Final[str] = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

# 上证 e 互动平台公司列表接口（可选增强；留空则仅用 akshare 股票映射）
PLATFORM_COMPANY_URL: Final[str] = ""

# ---------------------------------------------------------------------------
# 市场标识（统一存 SZ/SH/BJ，界面显示时再转中文）
# ---------------------------------------------------------------------------
MARKET_SZ: Final[str] = "SZ"   # 深交所
MARKET_SH: Final[str] = "SH"   # 上交所
MARKET_BJ: Final[str] = "BJ"   # 北交所

# 市场代码 → 中文展示名（界面下拉/标签显示用）
MARKET_DISPLAY_NAMES: Final[dict[str, str]] = {
    MARKET_SZ: "深交所",
    MARKET_SH: "上交所",
    MARKET_BJ: "北交所",
}

# 市场代码 → 行情接口交易所前缀（akshare/腾讯/新浪等接口需要 sz002415 这类前缀）
_MARKET_PREFIX_MAP: Final[dict[str, str]] = {
    MARKET_SZ: "sz",
    MARKET_SH: "sh",
    MARKET_BJ: "bj",
    # 兼容旧的中文/小写写法，保证幂等
    "深交所": "sz",
    "上交所": "sh",
    "北交所": "bj",
    "sz": "sz",
    "sh": "sh",
    "bj": "bj",
}


def market_display_name(market: str) -> str:
    """把市场代码（SZ/SH/BJ）转换为中文展示名；空值或已为中文时原样返回。"""
    if not market:
        return ""
    return MARKET_DISPLAY_NAMES.get(market, market)


def market_prefix(market: str) -> str:
    """把市场代码（SZ/SH/BJ）转换为行情接口前缀（sz/sh/bj）；无法识别返回空串。"""
    if not market:
        return ""
    return _MARKET_PREFIX_MAP.get(market, "")

# ---------------------------------------------------------------------------
# 情绪分析阈值
# ---------------------------------------------------------------------------
# 正向 / 负向占比的判定阈值（占比 = 命中情绪词字数 / 文本总字数）
POSITIVE_THRESHOLD: Final[float] = 0.0    # 大于该值记为“偏正向”
NEGATIVE_THRESHOLD: Final[float] = 0.0    # 大于该值记为“偏负向”

# ---------------------------------------------------------------------------
# 默认公司与数据 schema
# ---------------------------------------------------------------------------
DEFAULT_COMPANY: Final[str] = "海康威视"

# 评论表统一 schema（英文列名，界面下拉框做字段映射）
COMMENT_COLUMNS: Final[tuple[str, ...]] = ("company", "date", "question", "answer")

# 行情表 schema（可选）
MARKET_COLUMNS: Final[tuple[str, ...]] = ("date", "close", "pe")

# 情绪词典的四个 sheet 名（顺序固定）
DICT_SHEETS: Final[tuple[str, ...]] = ("年报负面", "年报正面", "社媒负面", "社媒正面")

# 情绪词典目录与默认词典文件（读取约定：header=None 读第一列并 strip，跳过「说明」页）
DICT_DIR: Final[Path] = DATA_DIR / "dictionary"
DEFAULT_DICT_PATH: Final[Path] = DICT_DIR / "金融领域中文情绪词典.xlsx"

# 内置行情样例命名规则：data/templates/market_{code}.csv
MARKET_TEMPLATE_PATTERN: Final[str] = "market_{code}.csv"

# 界面数据预览默认显示行数（避免在 app.py 中硬编码）
PREVIEW_ROWS: Final[int] = 10

# 事件前后对比默认窗口天数
DEFAULT_WINDOW_DAYS: Final[int] = 180


def market_template_path(code: str) -> Path:
    """返回指定股票代码的内置行情样例路径（data/templates/market_{code}.csv）。"""
    return TEMPLATE_DIR / MARKET_TEMPLATE_PATTERN.format(code=code)


def resolve_font_path() -> Path | None:
    """解析可用的中文字体路径（优先配置，其次字体目录，再次系统常见字体）。

    返回:
        存在的字体文件路径；都找不到时返回 None（调用方据此降级并给出提示，不崩溃）。
    """
    if FONT_PATH.exists():
        return FONT_PATH
    # 字体目录下任意字体文件
    for pattern in ("*.ttf", "*.ttc", "*.otf"):
        candidates = sorted(FONT_DIR.glob(pattern))
        if candidates:
            return candidates[0]
    # 系统常见中文字体（Linux 路径 + Windows 路径）
    system_candidates = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for path in system_candidates:
        if path.exists():
            return path
    return None


def ensure_dirs() -> None:
    """确保运行所需的目录都存在（不存在则创建）。"""
    for directory in (CACHE_DIR, DEMO_DIR, TEMPLATE_DIR, FONT_DIR, LOG_DIR, DICT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def setup_logging(level: int = logging.INFO) -> None:
    """初始化日志：同时输出到控制台与日志文件。

    参数:
        level: 日志级别，默认 INFO。
    """
    ensure_dirs()
    root = logging.getLogger()
    if root.handlers:  # 避免重复初始化
        return

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    root.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
