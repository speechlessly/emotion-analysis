# 上市公司投资者互动文本情绪监测与分拆上市效果分析仪表盘

基于 Streamlit 的一体化交互式仪表盘。用户选择/输入公司与数据源后，系统自动完成：

> 爬取 → 清洗 → 情绪分析 → 词云与图表 → 结论草稿 → 导出报告

本项目由课程项目《海康威视分拆上市效果研究——基于 Python 爬虫和文本挖掘技术》重构升级而来。所有图表由代码生成，不依赖手动制作。

---

## 一、目录结构

```
├── app.py                 # Streamlit 主界面（侧边栏 + 6 个 Tab 编排全流程）
├── config.py              # 路径/字体/超时/重试/限速/阈值/默认公司/缓存目录/行情配置
├── data_source.py         # 评论数据三源统一入口与回退 + 公司下拉合并/元数据解析
├── crawler.py             # 爬虫实现（可选在线模式，上证e互动平台）
├── market_data.py         # 行情五级回退：akshare→akshare新浪→腾讯→内置CSV→上传
├── platform_companies.py  # 平台公司列表加载（可选增强，默认关闭，失败静默回退）
├── cleaner.py             # 清洗、去重、时间解析、匿名化（阶段2）
├── sentiment.py           # 词典加载、匹配、正负占比（阶段2）
├── analysis.py            # 时间聚合、事件前后、相关性、回归（阶段3）
├── visualization.py       # 词云、趋势图、双轴图、散点图（阶段4）
├── report_generator.py    # 规则化结论草稿（阶段4）
├── exporter.py            # Excel/图表/Markdown/Word 导出
├── integration_check.py   # 全流程联调自检脚本（7 个检查点）
├── companies.json         # 预设公司列表（名称/code/market/事件日）
├── data/
│   ├── cache/             # 本地缓存目录（{公司名}.xlsx）
│   ├── dictionary/        # 默认情绪词典（金融领域中文情绪词典.xlsx）
│   └── templates/         # 上传模板 + 内置行情样例 market_002415.csv
├── assets/fonts/          # 中文字体（已内置 Noto Sans CJK，也可自行替换）
├── tests/                 # 各阶段测试用例
├── requirements.txt
└── README.md
```

> 全部模块已交付：`config.py`、`crawler.py`、`data_source.py`、`market_data.py`、
> `platform_companies.py`、`cleaner.py`、`sentiment.py`、`analysis.py`、
> `visualization.py`、`report_generator.py`、`exporter.py`、`app.py`、
> `integration_check.py`、`companies.json`、数据模板、内置行情样例、默认缓存与情绪词典。

---

## 二、快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

启动仪表盘：

```bash
streamlit run app.py
```

### 2. 配置中文字体（词云必需）

`config.resolve_font_path()` 按「`FONT_PATH` → `assets/fonts/` 下任意字体 → 系统常见中文字体」
三级解析字体；项目已内置 Noto Sans CJK（`assets/fonts/NotoSansCJK-Regular.ttc`），开箱即可用。
如需其他字体，把 `SimHei.ttf` / `msyh.ttc` 放入 `assets/fonts/` 或修改 `FONT_PATH`。
字体完全缺失时，词云会降级为提示图（标注“未找到中文字体”），不会崩溃，也不会静默乱码。

### 3. 运行测试

```bash
pytest -q tests/test_phase1.py
pytest -q tests/test_phase1b.py
pytest -q tests/test_phase2.py
pytest -q tests/test_phase3.py
pytest -q tests/test_phase4.py
pytest -q tests/test_phase5.py
# 或一次跑全部
pytest -q tests/

# 全流程联调自检（7 个检查点）
python integration_check.py
```

### 4. 快速验证数据源（Python 交互式）

```python
from data_source import get_data

# 本地缓存模式（默认缓存已内置：data/cache/海康威视.xlsx）
res = get_data("海康威视", "本地缓存")
print(res.data_origin, len(res.df), res.message)

# 在线模式但未填 Cookie —— 自动回退缓存
res = get_data("海康威视", "在线爬取", cookie_str=None)
print(res.data_origin, res.message)
```

---

## 三、数据模板与样例

模板与样例位于 `data/` 目录：

| 文件 | 内容 | 说明 |
|---|---|---|
| `data/templates/数据模板.xlsx` | `评论表`（company/date/question/answer）、`行情表`（date/close/pe） | 评论表为必需，行情表可选 |
| `data/templates/词典模板.xlsx` | `年报负面` / `年报正面` / `社媒负面` / `社媒正面` 四个 sheet | 供其他用户参考的示例模板 |
| `data/dictionary/金融领域中文情绪词典.xlsx` | 真实情绪词典（含「说明」页 + 四个词典 sheet） | 默认词典，界面也可上传覆盖 |
| `data/templates/market_002415.csv` | 海康威视日度行情样例（date/close/pe，2021-2024） | 内置行情样例 |

统一评论表 schema：

| 字段 | 含义 | 示例 |
|---|---|---|
| company | 公司名 | 海康威视 |
| date | 时间 | 2024-01-05 10:30 |
| question | 提问信息 | 请问分拆上市进展？ |
| answer | 回答信息 | 感谢关注，以公告为准 |

> 上传数据列名不必严格一致：后续「字段映射」通过界面下拉框将任意列名映射到统一 schema。

### 情绪词典读取约定

四个词典 sheet（年报负面/年报正面/社媒负面/社媒正面）均按 **`header=None` 读取第一列并 `strip()` 去空白**，
自动跳过「说明」页；`data/dictionary/金融领域中文情绪词典.xlsx` 即为可直接使用的默认词典。

### 内置行情样例说明（重要）

`market_002415.csv` 的 `close` 为公开行情整理（前复权日度收盘价）；**`pe` 为年度近似值
（公开估值数据整理），仅用于课程演示，不代表当日真实市盈率**。相关图表与报告中将标注
“PE 为年度近似值，仅用于课程演示”。

---

## 四、公司列表与行情获取逻辑

### 公司下拉列表

```
下拉框 = companies.json 预设(含 code/market/事件日) + 手动输入
         ⊕ [可选开关]「尝试加载平台公司列表」platform_companies.py
               ├─ 成功 → 合并去重进下拉
               └─ 失败 → 记录日志 + 静默回退预设（不阻塞主流程）
```

- 预设公司：海康威视（002415.SZ）、萤石网络（688475.SH）、贵州茅台（600519.SH）、
  中国平安（601318.SH）、招商银行（600036.SH）、三一重工（600031.SH）。
- `companies.json` 中 market 统一存代码 `SZ/SH/BJ`：界面显示用 `config.market_display_name()` 转中文，
  行情接口用 `market_data.build_symbol()` 转 `sz/sh/bj` 前缀（如 `sz002415`）。
- 用户选择公司后，`resolve_company()` 解析出 code/market 并透传给爬虫与行情模块。

> **数据覆盖限制（界面会注明）**：评论数据来自上证 e 互动平台（沪市），**不承诺“平台全部公司可选”**；
> 海康威视为深市（002415.SZ）股票，其缓存数据系课程演示样例，不代表平台全量覆盖。

### 行情五级回退

```
market_data.get_market(code)
   ① akshare.stock_zh_a_hist（东方财富，收盘价 + 尝试 PE）
   ② akshare.stock_zh_a_daily（新浪，仅收盘价）
   ③ 腾讯历史 K 线（仅收盘价，无 PE）
   ④ data/templates/market_{code}.csv（内置样例，含 close 与年度近似 pe）
   ⑤ 用户上传行情表（列名映射 date/close/pe）
   每级失败 → 记录日志 → 降级到下一级
```

- 最终只有 `close`、没有 `pe` 时，相关性模块降级为「情绪-股价相关性」，隐藏「情绪-PE 相关性」，
  并在界面注明“当前数据源不含 PE”，**不报错**。
- 行情完全缺失时，相关性模块整体隐藏并提示“请提供行情数据或安装 akshare”，其余功能不受影响。

---

## 五、清洗与情绪分析（双模式）

### 数据模式识别

`cleaner.detect_mode(df)` 自动识别两种输入：

| 模式 | 判别特征 | 处理方式 |
|---|---|---|
| 原始模式 | 有 question 提问信息列（answer 仅作兜底） | 加载词典 → match_emotions 匹配 → compute_ratios 计算占比 |
| 已分析模式 | 有四类情绪词列（用户已跑过匹配） | 跳过匹配，parse_emotion_columns 拆分情绪词、parse_ratio_columns 校验占比 |
| 两者都有 | 界面让用户手动选择 | — |
| 都没有 | 提示无法识别 | — |

### 原始模式

- 统一 schema：`company / date / question / answer`。
- `sentiment.load_lexicon()`：四个词典 sheet 按 `header=None` 读第一列并 `strip()`，自动跳过「说明」页。
- `sentiment.match_emotions()` 返回四类命中词列表（含重复，表示词频）；`compute_ratios()` 按命中次数归一化计算正负占比。
- 中文日期解析支持 `2019年02月21日 08:27`，无法解析的行默认丢弃并记录日志。

### 已分析模式

- 用户上传表（如 `时间、回答信息、四类情绪词、正向占比、负向占比`）直接复用已算好的占比，**不重算、不重新分词**。
- 四类情绪词按顿号 `、`（兼容中英文逗号/分号/空格等）拆成词列表，空值返回空列表；
  词云使用这四列词频（`emotion_word_frequency`），不对 answer 重新分词。
- 占比列经 `clip_ratio` 校验到 [0,1]；注意：占比约定为 0~1 的小数。
- 可选一致性校验：若用户同时上传了词典，可抽样重跑匹配比对，但不作为主流程。

---

## 六、统计与建模

`analysis.py` 提供以下函数（统计口径固定）：

| 函数 | 作用 | 关键口径 |
|---|---|---|
| `aggregate_daily(df)` | 情绪按日聚合 | 输出 `date/pos_ratio/neg_ratio/n` |
| `correlate(daily, market, target)` | 情绪占比 × 行情相关性 | 按 date 内连接；`scipy.stats.pearsonr` 输出 r、p、n、R²；交集空返回空 |
| `event_window_compare(df, event_date, window_days)` | 事件前后对比 | 前后窗口均值/样本量/差值；一侧为 0 不参与差值 |
| `linear_fit(x, y)` | 单变量线性回归 | n≥30 才出拟合线与 R² |
| `optional_grouped_correlate(daily, market, event_date)` | 可选分组相关性 | 按事件日分前后两组，默认关闭 |

> 口径说明：`aggregate_daily` 同日多条按**简单平均**聚合，未按文本长度加权；
> 与原报告“字符长度占比”口径不同，本仪表盘的情绪占比仅作趋势参考。

强制规则：

- **n < 30** → 结果带 `warning="样本量不足（n<30），相关性仅供参考"`，界面与结论都必须显示。
- **R² < 0.1** → 结果带 `caution="解释力有限，不宜作因果推断"`，图注与结论生成器都强制带。
- 行情有 close 无 pe → PE 相关性返回 `None`（界面隐藏 PE 图），股价相关性正常算，不报错。
- 不做：多元回归、格兰杰因果、滞后相关（课程项目不必要且易被质疑）。

---

## 七、Cookie 获取与合规说明（重要）

1. **Cookie 为选填项**：只有「填写 Cookie 且勾选『尝试在线爬取』」时才会发起在线请求；
   不填或爬取失败会自动回退到本地缓存；缓存也没有则提示上传数据。
2. **获取方式**：以浏览器登录上证 e 互动平台（`sns.sseinfo.com`）后，按 F12 打开开发者工具
   → Network 面板 → 刷新页面 → 在任意请求的 Request Headers 中复制完整 `Cookie` 值。
3. **合规红线**：
   - Cookie 仅用于当前会话内的数据抓取，**绝不写入日志、绝不写入任何导出文件**；
   - 爬取已内置限速（每页间隔）、重试与超时，请勿擅自调低间隔或高频访问；
   - 仅限个人学习与学术研究用途，遵守目标网站服务条款与 robots 约定；
   - 所抓取的问答文本版权归平台与发布者所有，请勿用于商业用途。

---

## 八、AI 使用说明模板（学术诚信）

本项目在开发过程中使用了 AI 辅助工具，如需在论文/报告中声明，可参考以下模板：

> 本研究使用生成式 AI 工具（如代码助手、文本生成模型）辅助完成代码编写、数据清洗脚本
> 生成与部分文档草稿撰写。AI 生成内容均已由作者逐项审阅、校验与修正；所有数据分析、
> 结论推导与观点表述由作者独立完成，作者对全文内容负全部责任。研究中引用的数据均来自
> 公开渠道（上证 e 互动平台等），不包含 AI 虚构或未经核实的数据。

---

## 九、工程约定

- 模块化分层，禁止把逻辑堆在单文件；函数均含 docstring 与类型提示。
- 禁止硬编码 Cookie、公司名、页码、行数；可变参数全部来自 `config.py` 或界面。
- 禁止 `except: pass`；异常必须记录日志或返回明确错误信息。
- 中间结果可缓存，但流程不依赖固定文件名链式传递（用参数显式传递，而非约定文件名）。
- 结论由规则模板生成且每条带数值；`R²<0.1` 时必须写明“解释力有限，不宜作因果推断”，不编造数据。
