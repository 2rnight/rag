#!/usr/bin/env python3
"""生成 A4 单页速查表（FastAPI 部分）—— 排版引擎 + 内容，单文件可跑。

为什么自己画 PDF：
    这份速查表要塞进**严格一页 A4**，还要密排三栏、代码行带底色、中文避头尾。
    用 Markdown/HTML 转 PDF 会受渲染器默认样式与分页策略影响，很难保证「恰好一页」。
    所以这里用 reportlab 直接按坐标画，并做两件事保证质量：
      1) 自动缩放：先在 measure 模式量高度，二分选出「刚好放得下」的最大字号；
      2) 版式断言：每行都校验是否超出列宽、是否以标点开头（中文避头尾），越界就报出来。

依赖：
    pip install reportlab
中文字体（任选其一，或用 --font 指定）：
    Ubuntu/Debian: apt install fonts-wqy-microhei   → /usr/share/fonts/truetype/wqy/wqy-microhei.ttc
    或下载:        https://github.com/anthonyfok/fonts-wqy-microhei
    macOS:         /System/Library/Fonts/PingFang.ttc（自带）
    Windows:       C:/Windows/Fonts/msyh.ttc（自带）

用法：
    python -m scripts.build_cheatsheet                       # 自动找字体
    python -m scripts.build_cheatsheet --font /path/font.ttc --out CHEATSHEET-fastapi.pdf
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont

FONT = "WQY"

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]



from reportlab.lib.colors import HexColor
from reportlab.pdfbase.pdfmetrics import stringWidth

PAGE_W, PAGE_H = 595.28, 841.89            # A4 (pt)
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 17.0, 14.0, 13.0
GAP = 6.5
COLS = 3
COL_W = (PAGE_W - 2 * MARGIN_X - GAP * (COLS - 1)) / COLS

FONT = "WQY"
BOLD_W = 0.30                               # 伪粗体描边宽度
BASE = {"title": 15.0, "sub": 7.4, "head": 9.0, "body": 8.3, "code": 7.8}
BASE_LEAD = {"body": 11.3, "code": 10.7}

C_TITLE = HexColor("#12233f")
C_RULE = HexColor("#c9d3e4")
C_H_BG = HexColor("#22375e")
C_H_FG = HexColor("#ffffff")
C_BODY = HexColor("#16233a")
C_DIM = HexColor("#5b6b85")
C_CODE_BG = HexColor("#eef2f9")
C_CODE_FG = HexColor("#12406e")
C_KEY = HexColor("#8a1f1f")
C_TB_BG = HexColor("#f6f8fc")


def asc(size: float) -> float:
    """文字基线以上的近似高度（用于给行盒/底色块定位）。"""
    return 0.80 * size


def desc(size: float) -> float:
    """文字基线以下的近似深度。"""
    return 0.26 * size


def width(text: str, size: float, bold: bool = False) -> float:
    return stringWidth(text, FONT, size) + (BOLD_W if bold else 0)


NO_START_CJK = "，。；、！？）」》】："


def _units(text: str) -> list[str]:
    """切成不可分单元：连续 ASCII 视为整体；标点/收尾引号粘连到前一个单元。

    这样贪心装箱时，标点天然不会落在行首（中文避头尾）。
    """
    units: list[str] = []
    i = 0
    word_chars = "_-./:[](),=<>*+'\"#@"
    while i < len(text):
        ch = text[i]
        if ch.isascii() and (ch.isalnum() or ch in word_chars):
            j = i
            while j < len(text) and text[j].isascii() and (text[j].isalnum() or text[j] in word_chars):
                j += 1
            unit, i = text[i:j], j
        else:
            unit, i = ch, i + 1
        while i < len(text) and text[i] in NO_START_CJK + "「《（":
            unit += text[i]
            i += 1
        units.append(unit)
    return units


def wrap(text: str, size: float, max_w: float, bold: bool = False) -> list[str]:
    """按渲染宽度折行：不可分单元贪心装箱；超长 ASCII 标识符硬断（并保持避头尾）。"""
    lines: list[str] = []
    cur = ""
    for unit in _units(text):
        if width(unit, size, bold) > max_w:
            if cur.strip():
                lines.append(cur.rstrip())
                cur = ""
            part = ""
            for ch in unit:
                if not part or width(part + ch, size, bold) <= max_w:
                    part += ch
                    continue
                if ch in NO_START_CJK and len(part) > 1:
                    # 标点不能另起一行：把前一个字符一起下移（新行从头开始，不会溢出）
                    lines.append(part[:-1].rstrip())
                    part = part[-1] + ch
                else:
                    lines.append(part.rstrip())
                    part = ch
            cur = part
            continue
        if not cur or width(cur + unit, size, bold) <= max_w:
            cur += unit
        else:
            lines.append(cur.rstrip())
            cur = "" if unit.isspace() else unit
    if cur.strip():
        lines.append(cur.rstrip())
    return lines


class Sheet:
    def __init__(self, path: str | None = None, scale: float = 1.0, measure: bool = False) -> None:
        self.scale = scale
        self.measure = measure
        self.violations: list[str] = []
        self.soft: list[str] = []
        self.c = None
        if not measure:
            from reportlab.pdfgen import canvas

            self.c = canvas.Canvas(path, pagesize=(PAGE_W, PAGE_H))
            self.c.setTitle("RAG 知识库问答系统 · FastAPI 速查表")

    # ---------- 尺寸换算
    def sz(self, key: str) -> float:
        return BASE[key] * self.scale

    def lead(self, key: str = "body") -> float:
        return BASE_LEAD[key] * self.scale

    # ---------- 基础绘制
    def text(self, x: float, y: float, s: str, size: float, *, bold: bool = False, color=C_BODY) -> None:
        if self.measure:
            return
        t = self.c.beginText(x, y)
        t.setFont(FONT, size)
        t.setFillColor(color)
        if bold:
            self.c.setLineWidth(BOLD_W)
            self.c.setStrokeColor(color)
            t.setTextRenderMode(2)
        t.textOut(s)
        self.c.drawText(t)

    def rect(self, x: float, y: float, w: float, h: float, color, radius: float = 2.0) -> None:
        if self.measure:
            return
        self.c.setFillColor(color)
        self.c.roundRect(x, y, w, h, radius, stroke=0, fill=1)

    def line(self, x1: float, y1: float, x2: float, y2: float, color=C_RULE, lw: float = 0.4) -> None:
        if self.measure:
            return
        self.c.setStrokeColor(color)
        self.c.setLineWidth(lw)
        self.c.line(x1, y1, x2, y2)

    def _guard(self, line: str, size: float, x: float, limit: float, bold: bool = False) -> None:
        over = width(line, size, bold) + x - limit
        if over > 4.0:
            self.violations.append(f"{line!r} x={x:.1f} 超过 {limit:.1f} (+{over:.1f}pt)")
        elif over > 0.5 and line and line[-1] in NO_START_CJK:
            self.soft.append(f"{line[-6:]!r} +{over:.1f}pt")

    # ---------- 小节标题条
    def _head(self, x: float, y: float, title: str) -> float:
        size = self.sz("head")
        lines = wrap(title, size, COL_W - 9.4, bold=True)
        line_h = size * 1.20
        h = len(lines) * line_h + 5.4
        bar_bottom = y - h + 2.6
        self.rect(x, bar_bottom, COL_W, h, C_H_BG, radius=2.6)
        ty = y - size * 0.92
        for ln in lines:
            self.text(x + 4.6, ty, ln, size, bold=True, color=C_H_FG)
            self._guard(ln, size, x + 4.6, x + COL_W - 3.0, True)
            ty -= line_h
        return bar_bottom - (asc(self.sz("body")) + 3.2)

    # ---------- 内容块：rows = [(kind, text)]，kind ∈ b/c/s/k
    def section(self, x: float, y: float, title: str, rows: list[tuple[str, str]]) -> float:
        y = self._head(x, y, title)
        body, code = self.sz("body"), self.sz("code")
        lead_b, lead_c = self.lead("body"), self.lead("code")
        for kind, raw in rows:
            if kind == "k":                                     # 键：值（键高亮，值折行）
                key, _, val = raw.partition("|")
                kw = width(key, body, True) + width(": ", body)
                avail = COL_W - 5.0 - kw - 3.0
                for n, part in enumerate(wrap(val, body, avail)):
                    if n == 0:
                        self.text(x + 5.0, y, key, body, bold=True, color=C_KEY)
                        self._guard(key, body, x + 5.0, x + 5.0 + kw, True)
                    self.text(x + 5.0 + kw, y, part, body, color=C_BODY)
                    self._guard(part, body, x + 5.0 + kw, x + COL_W)
                    y -= lead_b
                continue
            indent, prefix, size, lead, color = 5.0, "• ", body, lead_b, C_BODY
            if kind == "c":
                size, lead, color = code, lead_c, C_CODE_FG
            elif kind == "s":
                color = C_DIM
            pw = width(prefix, size) if kind == "b" else 0.0
            avail = COL_W - indent - pw - 3.0
            parts = wrap(raw, size, avail)
            for n, part in enumerate(parts):
                tx = x + indent + (pw if (n == 0 or kind == "c") else 0.0)
                if kind == "c" and n == 0:
                    self.rect(x, y - desc(size) - 1.6, COL_W, asc(size) + desc(size) + 3.2, C_CODE_BG, radius=1.2)
                if n == 0 and kind == "b":
                    self.text(x + indent, y, "•", size, color=C_DIM)
                self.text(tx, y, part, size, color=color)
                self._guard(part, size, tx, x + COL_W)
                y -= lead
            if kind == "c":
                y -= 1.2
        return y - 4.2

    # ---------- 两列小表格
    def table(self, x: float, y: float, rows: list[tuple[str, str]], *, lw_ratio: float = 0.42) -> float:
        body, lead = self.sz("body"), self.lead("body")
        lw, rw = COL_W * lw_ratio, COL_W * (1 - lw_ratio)
        for i, (left, right) in enumerate(rows):
            ll = wrap(left, body, lw - 6.0, bold=True)
            rl = wrap(right, body, rw - 6.2)
            rows_n = max(len(ll), len(rl))
            h = lead * rows_n + 3.4
            # 行的第一行基线在 y，向下延伸 (rows_n-1) 行 → 底色块必须向下算
            box_top = y + asc(body) + 1.7
            box_bottom = y - lead * (rows_n - 1) - desc(body) - 1.7
            if i % 2 == 0:
                self.rect(x, box_bottom, COL_W, box_top - box_bottom, C_TB_BG, radius=1.0)
            self.line(x + lw, box_bottom, x + lw, box_top)
            for n, s in enumerate(ll):
                self.text(x + 3.2, y - lead * n, s, body, bold=True, color=C_KEY)
                self._guard(s, body, x + 3.2, x + lw, True)
            for n, s in enumerate(rl):
                self.text(x + lw + 3.4, y - lead * n, s, body, color=C_BODY)
                self._guard(s, body, x + lw + 3.4, x + COL_W)
            y -= h
        return y - 4.2

    def save(self) -> None:
        if not self.measure:
            self.c.showPage()
            self.c.save()


COLX = [MARGIN_X + i * (COL_W + GAP) for i in range(COLS)]
FULL_W = COL_W * COLS + GAP * (COLS - 1)
TOP = PAGE_H - MARGIN_TOP
BODY_TOP = TOP - 31.0                     # 页眉之下
FOOTER_H = 19.0                           # 底部口诀条
BOTTOM = MARGIN_BOTTOM + FOOTER_H
AVAILABLE = BODY_TOP - BOTTOM

# ------------------------------------------------------------------ 内容
C1 = [
    ("① 应用组装 · app.py", [
        ("c", "FastAPI(title, version, lifespan=..., openapi_tags=...)"),
        ("b", "lifespan(@asynccontextmanager)：ensure_dirs → build_pipeline(strict=False) → 挂 app.state.pipeline"),
        ("k", "降级|加载失败先标记 ready=False，服务照常启动（可用性优先，不让坏索引拖死整个服务）"),
        ("c", 'app.include_router(router, prefix="/api/v1")'),
        ("b", "/health、/metrics、/、/favicon.ico 不带前缀 + include_in_schema=False → 探针可直接打，/docs 干净"),
        ("k", "中间件|CORSMiddleware 默认 [\"*\"]、credentials=False；@app.middleware(\"http\") 记访问日志与耗时"),
        ("s", "流式响应的耗时只记到「开始返回」，真实端到端耗时看 done.total_ms"),
        ("b", "启动顺序：ensure_dirs → 加载索引并重建 BM25 → 挂到 app.state；关闭时无需清理"),
        ("k", "异常|@app.exception_handler(RagError) → JSON{error, message, detail}"),
        ("s", "状态码由异常类自带：415 类型不支持 / 422 解析失败 / 409 索引未就绪 / 502 上游失败"),
    ]),
    ("② 依赖注入 · api/deps.py", [
        ("c", "with _lock:   # threading.Lock + 模块级 _pipeline 单例"),
        ("b", "防「并发首请求同时构建两份索引」的竞态；FAISS/BM25 是内存对象，绝不能每请求重建"),
        ("c", "def get_pipeline() -> Iterator[RagPipeline]: ... yield pipeline"),
        ("b", "生成器依赖可做请求级清理；这里统一取用与错误语义"),
        ("b", "ready=False 且索引非空 → HTTPException(409)：宁可不服务，也不喂半残数据"),
        ("c", "pipeline_from_request(request)   # /health 这类「必须能应答」的端点用它"),
        ("b", "收益：FAISS/BM25 只加载一次；放 app.state 又让测试能替换成已索引好的流水线"),
        ("b", "实例放在 app.state 而非模块全局 → 测试可注入替身（TestClient 的 with 会触发 lifespan）"),
    ]),
]

C2 = [
    ("③ 端点契约 · routes + schemas", [
        ("b", "13 个操作全带 tags / summary / response_model → /docs 里可直接交互调试"),
        ("c", "question: Field(min_length=1, max_length=1000) + field_validator 去空白"),
        ("c", "top_k ge=1 le=20 · threshold ge=0 le=1 · history max_length=20"),
        ("c", 'role: Literal["user", "assistant"]'),
        ("b", "非法输入自动 422 并给字段级提示，无需手写校验代码"),
        ("b", "上传：list[UploadFile]、后缀白名单、50MB 上限、逐文件容错、finally close"),
        ("b", "删除：removed_chunks==0 → 404，符合 HTTP 语义"),
        ("b", "rebuild 的服务端目录参数经 _resolve_server_path() 前缀校验，越界 400（防目录穿越）"),
        ("b", "POST /chat/stream 是主链路（可带 history、可 AbortController 取消）；GET 版兼容原生 EventSource"),
    ]),
    ("④ 异步模型（最容易被追问）", [
        ("t", "路由调同步逻辑|await run_in_threadpool(...)"),
        ("t", "生成器里检索|await asyncio.to_thread(...)"),
        ("t", "上游 LLM 流式|原生 async，不占线程"),
        ("t", "毫秒级内存计算|直接跑，不必进池"),
    ]),
    ("线程池与并发账", [
        ("k", "令牌|anyio 默认 40 个；池满时新任务排队，事件循环不阻塞"),
        ("b", "同步代码写在 async def 里 = 冻住整个事件循环，连别人的 SSE 心跳都发不出去"),
        ("b", "慢操作走 async IO（不占线程）＋ 重 CPU 进线程池（受 40 限制）→ SSE 能挂很多连接"),
        ("k", "多 worker|内存索引每个 worker 各存一份且互不同步（上传只落一个）→ 单 worker + 多副本 + 共享存储，或索引外置"),
    ]),
]

C3 = [
    ("⑤ SSE 实现 · api/sse.py", [
        ("c", 'media_type="text/event-stream"  +  X-Accel-Buffering: no'),
        ("c", "Cache-Control: no-cache, no-transform  +  Content-Encoding: identity"),
        ("b", "少一个 X-Accel-Buffering，Nginx 就把流攒成一次性输出 —— 「本地快、上线慢」的元凶"),
        ("c", "queue = asyncio.Queue(maxsize=64); task = create_task(produce())"),
        ("b", "生产/消费解耦：才有空隙插心跳；maxsize 顺带提供背压"),
        ("c", 'await asyncio.wait_for(queue.get(), timeout=15)  →  超时 yield ": ping"'),
        ("b", "心跳是 SSE 注释行（客户端自动忽略），防网关 60s 读超时掐断长连接"),
        ("b", "异常也走事件：event: error —— 响应头已发出就改不了状态码，所以接口永远 200，不甩 5xx"),
        ("c", "finally: task.cancel() + 吞掉 CancelledError / GeneratorExit"),
        ("b", "客户端断开必须取消上游，否则连接泄漏 + 白烧 token"),
    ]),
    ("⑥ 前端：为什么手写 SSE 而不用 EventSource", [
        ("b", "EventSource 只能 GET、不能带请求体 → 改用 fetch + response.body.getReader()"),
        ("b", "按 \\n\\n 切帧；末段留作 buffer；跳过 \":\" 开头的行；AbortController 取消"),
        ("s", "答法：SSE 是协议不是库，浏览器原生只有 GET 版，所以按协议自己解析"),
    ]),
    ("⑦ 追问速答", [
        ("t", "为何选 FastAPI？|原生 async + 校验 + DI + 自动文档"),
        ("t", "async 里写阻塞？|事件循环被占，全站排队"),
        ("t", "两个 to_thread？|Starlette 封装 vs 标准库"),
        ("t", "流式为何不返 5xx？|响应头已发，改不了"),
        ("t", "多 worker 怎么办？|单 worker + 多副本 + 共享存储"),
        ("t", "怎么测接口？|TestClient 打 ASGI + curl -N"),
    ]),
]


def build(scale: float, measure: bool) -> tuple[float, list[str]]:
    sh = Sheet(OUT, scale=scale, measure=measure)
    if not measure:
        ts = BASE["title"] * scale
        ss = BASE["sub"] * scale
        sh.text(MARGIN_X, TOP - 12.8, "FastAPI 接口层 · 速查表", ts, bold=True, color=C_TITLE)
        pn = "RAG 企业知识库智能问答系统"
        sh.text(MARGIN_X + FULL_W - width(pn, ss * 1.25), TOP - 12.0, pn, ss * 1.25, bold=True, color=C_TITLE)
        left_txt = "app.py · api/deps.py · api/routes.py · api/sse.py"
        sh.text(MARGIN_X, TOP - 22.4, left_txt, ss, color=C_DIM)
        tag_items = ["12 路径 / 13 操作", "线程池 40", "心跳 15s", "队列 64", "上传 50MB", "41 用例通过"]
        avail = FULL_W - width(left_txt, ss) - 12.0
        while tag_items:
            tag = " · ".join(tag_items)
            if width(tag, ss) <= avail:
                break
            tag_items.pop()
        sh.text(MARGIN_X + FULL_W - width(tag, ss), TOP - 22.4, tag, ss, color=C_DIM)
        sh.line(MARGIN_X, TOP - 26.6, MARGIN_X + FULL_W, TOP - 26.6, lw=0.7)
        sh.rect(MARGIN_X, MARGIN_BOTTOM - 1.0, FULL_W, FOOTER_H - 2.0, HexColor("#fdf6e3"), radius=2.4)
        sh.text(MARGIN_X + 5.0, MARGIN_BOTTOM + 3.0,
                "口诀：组装在 lifespan · 单例靠 Lock · 阻塞进线程池 · 流式永远 200 · 断开要 cancel · 降级不崩",
                BASE["body"] * min(scale, 1.0) * 0.95, bold=True, color=HexColor("#7a4b00"))

    deepest = BODY_TOP
    col_ends: list[float] = []
    for col, blocks in ((COLX[0], C1), (COLX[1], C2), (COLX[2], C3)):
        y = BODY_TOP
        for title, rows in blocks:
            if any(k == "t" for k, _ in rows):
                y = sh._head(col, y + 1.0, title)
                ratio = 0.46 if title.startswith("④") else 0.42
                y = sh.table(col, y + 1.0, [(t.split("|")[0], t.split("|")[1]) for k, t in rows if k == "t"],
                             lw_ratio=ratio)
            else:
                y = sh.section(col, y, title, rows)
        col_ends.append(y)
        deepest = min(deepest, y)
    if measure:
        sh.col_ends = col_ends  # type: ignore[attr-defined]
    sh.save()
    return deepest, sh.violations


def find_font(explicit: str | None) -> str:
    if explicit:
        if not Path(explicit).exists():
            raise SystemExit(f"字体文件不存在: {explicit}")
        return explicit
    env = os.environ.get("RAG_CHEATSHEET_FONT")
    if env and Path(env).exists():
        return env
    for cand in FONT_CANDIDATES:
        if Path(cand).exists():
            return cand
    raise SystemExit(
        "未找到可用的中文字体。请显式指定：\n"
        "    python -m scripts.build_cheatsheet --font /path/to/wqy-microhei.ttc\n"
        "字体可从 https://github.com/anthonyfok/fonts-wqy-microhei 获取，"
        "或 apt install fonts-wqy-microhei，或设置环境变量 RAG_CHEATSHEET_FONT。"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 A4 单页速查表（FastAPI 部分）")
    parser.add_argument("--font", default=None, help="中文字体文件（TTF/TTC）")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "CHEATSHEET-fastapi.pdf"),
                        help="输出 PDF 路径")
    args = parser.parse_args(argv)

    font_path = find_font(args.font)
    pdfmetrics.registerFont(TTFont(FONT, font_path, subfontIndex=0))
    globals()["OUT"] = args.out

    best = None
    scale = 1.0
    while scale <= 2.0:
        deepest, violations = build(scale, measure=True)
        used = BODY_TOP - deepest
        if used <= AVAILABLE:
            best = (scale, used, violations)
        else:
            break
        scale += 0.01
    if best is None:
        print("内容放不下（即使最小字号）", file=sys.stderr)
        return 1
    scale, used, violations = best
    build(scale, measure=False)
    print(f"已生成: {args.out}")
    print(f"  字体: {font_path}")
    print(f"  字号缩放: {scale:.2f}（正文字号约 {BASE['body'] * scale:.1f}pt）")
    print(f"  内容高度: {used:.1f}pt / 可用 {AVAILABLE:.1f}pt  填充率 {used / AVAILABLE:.0%}")
    print(f"  版式告警: {violations or '无'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
