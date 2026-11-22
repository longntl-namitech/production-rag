"""Draw the README diagrams in a hand-drawn style.

    uv run python docs/diagrams/make_diagrams.py

Standard library only. The output is deterministic (fixed seeds), so a re-run changes an SVG
only where the drawing code or its text changed. The font is Patrick Hand (SIL OFL 1.1, see
OFL.txt), cut down to Latin characters and embedded, so GitHub shows it without loading
anything.
"""

import base64
import html
import json
import math
import random
import textwrap
from pathlib import Path

HERE = Path(__file__).parent
INK = "#1f1f1f"
FILLS = {
    "lilac": "#ECE3F8",
    "yellow": "#FFF3C4",
    "green": "#DFF2DC",
    "blue": "#DCEAF8",
    "pink": "#FBE1E3",
    "paper": "#FBFAF6",
}
FONT = "Hand, 'Patrick Hand', 'Comic Sans MS', cursive"

# Line icons in a 40 x 40 box, drawn with the ink stroke and no fill.
ICONS = {
    "doc": '<path d="M9 4 H26 L33 11 V36 H9 Z M26 4 V11 H33 M14 18 H28 M14 24 H28 M14 30 H23"/>',
    "folder": '<path d="M4 11 H15 L19 15 H36 V34 H4 Z"/>',
    "scissors": (
        '<circle cx="11" cy="30" r="5"/><circle cx="27" cy="30" r="5"/>'
        '<path d="M14 26 L31 5 M24 26 L7 5"/>'
    ),
    "bolt": '<path d="M23 3 L9 22 H19 L15 37 L31 15 H21 Z"/>',
    "layers": '<path d="M20 5 L36 13 L20 21 L4 13 Z M4 20 L20 28 L36 20 M4 27 L20 35 L36 27"/>',
    "search": '<circle cx="16" cy="16" r="10"/><path d="M24 24 L35 35"/>',
    "trophy": (
        '<path d="M11 5 H29 V14 A9 9 0 0 1 11 14 Z M11 8 H5 A6 6 0 0 0 12 17 '
        'M29 8 H35 A6 6 0 0 1 28 17 M20 23 V30 M13 35 H27 M15 30 H25"/>'
    ),
    "pencil": '<path d="M8 32 L29 11 L34 16 L13 37 L6 38 Z M25 15 L30 20"/>',
    "sparkle": '<path d="M20 4 L23 17 L36 20 L23 23 L20 36 L17 23 L4 20 L17 17 Z"/>',
    "bubble": '<path d="M5 7 H35 V27 H18 L10 34 V27 H5 Z M11 14 H29 M11 20 H24"/>',
    "chart": '<path d="M5 5 V35 H36 M9 28 L17 19 L24 24 L34 11"/>',
    "box": '<path d="M5 12 L20 5 L35 12 V29 L20 36 L5 29 Z M5 12 L20 19 L35 12 M20 19 V36"/>',
}


class Sketch:
    def __init__(self, width: int, height: int, seed: int):
        self.width = width
        self.height = height
        self.rng = random.Random(seed)
        self.parts: list[str] = []

    def _j(self, amp: float = 1.5) -> float:
        return self.rng.uniform(-amp, amp)

    def _stroke(self, d: str, width: float = 2.4, opacity: float = 1.0, dash: str = "") -> None:
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<path d="{d}" fill="none" stroke="{INK}" stroke-width="{width}" '
            f'stroke-opacity="{opacity}" stroke-linecap="round" stroke-linejoin="round"{extra}/>'
        )

    def _seg(self, x1: float, y1: float, x2: float, y2: float, bow: float = 2.0) -> str:
        """A cubic from the current point to (x2, y2) that bows a little, like a pen line."""
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length
        b1, b2 = self._j(bow), self._j(bow)
        c1 = (x1 + dx / 3 + nx * b1, y1 + dy / 3 + ny * b1)
        c2 = (x1 + 2 * dx / 3 + nx * b2, y1 + 2 * dy / 3 + ny * b2)
        return f"C {c1[0]:.1f} {c1[1]:.1f} {c2[0]:.1f} {c2[1]:.1f} {x2:.1f} {y2:.1f}"

    def _rounded(self, x: float, y: float, w: float, h: float, r: float) -> str:
        j = self._j
        return " ".join(
            [
                f"M {x + r:.1f} {y + j(0.8):.1f}",
                self._seg(x + r, y, x + w - r, y),
                f"Q {x + w:.1f} {y:.1f} {x + w + j(0.8):.1f} {y + r:.1f}",
                self._seg(x + w, y + r, x + w, y + h - r),
                f"Q {x + w:.1f} {y + h:.1f} {x + w - r:.1f} {y + h + j(0.8):.1f}",
                self._seg(x + w - r, y + h, x + r, y + h),
                f"Q {x:.1f} {y + h:.1f} {x + j(0.8):.1f} {y + h - r:.1f}",
                self._seg(x, y + h - r, x, y + r),
                f"Q {x:.1f} {y:.1f} {x + r:.1f} {y:.1f}",
            ]
        )

    def text(
        self,
        x: float,
        y: float,
        s: str,
        size: float,
        anchor: str = "start",
        bold: bool = False,
        rotate: float = 0,
    ) -> None:
        extra = f' stroke="{INK}" stroke-width="{size / 40:.2f}"' if bold else ""
        if rotate:
            extra += f' transform="rotate({rotate} {x:.1f} {y:.1f})"'
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{INK}"{extra}>{html.escape(s)}</text>'
        )

    def icon(self, name: str, x: float, y: float, scale: float = 1.0) -> None:
        self.parts.append(
            f'<g transform="translate({x:.1f} {y:.1f}) scale({scale})" fill="none" '
            f'stroke="{INK}" stroke-width="2.2" stroke-linecap="round" '
            f'stroke-linejoin="round">{ICONS[name]}</g>'
        )

    def _content(self, x, y, title, bullets, icon, title_size, size) -> None:
        titles = [title] if isinstance(title, str) else list(title)
        tx = x + 22
        if icon:
            self.icon(icon, x + 18, y + 20)
            tx = x + 70
        ty = y + 50
        for line in titles:
            self.text(tx, ty, line, title_size, bold=True)
            ty += title_size + 6
        ty += 8
        for bullet in bullets:
            self.text(x + 24, ty, f"• {bullet}", size)
            ty += size + 11

    def box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        title,
        bullets=(),
        icon: str | None = None,
        title_size: float = 30,
        size: float = 22,
        r: float = 22,
    ) -> None:
        self.parts.append(f'<path d="{self._rounded(x, y, w, h, r)}" fill="{FILLS[fill]}"/>')
        self._stroke(self._rounded(x, y, w, h, r))
        self._stroke(self._rounded(x, y, w, h, r), width=1.2, opacity=0.5)
        self._content(x, y, title, bullets, icon, title_size, size)

    def label_box(
        self, x: float, y: float, w: float, h: float, fill: str, label: str, size: float = 22
    ) -> None:
        """A small box with one centred line, for lists of names."""
        self.parts.append(f'<path d="{self._rounded(x, y, w, h, 14)}" fill="{FILLS[fill]}"/>')
        self._stroke(self._rounded(x, y, w, h, 14), width=2.0)
        self.text(x + w / 2, y + h / 2 + size / 3, label, size, anchor="middle")

    def frame(self, x: float, y: float, w: float, h: float, label: str) -> None:
        self.parts.append(f'<path d="{self._rounded(x, y, w, h, 26)}" fill="{FILLS["paper"]}"/>')
        self._stroke(self._rounded(x, y, w, h, 26), width=2.0, dash="14 10")
        self.text(x + 28, y + 44, label, 30, bold=True)

    def cylinder(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        title,
        bullets=(),
        icon: str | None = None,
    ) -> None:
        rx, ry = w / 2, 20
        body = (
            f"M {x} {y + ry} L {x} {y + h - ry} A {rx} {ry} 0 0 0 {x + w} {y + h - ry} "
            f"L {x + w} {y + ry} A {rx} {ry} 0 0 0 {x} {y + ry} Z"
        )
        self.parts.append(f'<path d="{body}" fill="{FILLS[fill]}"/>')
        for width, opacity in ((2.4, 1.0), (1.2, 0.5)):
            top_ry = ry + self._j(1.2)
            self._stroke(
                f"M {x} {y + ry} A {rx} {top_ry:.1f} 0 0 1 {x + w} {y + ry} "
                f"A {rx} {top_ry:.1f} 0 0 1 {x} {y + ry}",
                width,
                opacity,
            )
            self._stroke(
                f"M {x} {y + ry} {self._seg(x, y + ry, x, y + h - ry)} "
                f"A {rx} {ry + self._j(1.2):.1f} 0 0 0 {x + w} {y + h - ry} "
                f"{self._seg(x + w, y + h - ry, x + w, y + ry)}",
                width,
                opacity,
            )
        self._content(x, y + 2 * ry - 8, title, bullets, icon, 30, 22)

    def arrow(self, *points, dash: str = "", head: float = 16) -> None:
        d = f"M {points[0][0]:.1f} {points[0][1]:.1f} " + " ".join(
            self._seg(a[0], a[1], b[0], b[1], bow=1.4)
            for a, b in zip(points, points[1:], strict=False)
        )
        self._stroke(d, dash=dash)
        (x1, y1), (x2, y2) = points[-2], points[-1]
        angle = math.atan2(y2 - y1, x2 - x1)
        for side in (-1, 1):
            a = angle + math.pi + side * 0.45
            self._stroke(
                f"M {x2:.1f} {y2:.1f} L {x2 + head * math.cos(a):.1f} {y2 + head * math.sin(a):.1f}"
            )

    def dashed_rule(self, y: float) -> None:
        self._stroke(f"M 30 {y} {self._seg(30, y, self.width - 30, y, bow=1)}", dash="20 14")

    def person(self, cx: float, top: float, label: str) -> None:
        self._stroke(
            f"M {cx + 18} {top + 18} A 18 18 0 1 1 {cx + 17.9} {top + 17} "
            f"M {cx} {top + 36} {self._seg(cx, top + 36, cx, top + 92)} "
            f"M {cx - 30} {top + 58} {self._seg(cx - 30, top + 58, cx + 30, top + 58)} "
            f"M {cx} {top + 92} L {cx - 26} {top + 134} M {cx} {top + 92} L {cx + 26} {top + 134}"
        )
        self.text(cx, top + 170, label, 26, anchor="middle")

    def heading(self, title: str, subtitle: str) -> None:
        self.text(40, 70, title, 52, bold=True)
        self.text(40, 116, subtitle, 25)

    def svg(self) -> str:
        font = base64.b64encode((HERE / "PatrickHand-Latin.ttf").read_bytes()).decode()
        style = (
            "@font-face{font-family:Hand;src:url(data:font/ttf;base64,"
            f"{font}) format('truetype');}}text{{font-family:{FONT};}}"
        )
        return "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} '
                f'{self.height}" width="{self.width}" height="{self.height}">',
                f"<style>{style}</style>",
                f'<rect width="{self.width}" height="{self.height}" fill="#FFFFFF"/>',
                *self.parts,
                "</svg>",
            ]
        )


def pipeline() -> Sketch:
    s = Sketch(2000, 1285, seed=1)
    s.heading(
        "production-rag: from files to answers",
        "Each box is a method folder or a core module; each one is measured on public data.",
    )
    s.text(40, 200, "INGEST (offline)", 32, bold=True)
    y, h, w, gap = 230, 250, 270, 52
    xs = [40 + i * (w + gap) for i in range(6)]
    s.box(xs[0], y, w, h, "lilac", "Files", ["PDF, DOCX, PPTX", "XLSX, HTML", "Markdown"], "folder")
    s.box(
        xs[1], y, w, h, "green", "Parse", ["to Markdown", "tables kept", "pip-only parsers"], "doc"
    )
    s.box(
        xs[2],
        y,
        w,
        h,
        "green",
        "Chunk",
        ["fixed, recursive", "heading-aware", "semantic"],
        "scissors",
    )
    s.box(
        xs[3],
        y,
        w,
        h,
        "yellow",
        ["Contextual", "headers"],
        ["one LLM sentence", "per chunk, cached"],
        "pencil",
    )
    s.box(
        xs[4],
        y,
        w,
        h,
        "green",
        "Embed",
        ["dense: Jina v5 nano", "sparse: BM25", "en + vi stopwords"],
        "bolt",
    )
    s.cylinder(
        xs[5],
        y - 10,
        300,
        h + 20,
        "blue",
        "Qdrant",
        ["dense + sparse", "vectors", "title + tags"],
        "box",
    )
    for a, b in zip(xs, xs[1:], strict=False):
        s.arrow((a + w + 8, y + h / 2), (b - 10, y + h / 2))

    s.dashed_rule(545)
    s.text(40, 615, "QUERY (online)", 32, bold=True)
    y, h, w = 650, 270, 300
    xs = [170 + i * (w + gap) for i in range(5)]
    s.person(90, y + 40, "question")
    s.arrow((130, y + h / 2), (xs[0] - 10, y + h / 2))
    s.box(
        xs[0],
        y,
        w,
        h,
        "yellow",
        ["Query", "rewrite"],
        ["multi_query", "HyDE", "step-back", "decomposition"],
        "sparkle",
    )
    s.box(
        xs[1],
        y,
        w,
        h,
        "green",
        "Retrieve",
        ["dense, BM25", "hybrid fusion", "deadline fan-out"],
        "search",
    )
    s.box(xs[2], y, w, h, "green", "Rerank", ["cross-encoder", "over a wider pool"], "trophy")
    s.box(
        xs[3],
        y,
        w,
        h,
        "green",
        "Context",
        ["neighbour chunks", "or the full doc", "compression"],
        "layers",
    )
    s.box(xs[4], y, w, h, "yellow", "Answer", ["LLM answer", "fact-check verdict"], "bubble")
    for a, b in zip(xs, xs[1:], strict=False):
        s.arrow((a + w + 8, y + h / 2), (b - 10, y + h / 2))
    qdrant_x = 40 + 5 * (270 + gap) + 150
    s.arrow((qdrant_x, 492), (qdrant_x, 590), (xs[1] + w / 2, 590), (xs[1] + w / 2, y - 10))
    s.text((qdrant_x + xs[1] + w / 2) / 2, 580, "search", 24, anchor="middle")

    s.box(
        xs[1],
        1000,
        3 * w + 2 * gap,
        210,
        "green",
        "Eval (core/eval)",
        [
            "retrieval: SciFact, TVPL (recall, nDCG)",
            "answers: mini_wiki, vi_legal (LLM judge)",
            "tokens estimated before any LLM run",
        ],
        "chart",
    )
    s.arrow((xs[2] + w / 2, 995), (xs[2] + w / 2, y + h + 10), dash="10 8")
    s.text(xs[2] + w / 2 + 14, 965, "measures every box", 24)
    s.cylinder(
        xs[4], 990, w, 230, "blue", "results/", ["one table per run", "linked from READMEs"], "doc"
    )
    s.arrow((xs[1] + 3 * w + 2 * gap + 8, 1105), (xs[4] - 10, 1105))
    s.text(1960, 1268, "yellow: needs OPENAI_API_KEY", 22, anchor="end")
    return s


METHODS = [
    "naive_rag",
    "dense",
    "bm25",
    "hybrid_fusion",
    "rerank",
    "multi_query",
    "context_expansion",
    "deadline_fanout",
    "parse_and_chunk",
    "fact_check",
    "hyde",
    "step_back",
    "query_decomposition",
    "contextual_headers",
    "contextual_compression",
    "llm_rerank",
]
NEEDS_LLM = {
    "naive_rag",
    "fact_check",
    "hyde",
    "step_back",
    "query_decomposition",
    "contextual_headers",
    "contextual_compression",
    "llm_rerank",
}
CORE = [
    "embedders",
    "sparse (BM25)",
    "rerankers",
    "store (Qdrant)",
    "indexing",
    "chunking",
    "semantic_chunking",
    "pipeline (search, merge)",
    "llm + token budget",
    "query_llm",
    "cache (on disk)",
]


def layout() -> Sketch:
    s = Sketch(2000, 1140, seed=2)
    s.heading(
        "Repository layout",
        "src/production_rag: methods use core; core/eval runs the methods on public data.",
    )
    fx, fw = 150, 1810
    s.frame(fx, 160, fw, 330, "methods/  (one folder each)")
    w, h, gap = 330, 66, 26
    for i, name in enumerate(METHODS):
        row, col = divmod(i, 5)
        fill = "yellow" if name in NEEDS_LLM else "green"
        s.label_box(fx + 28 + col * (w + gap), 230 + row * (h + 18), w, h, fill, name)
    s.arrow((1055, 498), (1055, 556))
    s.text(1075, 535, "use", 24)

    s.frame(fx, 565, fw, 255, "core/  (shared building blocks)")
    w, gap = 275, 22
    for i, name in enumerate(CORE):
        row, col = divmod(i, 6)
        s.label_box(fx + 25 + col * (w + gap), 635 + row * (h + 22), w, h, "blue", name, size=21)

    s.frame(fx, 870, fw, 220, "core/eval/  (measurement)")
    steps = [
        ("lilac", "datasets", "downloaded at runtime"),
        ("green", "run / run_llm", "retrieval metrics"),
        ("green", "answer", "LLM judge, token budget"),
        ("blue", "report", "results/*.md"),
    ]
    w, gap = 380, 60
    for i, (fill, title, line) in enumerate(steps):
        x = fx + 40 + i * (w + gap)
        s.box(x, 930, w, 130, fill, title, [line], title_size=28, size=22)
        if i:
            s.arrow((x - gap + 8, 995), (x - 10, 995))
    s.arrow((fx - 4, 980), (85, 980), (85, 325), (fx - 10, 325))
    s.text(70, 690, "runs the methods", 24, anchor="middle", rotate=-90)
    s.text(1960, 1125, "yellow: needs OPENAI_API_KEY", 22, anchor="end")
    return s


def add_method() -> Sketch:
    s = Sketch(2000, 400, seed=3)
    s.heading("Adding a method", "The same four steps for every folder.")
    steps = [
        (
            "lilac",
            "New folder",
            ["methods/<name>/", "README.md, pipeline.py", "example.py"],
            "folder",
        ),
        ("yellow", "Register", ["add the name to", "METHOD_NAMES"], "pencil"),
        (
            "green",
            "Measure",
            ["SciFact + TVPL", "run.py or run_llm.py", "estimate tokens first"],
            "chart",
        ),
        ("blue", "Report", ["eval.report writes", "results/*.md", "table in the README"], "doc"),
    ]
    w, gap = 410, 70
    for i, (fill, title, bullets, icon) in enumerate(steps):
        x = 75 + i * (w + gap)
        s.box(x, 170, w, 190, fill, title, bullets, icon)
        if i:
            s.arrow((x - gap + 8, 265), (x - 10, 265))
    return s


# (label, SciFact (file, key), TVPL (file, key)); numbers are read from results/*.json.
RETRIEVAL = [
    ("No LLM at query time", None, None),
    ("naive_rag", ("scifact_naive_rag", "naive_rag"), ("tvpl_naive_rag", "naive_rag")),
    ("dense", ("scifact", "dense"), ("tvpl", "dense")),
    ("bm25", ("scifact", "bm25"), ("tvpl", "bm25")),
    ("hybrid_rrf", ("scifact", "hybrid_rrf"), ("tvpl", "hybrid_rrf")),
    ("hybrid_weighted_rrf", ("scifact", "hybrid_weighted_rrf"), ("tvpl", "hybrid_weighted_rrf")),
    ("hybrid_rrf_rerank", ("scifact", "hybrid_rrf_rerank"), ("tvpl", "hybrid_rrf_rerank")),
    ("LLM query rewrite (gpt-5.4-nano)", None, None),
    ("multi_query", ("query_side_scifact", "multi_query"), ("query_side_tvpl", "multi_query")),
    (
        "hyde (passage + question)",
        ("query_side_scifact", "hyde_passage_query"),
        ("query_side_tvpl", "hyde_passage_query"),
    ),
    ("step_back", ("query_side_scifact", "step_back"), ("query_side_tvpl", "step_back")),
    (
        "query_decomposition",
        ("query_decomposition_scifact", "query_decomposition"),
        ("query_side_tvpl", "query_decomposition"),
    ),
]


def _ndcg(ref: tuple[str, str]) -> float:
    path = HERE.parent.parent / "results" / f"{ref[0]}.json"
    return json.loads(path.read_text())["summary"][ref[1]]["ndcg@10"]


def results_chart() -> Sketch:
    lo, hi = 0.60, 0.85
    x0, x1 = 560, 1680
    s = Sketch(2000, 1120, seed=4)
    s.heading(
        "Retrieval at a glance",
        "nDCG@10 on SciFact (English, 300 queries) and TVPL (Vietnamese legal, 1,000 queries).",
    )

    def px(v: float) -> float:
        return x0 + (v - lo) / (hi - lo) * (x1 - x0)

    def circle(cx: float, cy: float) -> None:
        s.parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="13" fill="{FILLS["blue"]}"/>')
        s._stroke(f"M {cx + 13:.1f} {cy:.1f} A 13 13 0 1 1 {cx + 12.9:.1f} {cy - 1:.1f}")

    def square(cx: float, cy: float) -> None:
        x, y = cx - 13, cy - 13
        s.parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="26" height="26" fill="{FILLS["pink"]}"/>'
        )
        s._stroke(s._rounded(x, y, 26, 26, 3), width=2.2)

    # shape and fill both differ, so the two sets read apart without colour
    circle(1180, 160)
    s.text(1206, 168, "SciFact", 26)
    square(1343, 160)
    s.text(1370, 168, "TVPL", 26)
    s.text(1960, 168, "SciFact / TVPL", 24, anchor="end")

    top, row_h = 215, 62
    y = top
    for label, sci, tvpl in RETRIEVAL:
        if sci is None:
            s.text(40, y + 40, label, 28, bold=True)
            y += row_h
            continue
        cy = y + row_h / 2
        s.text(60, cy + 8, label, 25)
        s._stroke(f"M {x0} {cy} {s._seg(x0, cy, x1, cy, bow=0.6)}", width=1.0, opacity=0.25)
        a, b = _ndcg(sci), _ndcg(tvpl)
        circle(px(a), cy)
        square(px(b), cy)
        s.text(1960, cy + 8, f"{a:.3f} / {b:.3f}", 24, anchor="end")
        y += row_h

    for i in range(6):
        v = lo + i * 0.05
        s._stroke(f"M {px(v):.1f} {top} L {px(v):.1f} {y}", width=1.0, opacity=0.35, dash="6 8")
        s.text(px(v), y + 34, f"{v:.2f}", 22, anchor="middle")
    s.text(
        40,
        y + 84,
        "Query-rewrite rows come from separate runs (2026-11-13). Full tables in results/.",
        22,
    )
    return s


LEGEND = (
    "lilac: input  |  yellow: needs OPENAI_API_KEY  |  green: plain code  |  "
    "blue: Qdrant or shared part  |  e.g.: output of a real run"
)

# The example in every box is read from example_query.json, written by capture_example.py
# (one real SciFact claim, run through each method with gpt-5.4-nano).
EX = json.loads((HERE / "example_query.json").read_text(encoding="utf-8"))
GOLD = EX["gold"]
RET = EX["retrieval"]


def cut(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."


def wrap(text: str, n: int, lines: int) -> list[str]:
    """`text` wrapped to `n` characters, at most `lines` lines, the last one cut with '...'."""
    parts = textwrap.wrap(" ".join(text.split()), n)
    if len(parts) > lines:
        parts = parts[:lines]
        parts[-1] = cut(parts[-1] + " ...", n)
    return parts


def hits(rows: list[dict], n: int, count: int = 3) -> list[str]:
    return [
        cut(f"{i}. {h['doc']} {h['score']:.3f} {h['title']}", n)
        for i, h in enumerate(rows[:count], 1)
    ]


def best(found: dict) -> dict:
    return found["top3"][0]


def rank_of(found: dict) -> str:
    return f"gold doc rank {found['rank']}"


def rank(name: str) -> str:
    return rank_of(RET[name])


BM, NI, HW = (best(RET[k]) for k in ("bm25", "bm25_no_idf", "hybrid_weighted_rrf"))


# One flow per method folder: (subtitle, columns). A column is a list of boxes
# (fill, title, bullets, icon, example lines); boxes in one column run side by side as
# parallel steps.
NR, PC, CH = EX["naive_rag"], EX["parse_and_chunk"], EX["contextual_headers"]
MQ, HY, SB, QD = EX["multi_query"], EX["hyde"], EX["step_back"], EX["query_decomposition"]
RR, CE, CC, DF, FC = (
    EX["rerank"],
    EX["context_expansion"],
    EX["contextual_compression"],
    EX["deadline_fanout"],
    EX["fact_check"],
)
LR = EX["llm_rerank"]
Q = EX["query"]
FLOWS = {
    "naive_rag": (
        "Fixed chunks, dense top 5, one prompt.",
        [
            [
                (
                    "lilac",
                    "Documents",
                    ["raw text"],
                    "doc",
                    [f"doc {GOLD['doc']}, {GOLD['chars']:,} chars", *wrap(GOLD["title"], 34, 2)],
                )
            ],
            [
                (
                    "green",
                    "Chunk",
                    ["fixed size", "up to 800 characters"],
                    "scissors",
                    [f"{GOLD['chars']:,} chars -> {NR['gold_fixed_chunks']} chunks"],
                )
            ],
            [
                (
                    "blue",
                    "Dense search",
                    ["query embedded", "top 5 chunks back"],
                    "search",
                    [
                        cut(f"{i}. {h['doc']} chunk {h['chunk']} ({h['score']:.3f})", 34)
                        for i, h in enumerate(NR["top5"][:3], 1)
                    ],
                )
            ],
            [
                (
                    "yellow",
                    "Generate",
                    ["passages stuffed in prompt", "NO_ANSWER if missing"],
                    "bubble",
                    [f"answer: {NR['answer']}"],
                )
            ],
        ],
    ),
    "parse_and_chunk": (
        "Files to Markdown, Markdown to chunks, chunks to vectors.",
        [
            [
                (
                    "lilac",
                    "Files",
                    ["PDF, DOCX, PPTX", "XLSX, HTML, Markdown"],
                    "folder",
                    [f"doc {GOLD['doc']} as .md", f"{PC['chars']:,} chars (title + text)"],
                )
            ],
            [
                (
                    "green",
                    "Parse",
                    ["one library per format", "output is Markdown"],
                    "doc",
                    ["Markdown is read as is"],
                )
            ],
            [
                (
                    "green",
                    "Chunk",
                    ["fixed, recursive, markdown", "semantic (embeds sentences)"],
                    "scissors",
                    [
                        f"fixed 800: {PC['fixed_800']} chunks",
                        f"recursive 800: {PC['recursive_800']} chunks",
                        f"semantic: {PC['semantic']} chunks",
                    ],
                )
            ],
            [
                (
                    "blue",
                    "Index",
                    ["dense vector per chunk", "BM25 vectors per chunk"],
                    "layers",
                    [
                        f"recursive: {PC['recursive_800']} chunks ->",
                        f"{PC['recursive_800']} dense + {PC['recursive_800']} BM25 vectors",
                    ],
                )
            ],
        ],
    ),
    "contextual_headers": (
        "One LLM-written sentence places each chunk in its document.",
        [
            [
                (
                    "lilac",
                    "Documents",
                    ["chunked at index time"],
                    "doc",
                    [
                        f"doc {GOLD['doc']}: {CH['chunks']} chunks",
                        "chunk 0 starts:",
                        f"'{cut(CH['chunk'], 40)}'",
                    ],
                )
            ],
            [
                (
                    "yellow",
                    "Context sentence",
                    ["LLM reads document + chunk", "200 token cap, cached on disk"],
                    "sparkle",
                    wrap(CH["context"], 50, 4),
                )
            ],
            [
                (
                    "blue",
                    "Index",
                    [
                        "title + sentence + chunk",
                        "embedded and BM25-encoded",
                        "stored text stays the chunk",
                    ],
                    "layers",
                    ["embedded: title + sentence + chunk", "stored: the raw chunk only"],
                )
            ],
        ],
    ),
    "multi_query": (
        "Each subquery retrieves on its own, each chunk keeps its best score.",
        [
            [
                (
                    "lilac",
                    "Subqueries",
                    ["up to 8", "embedded in one call"],
                    "bubble",
                    [
                        "1. the original claim",
                        *(f"{i}. {cut(s, 31)}" for i, s in enumerate(MQ["queries"][1:], 2)),
                    ],
                )
            ],
            [
                (
                    "green",
                    "Search each",
                    ["same spec for every", "subquery"],
                    "search",
                    [f"{len(MQ['queries'])} queries -> {len(MQ['queries'])} ranked lists"],
                )
            ],
            [
                (
                    "green",
                    "merge_max",
                    ["each chunk once", "with its best score"],
                    "layers",
                    [
                        f"doc {GOLD['doc']}: rank {MQ['rank']},",
                        f"score {MQ['top3'][0]['score']:.3f} (its best)",
                    ],
                )
            ],
            [
                (
                    "blue",
                    "Rerank (optional)",
                    ["scored against", "rerank_query"],
                    "trophy",
                    ["not used here: the base is", "hybrid_weighted_rrf"],
                )
            ],
        ],
    ),
    "hyde": (
        "An LLM-written passage replaces the question for dense search.",
        [
            [
                (
                    "lilac",
                    "Question",
                    ["short, or phrased unlike", "the documents"],
                    "bubble",
                    wrap(Q, 33, 4),
                )
            ],
            [
                (
                    "yellow",
                    "Passage",
                    ["LLM writes an answer", "256 token cap"],
                    "pencil",
                    wrap(HY["passage"], 33, 4),
                )
            ],
            [
                (
                    "blue",
                    "Dense search",
                    ["passage embedded", "sparse leg skipped"],
                    "search",
                    [
                        f"top hit {best(HY['hyde_passage'])['doc']}",
                        f"({best(HY['hyde_passage'])['score']:.3f}), {rank_of(HY['hyde_passage'])}",
                    ],
                )
            ],
            [
                (
                    "green",
                    "Merge",
                    ["hyde_passage: passage alone", "hyde_passage_query:", "question + passage"],
                    "layers",
                    [
                        f"hyde_passage: {rank_of(HY['hyde_passage'])}",
                        f"hyde_passage_query: {rank_of(HY['hyde_passage_query'])}",
                    ],
                )
            ],
        ],
    ),
    "step_back": (
        "The question and one more general question retrieve together.",
        [
            [("lilac", "Question", ["specific"], "bubble", wrap(Q, 33, 4))],
            [
                (
                    "yellow",
                    "Step-back",
                    ["LLM writes one more", "general question", "128 token cap"],
                    "pencil",
                    wrap(SB["queries"][1], 33, 5),
                )
            ],
            [
                ("green", "Search question", [], "search", ["the original claim"]),
                ("green", "Search step-back", [], "search", ["the step-back question"]),
            ],
            [
                (
                    "green",
                    "Merge",
                    ["each chunk once", "with its best score"],
                    "layers",
                    [
                        f"doc {GOLD['doc']}: rank {SB['rank']},",
                        f"score {SB['top3'][0]['score']:.3f}",
                    ],
                )
            ],
        ],
    ),
    "query_decomposition": (
        "One search per part of the question.",
        [
            [("lilac", "Question", ["asks several things"], "bubble", wrap(Q, 33, 4))],
            [
                (
                    "yellow",
                    "Split",
                    ["LLM writes up to 3", "sub-questions", "256 token cap"],
                    "scissors",
                    [f"{i}. {cut(s, 31)}" for i, s in enumerate(QD["queries"], 1)],
                )
            ],
            [
                ("green", f"Search part {i}", [], "search", wrap(s, 33, 2))
                for i, s in enumerate(QD["queries"], 1)
            ],
            [
                (
                    "green",
                    "Merge",
                    ["each chunk once", "with its best score"],
                    "layers",
                    [
                        f"doc {GOLD['doc']}: rank {QD['rank']},",
                        f"score {QD['top3'][0]['score']:.3f}",
                    ],
                )
            ],
        ],
    ),
    "dense": (
        "One embedding call per request, cosine search in Qdrant.",
        [
            [("lilac", "Query", ["text"], "bubble", wrap(Q, 50, 3))],
            [
                (
                    "green",
                    "Embed",
                    ["jina v5 nano, 768 dims", "or any OpenAI-compatible", "/embeddings endpoint"],
                    "sparkle",
                    ["claim -> 768 numbers, normalised", "(query prompt, not document)"],
                )
            ],
            [
                (
                    "blue",
                    "Qdrant",
                    ["cosine similarity", "top k chunks"],
                    "box",
                    hits(RET["dense"]["top3"], 50),
                )
            ],
        ],
    ),
    "bm25": (
        "Term weights in a sparse vector, with or without IDF.",
        [
            [("lilac", "Query", ["text"], "bubble", wrap(Q, 33, 4))],
            [
                (
                    "green",
                    "Strip stopwords",
                    ["whole-syllable phrases", "bm25_vi_stopwords only"],
                    "scissors",
                    [rank("bm25_vi_stopwords") + ",", f"{rank('bm25')} without"],
                )
            ],
            [
                (
                    "green",
                    "Encode",
                    ["fastembed Qdrant/bm25", "term-frequency weights"],
                    "pencil",
                    [],
                )
            ],
            [
                (
                    "blue",
                    "sparse",
                    ["IDF applied by Qdrant", "bm25"],
                    "box",
                    [
                        f"{rank('bm25')},",
                        f"top hit {best(RET['bm25'])['doc']} ({best(RET['bm25'])['score']:.2f})",
                    ],
                ),
                (
                    "blue",
                    "sparse_noidf",
                    ["no modifier", "bm25_no_idf"],
                    "box",
                    [
                        f"{rank('bm25_no_idf')},",
                        f"top {NI['doc']} ({NI['score']:.2f})",
                    ],
                ),
            ],
        ],
    ),
    "hybrid_fusion": (
        "Two legs fused inside Qdrant, one round trip.",
        [
            [("lilac", "Query", ["text"], "bubble", wrap(Q, 50, 3))],
            [
                (
                    "blue",
                    "Dense leg",
                    ["prefetch, limit k * 4"],
                    "search",
                    [f"{rank('dense')} ({best(RET['dense'])['score']:.3f})"],
                ),
                (
                    "blue",
                    "Sparse leg",
                    ["prefetch, limit k * 4"],
                    "search",
                    [f"{rank('bm25')} (BM25 with IDF)"],
                ),
            ],
            [
                (
                    "green",
                    "Fuse",
                    [
                        "hybrid_rrf: ranks only",
                        "hybrid_dbsf: normalised scores",
                        "hybrid_weighted_rrf: dense 1.0, sparse 0.5",
                    ],
                    "layers",
                    [
                        f"hybrid_rrf: {rank('hybrid_rrf')}",
                        f"hybrid_dbsf: {rank('hybrid_dbsf')}",
                        f"weighted_rrf: {rank('hybrid_weighted_rrf')} ({HW['score']:.3f})",
                    ],
                )
            ],
        ],
    ),
    "rerank": (
        "Retrieve a wider pool, rescore it with a cross-encoder, cut to k.",
        [
            [("lilac", "Query", ["text"], "bubble", wrap(Q, 33, 4))],
            [
                (
                    "blue",
                    "Retrieve",
                    ["hybrid_rrf", "pool min(k * 4, 100)"],
                    "search",
                    [
                        f"k = 10 -> pool of {RR['pool']}",
                        f"{rank_of(RR['before'])} ({RR['before']['top3'][0]['score']:.3f})",
                    ],
                )
            ],
            [
                (
                    "green",
                    "Cross-encoder",
                    ["mmarco-mMiniLMv2, local", "or a Cohere-compatible", "/rerank endpoint"],
                    "trophy",
                    [
                        "scores: "
                        + ", ".join(
                            f"{h['doc']} {h['score']:.2f}" for h in RR["after"]["top3"][:2]
                        ),
                        "(top 2, best first)",
                    ],
                )
            ],
            [
                (
                    "green",
                    "Top k",
                    [
                        "rerank skipped under 300 ms",
                        "of budget left; on error",
                        "retrieval order is kept",
                    ],
                    "layers",
                    [
                        f"gold doc: rank {RR['before']['rank']} -> {RR['after']['rank']}",
                        f"rerank took {RR['rerank_ms']} ms",
                    ],
                )
            ],
        ],
    ),
    "context_expansion": (
        "Neighbour chunks or the whole document, added after ranking.",
        [
            [("lilac", "Query", ["text"], "bubble", wrap(Q, 50, 3))],
            [
                (
                    "blue",
                    "Search and rank",
                    ["scores never change"],
                    "search",
                    [
                        f"top hit: doc {CE['hit']['doc']} chunk {CE['hit']['chunk']}",
                        f"{CE['hit_chars']} chars, score {CE['hit']['score']:.3f}",
                    ],
                )
            ],
            [
                (
                    "green",
                    "Expand",
                    [
                        "ContextOption(prev=n, next=m)",
                        "or ContextOption(full=True)",
                        "one filtered scroll per hit",
                    ],
                    "layers",
                    [
                        f"prev=1, next=1: {CE['hit_chars']} -> {CE['expanded_chars']:,} chars",
                        f"all 5: {CE['total_chars'][0]:,} -> {CE['total_chars'][1]:,} chars",
                    ],
                )
            ],
        ],
    ),
    "llm_rerank": (
        "Retrieve 20 candidates, an LLM scores each one, keep the best 5.",
        [
            [("lilac", "Question", ["text"], "bubble", wrap(Q, 33, 4))],
            [
                (
                    "blue",
                    "Retrieve",
                    ["naive_rag", "800-char chunks, dense top 20"],
                    "search",
                    [
                        f"{LR['candidates']} candidates",
                        f"gold doc: {rank_of(LR['before'])}",
                    ],
                )
            ],
            [
                (
                    "yellow",
                    "Score",
                    ["one LLM call, passages numbered", "reply: a 0 to 10 score each"],
                    "sparkle",
                    ["scores sort the candidates", "ties keep search order"],
                )
            ],
            [
                (
                    "green",
                    "Top 5",
                    ["unusable reply or failed call:", "search order is kept"],
                    "layers",
                    [f"gold doc: rank {LR['before']['rank']} -> {LR['after']['rank']}"],
                )
            ],
        ],
    ),
    "contextual_compression": (
        "An LLM keeps only the sentences that help. It never writes new words.",
        [
            [("lilac", "Question", ["text"], "bubble", wrap(Q, 33, 4))],
            [
                (
                    "blue",
                    "Retrieve",
                    ["naive_rag", "800-char chunks, dense top 5"],
                    "search",
                    [f"{CC['passages'][0]} passages, {CC['chars'][0]:,} chars"],
                )
            ],
            [
                (
                    "yellow",
                    "Pick sentences",
                    ["passages as numbered sentences", "LLM replies with ids to keep"],
                    "sparkle",
                    [f"sentences kept in {CC['passages'][1]} of {CC['passages'][0]} passages"],
                )
            ],
            [
                (
                    "green",
                    "Kept text",
                    ["original sentences, in order", "no kept sentence: passage dropped"],
                    "scissors",
                    [
                        f"{CC['chars'][0]:,} -> {CC['chars'][1]:,} chars",
                        *wrap("first: " + CC["first_kept"], 33, 2),
                    ],
                )
            ],
        ],
    ),
    "deadline_fanout": (
        "One time budget for the whole call, one task per spec.",
        [
            [
                (
                    "lilac",
                    "Request",
                    ["several specs", "deadline_ms, default 4000"],
                    "bolt",
                    ["dense, sparse, hybrid + rerank", "run at 2000 ms, then 150 ms"],
                )
            ],
            [
                (
                    "blue",
                    "Embed once",
                    ["DeadlineExpired if the", "budget is gone first"],
                    "sparkle",
                    [],
                )
            ],
            [
                (
                    "green",
                    f"Spec {i}",
                    [],
                    "search",
                    [
                        f"{name}: {DF['2000']['specs'][name]['ms']} ms at 2000",
                        f"{DF['150']['specs'][name]['ms']} ms at 150"
                        + (
                            f", rerank {DF['150']['specs'][name]['rerank']}"
                            if DF["150"]["specs"][name]["rerank"] != "off"
                            else ""
                        ),
                    ],
                )
                for i, name in enumerate(DF["2000"]["specs"], 1)
            ],
            [
                (
                    "green",
                    "Result",
                    ["failed spec: status failed", "the others still return"],
                    "doc",
                    [
                        f"2000 ms budget: {DF['2000']['total_ms']} ms in all",
                        f"150 ms budget: {DF['150']['total_ms']} ms in all",
                    ],
                )
            ],
        ],
    ),
    "fact_check": (
        "A statement becomes claims, each claim gets evidence and a verdict.",
        [
            [
                (
                    "lilac",
                    "Statement",
                    ["answer, message,", "paragraph"],
                    "bubble",
                    wrap(Q, 26, 4),
                )
            ],
            [
                (
                    "yellow",
                    "Claims",
                    ["LLM, up to 5", "each with a query"],
                    "pencil",
                    [f"{len(FC['claims'])} claims, e.g.", *wrap(FC["claims"][1]["claim"], 26, 3)],
                )
            ],
            [
                (
                    "blue",
                    "Evidence",
                    ["any retrieval variant", "none: unverified"],
                    "search",
                    [
                        f"claim 1: top hit {FC['claims'][0]['evidence'][0]['doc']}",
                        f"({FC['claims'][0]['evidence'][0]['score']:.3f}), the gold doc",
                    ],
                )
            ],
            [
                (
                    "yellow",
                    "Judge",
                    ["supported, refuted,", "unverified"],
                    "trophy",
                    [f"claim {i}: {c['label']}" for i, c in enumerate(FC["claims"], 1)],
                )
            ],
            [
                (
                    "green",
                    "Aggregate",
                    ["any refuted: refuted", "else any unverified", "else supported"],
                    "layers",
                    [f"statement: {FC['label']}", f"gold label: {FC['gold_label']}"],
                )
            ],
        ],
    ),
}


def flow(subtitle: str, cols: list, seed: int, title: str) -> Sketch:
    """A row of boxes joined by arrows; a column with several boxes is a parallel step."""
    margin, gap, top = 75, 80, 205

    def box_h(box) -> float:
        lines = 1 if isinstance(box[1], str) else len(box[1])
        example = 26 + 28 * len(box[4]) if box[4] else 0
        return 50 + 36 * lines + 8 + 33 * len(box[2]) + example + 24

    n = len(cols)
    w = (2000 - 2 * margin - gap * (n - 1)) / n
    heights = [max(box_h(b) for b in col) for col in cols]
    single = max(h for col, h in zip(cols, heights, strict=True) if len(col) == 1)
    heights = [single if len(col) == 1 else h for col, h in zip(cols, heights, strict=True)]
    totals = [len(col) * h + (len(col) - 1) * 30 for col, h in zip(cols, heights, strict=True)]
    mid = top + max(totals) / 2
    s = Sketch(2000, int(top + max(totals) + 110), seed)
    s.heading(title, subtitle)
    s.text(40, 156, f"e.g. SciFact claim {EX['qid']}: {EX['query']}", 22)
    s.text(40, 186, f"gold document: {GOLD['title']}", 22)
    centres: list[list[float]] = []
    for i, (col, h, total) in enumerate(zip(cols, heights, totals, strict=True)):
        x = margin + i * (w + gap)
        ys = [mid - total / 2 + j * (h + 30) for j in range(len(col))]
        centres.append([y + h / 2 for y in ys])
        for y, (fill, name, bullets, icon, example) in zip(ys, col, strict=True):
            s.box(x, y, w, h, fill, name, bullets, icon)
            if example:
                first = y + h - 20 - 28 * (len(example) - 1)
                s._stroke(f"M {x + 22} {first - 28} L {x + w - 22} {first - 28}", 1.6, 0.5, "8 8")
                for k, line in enumerate(example):
                    s.text(x + 24, first + 28 * k, ("e.g. " if k == 0 else "       ") + line, 21)
    for i in range(n - 1):
        x1, x2 = margin + i * (w + gap) + w + 8, margin + (i + 1) * (w + gap) - 10
        for a in centres[i]:
            for b in centres[i + 1]:
                if len(centres[i]) > 1 and len(centres[i + 1]) > 1:
                    continue
                s.arrow((x1, a), (x2, b))
    s.text(1960, s.height - 36, LEGEND, 22, anchor="end")
    return s


def main() -> None:
    for name, sketch in [
        ("pipeline", pipeline()),
        ("layout", layout()),
        ("add_method", add_method()),
        ("results", results_chart()),
    ]:
        (HERE / f"{name}.svg").write_text(sketch.svg(), encoding="utf-8")
        print(f"wrote docs/diagrams/{name}.svg")
    (HERE / "methods").mkdir(exist_ok=True)
    for seed, (name, (subtitle, cols)) in enumerate(FLOWS.items(), start=100):
        sketch = flow(subtitle, cols, seed, name)
        (HERE / "methods" / f"{name}.svg").write_text(sketch.svg(), encoding="utf-8")
        print(f"wrote docs/diagrams/methods/{name}.svg")


if __name__ == "__main__":
    main()
