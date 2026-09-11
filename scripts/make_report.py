"""Render a self-contained HTML report for one bapred2 run directory.

Reads history.json (per-epoch), test.json (final test eval), recycle_sweep.json (bapred2-eval --out) and
run_config.yaml; tolerates any of them being absent so the page can be regenerated while training runs.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path

import yaml

# Reference dataviz palette (light / dark): slot-1 blue, slot-2 orange, chrome tokens.
CSS = """
:root {
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7;
  --border: rgba(11,11,11,0.10); --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s1-wash: rgba(42,120,214,0.10); --good: #006300;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink2: #c3c2b7; --muted: #898781; --grid: #2c2c2a; --axis: #383835;
  --border: rgba(255,255,255,0.10); --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s1-wash: rgba(57,135,229,0.14); --good: #0ca30c;
} }
:root[data-theme="dark"] {
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink2: #c3c2b7; --muted: #898781; --grid: #2c2c2a; --axis: #383835;
  --border: rgba(255,255,255,0.10); --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s1-wash: rgba(57,135,229,0.14); --good: #0ca30c;
}
body { background: var(--page); color: var(--ink); font-family: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif; font-size: 14.5px; line-height: 1.55; }
main { max-width: 1120px; margin: 0 auto; padding: 32px 22px 56px; display: flex; flex-direction: column; gap: 32px; }
h1 { font-size: 28px; font-weight: 600; margin: 0; letter-spacing: -0.01em; text-wrap: balance; }
h2 { font-size: 17px; font-weight: 600; margin: 0; }
p { margin: 0; max-width: 72ch; }
.mono, code { font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.93em; }
header { display: flex; flex-direction: column; gap: 10px; }
.eyebrow { font-family: "IBM Plex Mono", monospace; font-size: 11.5px; letter-spacing: 0.07em; text-transform: uppercase; color: var(--muted); }
.status { display: inline-flex; align-items: center; gap: 8px; font-size: 13px; color: var(--ink2); }
.pill { display: inline-flex; align-items: center; gap: 6px; padding: 2px 10px 2px 8px; border-radius: 999px; border: 1px solid var(--border); background: var(--surface); font-weight: 500; color: var(--ink); }
.pill i { width: 8px; height: 8px; border-radius: 50%; background: var(--s1); }
.pill.done i { background: var(--good); }
.pill.queued i { background: var(--muted); }
.progress { height: 6px; background: var(--grid); border-radius: 3px; overflow: hidden; max-width: 520px; }
.progress b { display: block; height: 100%; background: var(--s1); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1px; background: var(--border); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
.tile { background: var(--surface); padding: 12px 14px; display: flex; flex-direction: column; gap: 2px; }
.tile .k { font-size: 12px; color: var(--ink2); }
.tile .v { font-size: 24px; font-weight: 600; line-height: 1.2; }
.tile .u { font-size: 12px; color: var(--muted); }
.tile.pending .v { color: var(--muted); font-weight: 500; font-size: 18px; }
section { display: flex; flex-direction: column; gap: 12px; }
.row { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 12px 12px 8px; display: flex; flex-direction: column; gap: 4px; position: relative; }
.card h3 { margin: 0; font-size: 13px; font-weight: 500; color: var(--ink2); display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.legend { display: inline-flex; align-items: center; gap: 5px; font-size: 11.5px; color: var(--ink2); font-weight: 400; }
.legend i { display: inline-block; width: 14px; height: 2px; background: var(--s1); border-radius: 1px; margin-left: 8px; }
.legend i.k2 { background: var(--s2); }
.legend i.k3 { background: var(--s3); }
svg.chart .pt.s2 { fill: var(--s2); }
svg.chart .pt.s3 { fill: var(--s3); }
svg.chart .line.s3 { stroke: var(--s3); }
svg.chart { width: 100%; height: auto; display: block; font-family: inherit; }
svg.chart text { fill: var(--ink2); font-size: 11px; }
svg.chart .grid { stroke: var(--grid); stroke-width: 1; }
svg.chart .axis { stroke: var(--axis); stroke-width: 1; }
svg.chart .line { fill: none; stroke: var(--s1); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
svg.chart .line.s2 { stroke: var(--s2); }
svg.chart .dot { fill: var(--s1); stroke: var(--surface); stroke-width: 2; }
svg.chart .dot.s2 { fill: var(--s2); }
svg.chart .hit { fill: transparent; pointer-events: all; }
svg.chart .pt { fill: var(--s1); fill-opacity: 0.55; stroke: var(--surface); stroke-width: 1.5; }
svg.chart .pt.active { fill-opacity: 1; stroke: var(--ink); }
svg.chart .ref { stroke: var(--axis); stroke-width: 1; }
svg.chart .xhair { stroke: var(--ink2); stroke-width: 1; opacity: 0; }
svg.chart .lbl { fill: var(--ink); font-size: 11px; font-weight: 500; }
svg.chart .ring { fill: none; stroke: var(--s1); stroke-width: 2; }
.tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border); border-radius: 4px; padding: 6px 9px; font-size: 12px; line-height: 1.45; box-shadow: 0 2px 8px rgba(0,0,0,0.12); display: none; z-index: 2; white-space: nowrap; }
.tip b { font-weight: 600; font-variant-numeric: tabular-nums; }
.tip .n { color: var(--ink2); }
.filters { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; font-size: 13px; color: var(--ink2); }
.filters button { font: inherit; font-size: 13px; padding: 4px 11px; border-radius: 999px; border: 1px solid var(--border); background: var(--surface); color: var(--ink); cursor: pointer; }
.filters button[aria-pressed="true"] { border-color: var(--s1); box-shadow: inset 0 0 0 1px var(--s1); font-weight: 600; }
.filters button:focus-visible { outline: 2px solid var(--s1); outline-offset: 2px; }
.scatter-wrap { display: grid; grid-template-columns: minmax(300px, 480px) 1fr; gap: 18px; align-items: start; }
@media (max-width: 760px) { .scatter-wrap { grid-template-columns: 1fr; } }
.kv { display: grid; grid-template-columns: auto 1fr; gap: 4px 14px; font-size: 13.5px; align-content: start; }
.kv dt { color: var(--ink2); } .kv dd { margin: 0; font-variant-numeric: tabular-nums; font-weight: 500; }
details { border: 1px solid var(--border); border-radius: 6px; background: var(--surface); }
details summary { cursor: pointer; padding: 8px 12px; font-size: 13px; color: var(--ink2); }
details .wrap { overflow-x: auto; max-height: 420px; overflow-y: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: right; padding: 5px 10px; border-top: 1px solid var(--grid); font-variant-numeric: tabular-nums; white-space: nowrap; }
th { font-weight: 500; color: var(--ink2); position: sticky; top: 0; background: var(--surface); }
th:first-child, td:first-child { text-align: left; }
.pending { color: var(--muted); font-size: 13.5px; }
footer { font-size: 12.5px; color: var(--muted); border-top: 1px solid var(--border); padding-top: 12px; display: flex; flex-direction: column; gap: 4px; }
"""


def load_json(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def nice_ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min(s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw)
    start = math.floor(lo / step) * step
    ticks = []
    t = start
    while t <= hi + step * 0.5:
        ticks.append(round(t, 10))
        t += step
    return ticks


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "–"
    return f"{v:.{nd}f}"


class Line:
    """One-series line chart with hairline grid, end marker, optional highlight, and a crosshair tooltip (JS)."""

    W, H, ML, MR, MT, MB = 340, 190, 44, 14, 12, 28

    def __init__(self, key: str, xs: list[float], ys: list[float], xlabels: list[str] | None = None, ylo=None, yhi=None, highlight: int | None = None, highlight_label: str = "", series_class: str = "", ys2: list[float] | None = None, legend: tuple[str, str] | None = None):
        self.key, self.xs, self.ys = key, xs, ys
        self.xlabels = xlabels
        self.highlight, self.highlight_label = highlight, highlight_label
        self.series_class = series_class
        self.ys2, self.legend = ys2, legend
        vals = [y for y in ys + (ys2 or []) if y is not None and not math.isnan(y)]
        lo = min(vals) if ylo is None else ylo
        hi = max(vals) if yhi is None else yhi
        pad = (hi - lo) * 0.12 or 0.5
        self.ticks = nice_ticks(lo - pad if ylo is None else lo, hi + pad if yhi is None else hi)
        self.y0, self.y1 = self.ticks[0], self.ticks[-1]

    def sx(self, i):
        n = max(len(self.xs) - 1, 1)
        return self.ML + (self.W - self.ML - self.MR) * (i / n)

    def sy(self, v):
        return self.MT + (self.H - self.MT - self.MB) * (1 - (v - self.y0) / (self.y1 - self.y0))

    def svg(self) -> str:
        pts = [(self.sx(i), self.sy(y)) for i, y in enumerate(self.ys) if y is not None and not math.isnan(y)]
        parts = [f'<svg class="chart" viewBox="0 0 {self.W} {self.H}" data-chart="{self.key}" data-ml="{self.ML}" data-mr="{self.MR}" data-mt="{self.MT}" data-mb="{self.MB}" data-w="{self.W}" data-h="{self.H}" role="img" aria-label="{self.key}">']
        for t in self.ticks:
            y = self.sy(t)
            parts.append(f'<line class="grid" x1="{self.ML}" x2="{self.W - self.MR}" y1="{y:.1f}" y2="{y:.1f}"/><text x="{self.ML - 6}" y="{y + 3.5:.1f}" text-anchor="end">{t:g}</text>')
        parts.append(f'<line class="axis" x1="{self.ML}" x2="{self.W - self.MR}" y1="{self.H - self.MB}" y2="{self.H - self.MB}"/>')
        n = len(self.xs)
        label_every = max(1, math.ceil(n / 8))
        for i, x in enumerate(self.xs):
            if i % label_every == 0 or i == n - 1:
                lab = self.xlabels[i] if self.xlabels else f"{x:g}"
                parts.append(f'<text x="{self.sx(i):.1f}" y="{self.H - self.MB + 15}" text-anchor="middle">{lab}</text>')
        if self.ys2:
            pts2 = [(self.sx(i), self.sy(y)) for i, y in enumerate(self.ys2) if y is not None and not math.isnan(y)]
            if pts2:
                parts.append('<path class="line s2" d="M' + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts2) + '"/>')
                parts.append(f'<circle class="dot s2" cx="{pts2[-1][0]:.1f}" cy="{pts2[-1][1]:.1f}" r="4"/>')
        if pts:
            d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            parts.append(f'<path class="line {self.series_class}" d="{d}"/>')
            parts.append(f'<circle class="dot {self.series_class}" cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="4"/>')
        if self.highlight is not None and 0 <= self.highlight < n and self.ys[self.highlight] is not None:
            hx, hy = self.sx(self.highlight), self.sy(self.ys[self.highlight])
            parts.append(f'<circle class="ring" cx="{hx:.1f}" cy="{hy:.1f}" r="6"/>')
            anchor = "end" if hx > self.W * 0.6 else "start"
            tx = hx - 9 if anchor == "end" else hx + 9
            parts.append(f'<text class="lbl" x="{tx:.1f}" y="{hy - 8:.1f}" text-anchor="{anchor}">{self.highlight_label}</text>')
        parts.append(f'<line class="xhair" x1="0" x2="0" y1="{self.MT}" y2="{self.H - self.MB}"/>')
        parts.append(f'<rect class="hit" x="{self.ML}" y="{self.MT}" width="{self.W - self.ML - self.MR}" height="{self.H - self.MT - self.MB}"/>')
        parts.append("</svg>")
        return "".join(parts)


def scatter_svg(lo: float, hi: float, size: int = 440) -> tuple[str, dict]:
    ML, MR, MT, MB = 44, 14, 12, 34
    plot = size - ML - MR
    ploth = size - MT - MB
    ticks = nice_ticks(lo, hi, 6)

    def sx(v):
        return ML + plot * (v - lo) / (hi - lo)

    def sy(v):
        return MT + ploth * (1 - (v - lo) / (hi - lo))

    parts = [f'<svg class="chart" id="scatter" viewBox="0 0 {size} {size}" role="img" aria-label="predicted versus experimental pKd on the CASF-2016 core set">']
    for t in ticks:
        if t < lo or t > hi:
            continue
        parts.append(f'<line class="grid" x1="{ML}" x2="{size - MR}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/><text x="{ML - 6}" y="{sy(t) + 3.5:.1f}" text-anchor="end">{t:g}</text>')
        parts.append(f'<line class="grid" y1="{MT}" y2="{size - MB}" x1="{sx(t):.1f}" x2="{sx(t):.1f}"/><text x="{sx(t):.1f}" y="{size - MB + 15}" text-anchor="middle">{t:g}</text>')
    parts.append(f'<line class="axis" x1="{ML}" x2="{size - MR}" y1="{size - MB}" y2="{size - MB}"/><line class="axis" x1="{ML}" x2="{ML}" y1="{MT}" y2="{size - MB}"/>')
    parts.append(f'<line class="ref" x1="{sx(lo):.1f}" y1="{sy(lo):.1f}" x2="{sx(hi):.1f}" y2="{sy(hi):.1f}"/>')
    parts.append(f'<text x="{(ML + size - MR) / 2:.1f}" y="{size - 4}" text-anchor="middle">experimental pKd</text>')
    parts.append(f'<text transform="translate(11 {(MT + size - MB) / 2:.1f}) rotate(-90)" text-anchor="middle">predicted pKd</text>')
    parts.append('<g id="pts"></g><g id="hits"></g>')
    parts.append("</svg>")
    scale = {"ml": ML, "mt": MT, "plot": plot, "ploth": ploth, "lo": lo, "hi": hi}
    return "".join(parts), scale


def compute_curve_svg(fixed: dict, adaptive: dict) -> str:
    """Accuracy against cycles actually spent (SPEC 18): the fixed-T curve, with each adaptive
    stopping policy placed at the mean number of cycles it used."""
    W, H, ML, MR, MT, MB = 470, 260, 46, 20, 14, 34
    pts_fixed = [(float(t), fixed[t]["rmse"], f"fixed T={t}") for t in sorted(fixed, key=int)]
    rules = [("pred_delta", "s2"), ("state_delta", "s3")]
    pts_adaptive = []
    for rule, cls in rules:
        for eps, r in adaptive.get(rule, {}).items():
            pts_adaptive.append((r["mean_cycles"], r["rmse"], f"{rule} eps={eps}", cls, r["median_cycles"]))
    xs = [p[0] for p in pts_fixed] + [p[0] for p in pts_adaptive]
    ys = [p[1] for p in pts_fixed] + [p[1] for p in pts_adaptive]
    x0, x1 = 0.0, max(xs) * 1.04
    yt = nice_ticks(min(ys), max(ys), 4)
    y0, y1 = yt[0], yt[-1]

    def sx(v):
        return ML + (W - ML - MR) * (v - x0) / max(x1 - x0, 1e-9)

    def sy(v):
        return MT + (H - MT - MB) * (1 - (v - y0) / max(y1 - y0, 1e-9))

    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="RMSE against the number of recurrent cycles spent">']
    for t in yt:
        out.append(f'<line class="grid" x1="{ML}" x2="{W - MR}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/><text x="{ML - 6}" y="{sy(t) + 3.5:.1f}" text-anchor="end">{t:g}</text>')
    out.append(f'<line class="axis" x1="{ML}" x2="{W - MR}" y1="{H - MB}" y2="{H - MB}"/>')
    for v in range(0, int(x1) + 1, 4):
        out.append(f'<text x="{sx(v):.1f}" y="{H - MB + 15}" text-anchor="middle">{v}</text>')
    out.append(f'<text x="{(ML + W - MR) / 2:.1f}" y="{H - 3}" text-anchor="middle">cycles spent (mean)</text>')
    out.append('<path class="line" d="M' + " L".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y, _ in pts_fixed) + '"/>')
    for x, y, label in pts_fixed:
        out.append(f'<circle class="hit" cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="10" data-tip="{label} | RMSE {y:.3f}"/>')
    for x, y, label, cls, med in pts_adaptive:
        out.append(f'<circle class="pt {cls}" cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4.5"/>')
        out.append(f'<circle class="hit" cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="11" data-tip="{label} | RMSE {y:.3f} | mean {x:.2f} cycles, median {med:g}"/>')
    best_t = min(fixed, key=lambda k: fixed[k]["rmse"])
    bx, by = sx(float(best_t)), sy(fixed[best_t]["rmse"])
    out.append(f'<circle class="ring" cx="{bx:.1f}" cy="{by:.1f}" r="7"/>')
    anchor = "end" if bx > W * 0.62 else "start"
    out.append(f'<text class="lbl" x="{bx + (-10 if anchor == "end" else 10):.1f}" y="{by + 20:.1f}" text-anchor="{anchor}">best fixed T={best_t} ({fixed[best_t]["rmse"]:.3f})</text>')
    out.append("</svg>")
    return "".join(out)


def card(title: str, chart: Line) -> str:
    leg = ""
    if chart.legend:
        leg = f'<span class="legend"><i class="k1"></i>{chart.legend[0]}<i class="k2"></i>{chart.legend[1]}</span>'
    return f'<div class="card"><h3>{title}{leg}</h3>{chart.svg()}<div class="tip"></div></div>'


def table(headers: list[str], rows: list[list[str]], cls: str = "") -> str:
    h = "".join(f"<th>{x}</th>" for x in headers)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="wrap"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def parse_progress(log: Path) -> str | None:
    if not log.is_file():
        return None
    tail = log.read_bytes()[-4000:].decode("utf-8", errors="ignore").replace("\r", "\n")
    m = None
    for line in tail.splitlines():
        mm = re.search(r"epoch (\d+):\s+(\d+)%\|.*?\| (\d+)/(\d+)", line)
        if mm:
            m = mm
    if not m:
        return None
    return f"epoch {int(m.group(1))} · step {m.group(3)}/{m.group(4)}"


def build(run_dir: Path, title: str) -> str:
    history = load_json(run_dir / "history.json")
    test = load_json(run_dir / "test.json")
    sweep = load_json(run_dir / "recycle_sweep.json")
    run_cfg = yaml.safe_load((run_dir / "run_config.yaml").read_text()) if (run_dir / "run_config.yaml").is_file() else {}
    cfg = run_cfg.get("config", {})
    tcfg, mcfg = cfg.get("train", {}), cfg.get("model", {})
    epochs = history["epochs"] if history else []
    n_params = history["n_params"] if history else None
    total_epochs = int(run_cfg.get("args", {}).get("epochs") or tcfg.get("epochs") or 0)
    eval_T = int(mcfg.get("eval_recycles", 6))
    now = datetime.now(UTC).astimezone().strftime("%Y-%m-%d %H:%M")

    finished = sweep is not None
    status_cls, status_txt = ("done", "finished") if finished else ("queued", "queued") if not epochs else ("", "running")
    best_i = min(range(len(epochs)), key=lambda i: epochs[i]["val"]["rmse"]) if epochs else None
    best = epochs[best_i] if epochs else None
    done_n = len(epochs)
    mean_epoch = sum(e["epoch_sec"] for e in epochs) / done_n if epochs else None
    eta_min = None if finished or not epochs else (total_epochs - done_n) * mean_epoch * 1.08 / 60
    patience = int(tcfg.get("early_stop_patience", 0) or 0)
    since_best = done_n - (best_i + 1) if best_i is not None else 0

    # ---- header
    prog = parse_progress(run_dir / "train.log") if not finished else None
    parts = [f'<title>{title}</title>',
             '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">',
             f"<style>{CSS}</style>", "<main>", "<header>",
             f'<div class="eyebrow">bapred2 · {run_dir.name} · {tcfg.get("loss", "?")} loss · bf16 · rendered {now}</div>',
             f"<h1>{title}</h1>"]
    status_bits = [f'<span class="pill {status_cls}"><i></i>{status_txt}</span>']
    if epochs:
        status_bits.append(f"epoch {done_n}/{total_epochs}")
        if prog:
            status_bits.append(prog)
        if eta_min is not None:
            status_bits.append(f"ETA ≈ {eta_min / 60:.1f} h (100 epoch 기준, early stop 시 단축)")
        if patience:
            status_bits.append(f"early-stop {since_best}/{patience}")
    parts.append('<div class="status">' + " · ".join(status_bits) + "</div>")
    if epochs and total_epochs:
        parts.append(f'<div class="progress"><b style="width:{100 * done_n / total_epochs:.1f}%"></b></div>')
    parts.append("</header>")

    # ---- tiles
    tiles = []
    if best:
        tiles.append(("best val RMSE", fmt(best["val"]["rmse"]), f"epoch {best['epoch']} · Pearson {fmt(best['val']['pearson'])} · val n=999"))
    src = sweep["sweep"].get(str(eval_T)) if sweep else test
    if not src and best and "test" in best:
        tiles.append(("CASF core @ best epoch", fmt(best["test"]["rmse"]), f"RMSE · Pearson {fmt(best['test']['pearson'])} · Spearman {fmt(best['test']['spearman'])} · n=285 · T={eval_T}"))
    if src:
        tiles += [("test RMSE", fmt(src["rmse"]), f"CASF-2016 core · n=285 · T={eval_T}"),
                  ("test MAE", fmt(src["mae"]), "pKd"),
                  ("test Pearson r", fmt(src["pearson"]), ""),
                  ("test Spearman ρ", fmt(src["spearman"]), "")]
    elif not (best and "test" in best):
        tiles.append(("test (CASF core)", "pending", "best.pt로 학습 종료 후 평가", "pending"))
    if n_params:
        tiles.append(("parameters", f"{n_params / 1e6:.2f} M", f"hidden {mcfg.get('hidden_dim')} · prelude {mcfg.get('prelude_layers')}"))
    if epochs:
        tiles.append(("epoch time", f"{mean_epoch:.0f} s", f"peak GPU {max(e['peak_gpu_mem_gb'] or 0 for e in epochs):.1f} GB · batch {tcfg.get('batch_size')}"))
    parts.append('<div class="tiles">' + "".join(
        f'<div class="tile {t[3] if len(t) > 3 else ""}"><span class="k">{t[0]}</span><span class="v">{t[1]}</span><span class="u">{t[2]}</span></div>' for t in tiles) + "</div>")

    data = {"epochs": epochs, "eval_T": eval_T}

    # ---- training curves
    parts.append("<section><h2>학습 곡선</h2>")
    if epochs:
        xs = [e["epoch"] for e in epochs]
        hl = f"best {fmt(best['val']['rmse'])} @ {best['epoch']}"
        has_test = any("test" in e for e in epochs)
        t_rmse = [e.get("test", {}).get("rmse", float("nan")) for e in epochs] if has_test else None
        t_pear = [e.get("test", {}).get("pearson", float("nan")) for e in epochs] if has_test else None
        leg = ("val (999)", "CASF core (285)") if has_test else None
        charts = [
            ("train Huber loss", Line("loss", xs, [e["train_loss"] for e in epochs])),
            (f"RMSE per epoch (T={eval_T})", Line("rmse", xs, [e["val"]["rmse"] for e in epochs], highlight=best_i, highlight_label=hl, ys2=t_rmse, legend=leg)),
            ("Pearson r per epoch", Line("pearson", xs, [e["val"]["pearson"] for e in epochs], highlight=best_i, highlight_label="", ys2=t_pear, legend=leg)),
        ]
        parts.append('<div class="row">' + "".join(card(h, c) for h, c in charts) + "</div>")
        if has_test:
            parts.append("<p>파란 선이 모델 선택 기준인 val이고, 주황 선은 같은 epoch의 체크포인트를 CASF core에 그대로 찍은 값입니다. core 점수는 선택에 쓰지 않습니다.</p>")
        rows = [[str(e["epoch"]), fmt(e["train_loss"], 4), fmt(e["val"]["rmse"]), fmt(e["val"]["pearson"]), fmt(e.get("test", {}).get("rmse")), fmt(e.get("test", {}).get("pearson")), fmt(e.get("test", {}).get("spearman")), f"{e['lr']:.2e}", fmt(e["mean_train_recycles"], 2), f"{e['epoch_sec']:.0f}", fmt(e["peak_gpu_mem_gb"], 1)] for e in epochs]
        parts.append(f"<details><summary>epoch table ({done_n} rows)</summary>" + table(["epoch", "train loss", "val RMSE", "val r", "core RMSE", "core r", "core ρ", "lr", "mean T", "sec", "GPU GB"], rows) + "</details>")
    else:
        parts.append('<p class="pending">첫 epoch이 끝나면 곡선이 나타납니다.</p>')
    parts.append("</section>")

    # ---- recycle sweep
    parts.append("<section><h2>Test-time recycle sweep (best.pt, CASF-2016 core)</h2>")
    if sweep:
        Ts = sorted(sweep["sweep"], key=int)
        rm = [sweep["sweep"][t]["rmse"] for t in Ts]
        pr = [sweep["sweep"][t]["pearson"] for t in Ts]
        ms = [sweep["sweep"][t].get("wall_ms", float("nan")) / 1000 for t in Ts]
        hi_i = Ts.index(str(eval_T)) if str(eval_T) in Ts else None
        charts = [
            ("RMSE vs T", Line("sw_rmse", list(range(len(Ts))), rm, xlabels=Ts, highlight=hi_i, highlight_label=f"train eval T={eval_T}")),
            ("Pearson r vs T", Line("sw_pearson", list(range(len(Ts))), pr, xlabels=Ts, highlight=hi_i, highlight_label="")),
            ("inference wall time, s (285 graphs)", Line("sw_ms", list(range(len(Ts))), ms, xlabels=Ts, ylo=0, series_class="s2")),
        ]
        parts.append('<div class="row">' + "".join(card(h, c) for h, c in charts) + "</div>")
        rows = [[t, fmt(sweep["sweep"][t]["rmse"]), fmt(sweep["sweep"][t]["mae"]), fmt(sweep["sweep"][t]["pearson"]), fmt(sweep["sweep"][t]["spearman"]), f"{sweep['sweep'][t].get('wall_ms', float('nan')):.0f}", fmt(sweep["sweep"][t]["cycle_delta"][-1], 4)] for t in Ts]
        parts.append("<details open><summary>sweep table</summary>" + table(["T", "RMSE", "MAE", "Pearson", "Spearman", "wall ms", "last Δ state"], rows) + "</details>")
        parts.append(f'<p>학습은 T ∈ {mcfg.get("train_recycles")} 에서 샘플링했고 평가 기본값은 T={eval_T}입니다. T를 학습 범위 밖(12, 16)으로 늘렸을 때 성능이 유지되는지가 recurrent 설계의 핵심 검증입니다.</p>')
        data["sweep"] = {t: {"rmse": sweep["sweep"][t]["rmse"], "pearson": sweep["sweep"][t]["pearson"], "spearman": sweep["sweep"][t]["spearman"], "mae": sweep["sweep"][t]["mae"], "cycle_delta": sweep["sweep"][t]["cycle_delta"], "wall_ms": sweep["sweep"][t].get("wall_ms")} for t in Ts}
    else:
        parts.append('<p class="pending">학습이 끝나면 best.pt로 T = 1, 2, 3, 4, 6, 8, 12, 16 sweep을 돌립니다.</p>')
    parts.append("</section>")

    # ---- adaptive early exit
    adaptive = load_json(run_dir / "adaptive.json")
    if adaptive:
        fx, ad = adaptive["fixed"], adaptive["adaptive"]
        bf = adaptive["best_fixed_T"]
        parts.append("<section><h2>컴플렉스별 조기 종료 (per-complex early exit)</h2>")
        parts.append("<p>한 번의 forward에서 cycle마다 readout과 상태 변화량을 기록한 뒤, 각 컴플렉스를 자기 수렴 시점에서 멈춥니다. "
                     "모델이 결정적이므로 cycle t에서 멈춘 값은 T=t로 돌린 값과 정확히 같습니다. "
                     "가로축은 실제로 쓴 cycle 수라서 정확도와 연산량을 한 그림에서 비교할 수 있습니다. "
                     "세로축은 좁게 잡혀 있습니다. 실제 차이의 크기는 오른쪽 fixed-T 전체 폭으로 확인하십시오.</p>")
        legend = '<span class="legend"><i></i>fixed T<i class="k2"></i>stop on |Δpred|<i class="k3"></i>stop on Δstate</span>'
        spread = max(v["rmse"] for v in fx.values()) - min(v["rmse"] for v in fx.values())
        best_ad = min(((rule, eps, r) for rule, e in ad.items() for eps, r in e.items()), key=lambda x: x[2]["rmse"])
        cheapest = min(((rule, eps, r) for rule, e in ad.items() for eps, r in e.items() if r["rmse"] <= fx[bf]["rmse"] + 0.005), key=lambda x: x[2]["mean_cycles"], default=None)
        summary = [f'<dt>best fixed</dt><dd>T={bf} · RMSE {fx[bf]["rmse"]:.3f}</dd>',
                   f'<dt>best adaptive</dt><dd>{best_ad[0]} ε={best_ad[1]} · RMSE {best_ad[2]["rmse"]:.3f} · {best_ad[2]["mean_cycles"]:.2f} cycles</dd>']
        if cheapest:
            summary.append(f'<dt>같은 정확도 최소 연산</dt><dd>{cheapest[0]} ε={cheapest[1]} · {cheapest[2]["mean_cycles"]:.2f} cycles</dd>')
        summary += [f'<dt>fixed-T 전체 폭</dt><dd>RMSE {spread:.3f} (T=1…{adaptive["max_recycles"]})</dd>',
                    f'<dt>oracle 상한</dt><dd>RMSE {adaptive["oracle_per_complex_T"]["rmse"]:.3f} (라벨 사용)</dd>']
        parts.append('<div class="scatter-wrap"><div class="card">'
                     f'<h3>RMSE vs 쓴 cycle 수{legend}</h3>{compute_curve_svg(fx, ad)}<div class="tip"></div></div>'
                     f'<dl class="kv">{"".join(summary)}</dl></div>')
        rows = []
        for rule, entries in ad.items():
            for eps, r in entries.items():
                rows.append([f"{rule} ε={eps}", fmt(r["rmse"]), fmt(r["mae"]), fmt(r["pearson"]), fmt(r["spearman"]),
                             fmt(r["mean_cycles"], 2), fmt(r["median_cycles"], 0), f"{r['max_cycles_hit_frac']:.0%}"])
        rows.append([f"best fixed T={bf}", fmt(fx[bf]["rmse"]), fmt(fx[bf]["mae"]), fmt(fx[bf]["pearson"]), fmt(fx[bf]["spearman"]), bf, bf, "–"])
        o = adaptive["oracle_per_complex_T"]
        rows.append(["oracle (라벨 사용, 도달 불가)", fmt(o["rmse"]), fmt(o["mae"]), fmt(o["pearson"]), fmt(o["spearman"]), fmt(o["mean_cycles"], 2), fmt(o["median_cycles"], 0), "–"])
        parts.append("<details open><summary>stopping policy table</summary>" + table(["policy", "RMSE", "MAE", "Pearson", "Spearman", "mean cycles", "median", "hit max"], rows) + "</details>")
        m = adaptive["prediction_movement"]
        parts.append(f'<p>예측은 cycle에 따라 실제로 움직입니다. 첫 cycle에서 마지막 cycle까지 평균 {m["mean_abs_first_to_last"]:.3f} pKd, '
                     f'중앙값 {m["median_abs_first_to_last"]:.3f} pKd입니다. 다만 이동량과 첫 cycle 오차의 상관은 '
                     f'{m["corr_movement_vs_error_at_T1"]:+.3f}로 사실상 0이라, 많이 움직이는 컴플렉스가 어려운 컴플렉스는 아닙니다. '
                     f'즉 지금 상태 변화량은 "얼마나 더 계산해야 하는지"를 알려주는 신호가 아닙니다.</p>')
        parts.append("</section>")

    # ---- scatter
    parts.append("<section><h2>예측 vs 실험 pKd (CASF-2016 core, 285 complexes)</h2>")
    pred_sets = {}
    if sweep:
        for t, r in sweep["sweep"].items():
            if r.get("predictions"):
                pred_sets[t] = r["predictions"]
    if not pred_sets and test and test.get("predictions"):
        pred_sets[str(eval_T)] = test["predictions"]
    if pred_sets:
        allv = [v for ps in pred_sets.values() for p in ps for v in (p["y"], p["pred"])]
        lo, hi = math.floor(min(allv) - 0.5), math.ceil(max(allv) + 0.5)
        svg, scale = scatter_svg(lo, hi)
        data["scale"] = scale
        data["preds"] = pred_sets
        keys = sorted(pred_sets, key=int)
        default = str(eval_T) if str(eval_T) in pred_sets else keys[0]
        data["default_T"] = default
        btns = "".join(f'<button type="button" data-t="{t}" aria-pressed="{"true" if t == default else "false"}">T = {t}</button>' for t in keys)
        parts.append(f'<div class="filters"><span>recycles</span>{btns}</div>')
        parts.append('<div class="scatter-wrap"><div class="card">' + svg + '<div class="tip"></div></div>'
                     '<dl class="kv"><dt>RMSE</dt><dd id="k-rmse"></dd><dt>MAE</dt><dd id="k-mae"></dd><dt>Pearson r</dt><dd id="k-pearson"></dd><dt>Spearman ρ</dt><dd id="k-spearman"></dd><dt>n</dt><dd id="k-n"></dd><dt>회색 선</dt><dd>y = x</dd><dt>hover</dt><dd>PDB code · 실험값 · 예측 · 잔차</dd></dl></div>')
        parts.append('<details><summary>prediction table (selected T)</summary><div class="wrap"><table id="predtable"><thead><tr><th>id</th><th>experimental</th><th>predicted</th><th>residual</th></tr></thead><tbody></tbody></table></div></details>')
        if not sweep and test:
            data["test_stats"] = {k: test[k] for k in ("rmse", "mae", "pearson", "spearman")}
    else:
        parts.append('<p class="pending">test 예측은 학습 종료 후 best.pt로 계산됩니다.</p>')
    parts.append("</section>")

    # ---- footer
    parts.append("<footer>")
    parts.append(f"<span>config: lr {tcfg.get('lr')} · weight decay {tcfg.get('weight_decay')} · batch {tcfg.get('batch_size')} · epochs {total_epochs} · grad clip {tcfg.get('grad_clip')} · train recycles {mcfg.get('train_recycles')} p={mcfg.get('train_recycle_probs')} · early-stop patience {patience}</span>")
    parts.append("<span>data: PDBbind v2020 general-set, train 18154 · val 999 (random, seed 42) · test 285 (CASF-2016 core) · pocket 8 Å · contact 5 Å · 5 complexes skipped at preprocessing</span>")
    parts.append(f"<span>manifest: <code>{run_cfg.get('manifest', '')}</code> · run: <code>{run_dir}</code></span>")
    parts.append("</footer></main>")

    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    parts.append(f'<script type="application/json" id="data">{payload}</script>')
    parts.append("<script>" + JS + "</script>")
    return "\n".join(parts)


JS = r"""
(function () {
  const D = JSON.parse(document.getElementById('data').textContent);
  const f = (v, n) => (v == null || Number.isNaN(v)) ? '–' : Number(v).toFixed(n == null ? 3 : n);
  const core = (e, k, name) => e.test ? [[name, f(e.test[k])]] : [];
  const seriesFor = {
    loss: e => [['train loss', f(e.train_loss, 4)], ['val RMSE', f(e.val.rmse)], ...core(e, 'rmse', 'core RMSE'), ['mean T', f(e.mean_train_recycles, 2)]],
    rmse: e => [['val RMSE', f(e.val.rmse)], ...core(e, 'rmse', 'core RMSE'), ['val MAE', f(e.val.mae)], ...core(e, 'mae', 'core MAE'), ['train loss', f(e.train_loss, 4)]],
    pearson: e => [['val Pearson', f(e.val.pearson)], ...core(e, 'pearson', 'core Pearson'), ['val Spearman', f(e.val.spearman)], ...core(e, 'spearman', 'core Spearman')],
  };
  document.querySelectorAll('svg.chart[data-chart]').forEach(svg => {
    const card = svg.closest('.card'); const tip = card.querySelector('.tip'); const xhair = svg.querySelector('.xhair');
    const ml = +svg.dataset.ml, mr = +svg.dataset.mr, W = +svg.dataset.w;
    const key = svg.dataset.chart;
    const isSweep = key.startsWith('sw_');
    const xs = isSweep ? Object.keys(D.sweep || {}).sort((a, b) => a - b) : D.epochs.map(e => e.epoch);
    const n = xs.length; if (!n) return;
    const hit = svg.querySelector('.hit');
    const show = ev => {
      const r = svg.getBoundingClientRect(); const px = (ev.clientX - r.left) * W / r.width;
      const i = Math.max(0, Math.min(n - 1, Math.round((px - ml) / ((W - ml - mr) / Math.max(n - 1, 1)))));
      const sx = ml + (W - ml - mr) * (n > 1 ? i / (n - 1) : 0);
      xhair.setAttribute('x1', sx); xhair.setAttribute('x2', sx); xhair.style.opacity = 1;
      let rows;
      if (isSweep) { const s = D.sweep[xs[i]]; rows = [['RMSE', f(s.rmse)], ['MAE', f(s.mae)], ['Pearson', f(s.pearson)], ['Spearman', f(s.spearman)], ['wall', f(s.wall_ms / 1000, 2) + ' s'], ['last Δ', f(s.cycle_delta[s.cycle_delta.length - 1], 4)]]; }
      else rows = seriesFor[key](D.epochs[i]);
      tip.replaceChildren();
      const h = document.createElement('div'); h.className = 'n'; h.textContent = isSweep ? 'T = ' + xs[i] : 'epoch ' + xs[i]; tip.appendChild(h);
      rows.forEach(([k, v]) => { const d = document.createElement('div'); const b = document.createElement('b'); b.textContent = v; d.appendChild(b); d.appendChild(document.createTextNode(' ' + k)); tip.appendChild(d); });
      tip.style.display = 'block';
      const cr = card.getBoundingClientRect(); let lx = ev.clientX - cr.left + 12; if (lx + 150 > cr.width) lx = ev.clientX - cr.left - 150;
      tip.style.left = lx + 'px'; tip.style.top = (ev.clientY - cr.top + 10) + 'px';
    };
    hit.addEventListener('pointermove', show);
    hit.addEventListener('pointerleave', () => { tip.style.display = 'none'; xhair.style.opacity = 0; });
  });

  document.querySelectorAll('svg.chart [data-tip]').forEach(el => {
    const card = el.closest('.card'); const tip = card.querySelector('.tip');
    const show = ev => {
      tip.replaceChildren();
      const parts = el.dataset.tip.split('|');
      parts.forEach((t, i) => { const d = document.createElement('div'); if (i === 0) d.className = 'n'; d.textContent = t.trim(); tip.appendChild(d); });
      tip.style.display = 'block';
      const cr = card.getBoundingClientRect();
      let lx = ev.clientX - cr.left + 12; if (lx + 190 > cr.width) lx = ev.clientX - cr.left - 190;
      tip.style.left = lx + 'px'; tip.style.top = (ev.clientY - cr.top + 10) + 'px';
    };
    el.addEventListener('pointerenter', show);
    el.addEventListener('pointerleave', () => { tip.style.display = 'none'; });
  });

  if (D.preds) {
    const svg = document.getElementById('scatter'); const card = svg.closest('.card'); const tip = card.querySelector('.tip');
    const S = D.scale; const sx = v => S.ml + S.plot * (v - S.lo) / (S.hi - S.lo); const sy = v => S.mt + S.ploth * (1 - (v - S.lo) / (S.hi - S.lo));
    const pts = svg.querySelector('#pts'), hits = svg.querySelector('#hits'); const NS = 'http://www.w3.org/2000/svg';
    const tbody = document.querySelector('#predtable tbody');
    const stats = arr => {
      const n = arr.length; const ys = arr.map(p => p.y), ps = arr.map(p => p.pred);
      const rmse = Math.sqrt(arr.reduce((a, p) => a + (p.y - p.pred) ** 2, 0) / n); const mae = arr.reduce((a, p) => a + Math.abs(p.y - p.pred), 0) / n;
      const mean = a => a.reduce((x, y) => x + y, 0) / a.length; const my = mean(ys), mp = mean(ps);
      const cov = ys.reduce((a, y, i) => a + (y - my) * (ps[i] - mp), 0); const vy = ys.reduce((a, y) => a + (y - my) ** 2, 0), vp = ps.reduce((a, p) => a + (p - mp) ** 2, 0);
      const rank = a => { const idx = a.map((v, i) => [v, i]).sort((u, w) => u[0] - w[0]); const r = new Array(a.length); idx.forEach((e, k) => r[e[1]] = k); return r; };
      const ry = rank(ys), rp = rank(ps); const mry = mean(ry), mrp = mean(rp);
      const sc = ry.reduce((a, r, i) => a + (r - mry) * (rp[i] - mrp), 0) / Math.sqrt(ry.reduce((a, r) => a + (r - mry) ** 2, 0) * rp.reduce((a, r) => a + (r - mrp) ** 2, 0));
      return { rmse, mae, pearson: cov / Math.sqrt(vy * vp), spearman: sc, n };
    };
    const render = t => {
      const arr = D.preds[t]; pts.replaceChildren(); hits.replaceChildren(); tbody.replaceChildren();
      arr.forEach(p => {
        const c = document.createElementNS(NS, 'circle'); c.setAttribute('class', 'pt'); c.setAttribute('cx', sx(p.y).toFixed(1)); c.setAttribute('cy', sy(p.pred).toFixed(1)); c.setAttribute('r', 4); pts.appendChild(c);
        const h = document.createElementNS(NS, 'circle'); h.setAttribute('class', 'hit'); h.setAttribute('cx', c.getAttribute('cx')); h.setAttribute('cy', c.getAttribute('cy')); h.setAttribute('r', 12); h.setAttribute('tabindex', '0');
        const show = ev => {
          c.classList.add('active'); tip.replaceChildren();
          const rows = [[f(p.pred, 2), 'predicted'], [f(p.y, 2), 'experimental'], [(p.pred - p.y >= 0 ? '+' : '') + f(p.pred - p.y, 2), 'residual']];
          const hd = document.createElement('div'); hd.className = 'n'; hd.textContent = p.id; tip.appendChild(hd);
          rows.forEach(([v, k]) => { const d = document.createElement('div'); const b = document.createElement('b'); b.textContent = v; d.appendChild(b); d.appendChild(document.createTextNode(' ' + k)); tip.appendChild(d); });
          tip.style.display = 'block'; const cr = card.getBoundingClientRect(); const r = svg.getBoundingClientRect();
          const px = r.left + (+c.getAttribute('cx')) * r.width / 440 - cr.left, py = r.top + (+c.getAttribute('cy')) * r.height / 440 - cr.top;
          tip.style.left = (px + 14 + 130 > cr.width ? px - 140 : px + 14) + 'px'; tip.style.top = (py - 10) + 'px';
        };
        h.addEventListener('pointerenter', show); h.addEventListener('focus', show);
        const hide = () => { c.classList.remove('active'); tip.style.display = 'none'; };
        h.addEventListener('pointerleave', hide); h.addEventListener('blur', hide);
        hits.appendChild(h);
        const tr = document.createElement('tr'); [p.id, f(p.y, 2), f(p.pred, 2), f(p.pred - p.y, 2)].forEach(v => { const td = document.createElement('td'); td.textContent = v; tr.appendChild(td); }); tbody.appendChild(tr);
      });
      const s = (D.sweep && D.sweep[t]) || D.test_stats || stats(arr);
      document.getElementById('k-rmse').textContent = f(s.rmse); document.getElementById('k-mae').textContent = f(s.mae);
      document.getElementById('k-pearson').textContent = f(s.pearson); document.getElementById('k-spearman').textContent = f(s.spearman);
      document.getElementById('k-n').textContent = arr.length + ' · T = ' + t;
    };
    document.querySelectorAll('.filters button').forEach(b => b.addEventListener('click', () => {
      document.querySelectorAll('.filters button').forEach(x => x.setAttribute('aria-pressed', x === b ? 'true' : 'false')); render(b.dataset.t);
    }));
    render(D.default_T);
  }
})();
"""


def main():
    ap = argparse.ArgumentParser(description="Render an HTML report for a bapred2 run directory")
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    html = build(args.run, args.title or f"BA-Pred2 {args.run.name}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"wrote {args.out} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
