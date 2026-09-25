"""Self-contained HTML report generation (no JavaScript dependencies).

The report embeds an inline-SVG risk trend chart, band-colored batch table,
summary statistics, and per-batch top-driver analysis.
"""

from __future__ import annotations

import html as _html
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from ..core.explain import explain as _explain

_BAND_COLORS = {
    "LOW": "#2e7d32",
    "MODERATE": "#f9a825",
    "HIGH": "#ef6c00",
    "CRITICAL": "#c62828",
    "UNKNOWN": "#616161",
    None: "#616161",
}


def _svg_chart(risks: Sequence[float], bands: Sequence[Optional[str]],
               width: int = 760, height: int = 220) -> str:
    if not risks:
        return "<p>No history yet.</p>"
    pad = 30
    w, h = width - 2 * pad, height - 2 * pad
    n = len(risks)
    step = w / max(n - 1, 1)
    pts = [(pad + i * step, pad + h * (1 - min(max(r, 0.0), 1.0))) for i, r in enumerate(risks)]
    path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{_BAND_COLORS.get(b, "#616161")}"/>'
        for (x, y), b in zip(pts, bands)
    )
    grid = "".join(
        f'<line x1="{pad}" y1="{pad + h * (1 - t)}" x2="{pad + w}" '
        f'y2="{pad + h * (1 - t)}" stroke="#ddd" stroke-dasharray="3,3"/>'
        f'<text x="2" y="{pad + h * (1 - t) + 4}" font-size="10" fill="#999">{t}</text>'
        for t in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'style="max-width:{width}px;background:#fafafa;border:1px solid #eee">'
        f"{grid}"
        f'<path d="{path}" fill="none" stroke="#1565c0" stroke-width="2"/>{dots}</svg>'
    )


def generate_html_report(
    records: List[Dict],
    *,
    title: str = "DFRE Monitoring Report",
    top_driver_signal_names: Optional[List[str]] = None,
) -> str:
    """Render a list of history records (see HistoryStore.load) to HTML.

    Parameters
    ----------
    records : list of dict
        Each with keys ``batch_id``, ``timestamp``, ``risk``, ``band``, ``inputs``.
    title : str
    top_driver_signal_names : list of str, optional
        Which input signals to attribute (default M, V, D, U).
    """
    names = top_driver_signal_names or ["M", "V", "D", "U"]
    risks = [float(r.get("risk", 0.0)) for r in records]
    bands = [r.get("band") for r in records]

    band_counts: Dict[str, int] = {}
    for b in bands:
        key = b or "UNKNOWN"
        band_counts[key] = band_counts.get(key, 0) + 1

    mean_risk = sum(risks) / len(risks) if risks else 0.0
    max_risk = max(risks) if risks else 0.0

    # Top-driver analysis per batch (leave-one-out on the core signals).
    driver_counts: Dict[str, int] = {}
    rows = []
    for r in records:
        inputs = r.get("inputs", {})
        signals = {k: float(inputs.get(k, 0.0)) for k in names if k in inputs}
        top = None
        if signals and any(v > 0 for v in signals.values()):
            try:
                top = _explain(
                    signals,
                    n=int(inputs.get("N", 1000)),
                    lam=float(inputs.get("lam", 1.0)),
                    gamma=float(inputs.get("gamma", 1.0)),
                ).top_driver
            except Exception:
                top = None
        if top:
            driver_counts[top] = driver_counts.get(top, 0) + 1
        band = r.get("band")
        color = _BAND_COLORS.get(band, _BAND_COLORS["UNKNOWN"])
        sig_cells = "".join(
            f"<td>{float(inputs.get(k, 0.0)):.3f}</td>" for k in names
        )
        rows.append(
            f"<tr><td>{_html.escape(str(r.get('batch_id', '?')))}</td>"
            f"<td>{_html.escape(str(r.get('timestamp', '')))}</td>"
            f"<td><b>{float(r.get('risk', 0.0)):.4f}</b></td>"
            f'<td><span style="background:{color};color:#fff;padding:2px 8px;'
            f'border-radius:8px;font-size:12px">{_html.escape(str(band or "—"))}</span></td>'
            f"{sig_cells}<td>{_html.escape(str(top or '—'))}</td></tr>"
        )

    sig_headers = "".join(f"<th>{_html.escape(k)}</th>" for k in names)
    cards = "".join(
        f'<div class="card"><div class="num">{v}</div>'
        f'<div class="lbl" style="color:{_BAND_COLORS.get(k, "#616161")}">{k}</div></div>'
        for k, v in sorted(band_counts.items())
    )
    driver_summary = ", ".join(f"{k} ({v}×)" for k, v in
                               sorted(driver_counts.items(), key=lambda kv: -kv[1])) or "—"
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html.escape(title)}</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f5f6f8;color:#222}}
header{{background:#0d1b2a;color:#fff;padding:24px 32px}}
header h1{{margin:0;font-size:22px}} header p{{margin:4px 0 0;opacity:.7;font-size:13px}}
main{{padding:24px 32px;max-width:900px;margin:auto}}
.cards{{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}}
.card{{background:#fff;border-radius:10px;padding:14px 22px;box-shadow:0 1px 3px rgba(0,0,0,.08);text-align:center}}
.num{{font-size:24px;font-weight:700}} .lbl{{font-size:12px;text-transform:uppercase}}
section{{background:#fff;border-radius:10px;padding:20px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{padding:8px 10px;border-bottom:1px solid #eee;text-align:left}}
th{{background:#f0f2f5}}
</style></head><body>
<header><h1>{_html.escape(title)}</h1><p>Generated {generated} · {len(records)} batches ·
mean 𝓡 = {mean_risk:.4f} · max 𝓡 = {max_risk:.4f}</p></header>
<main>
<div class="cards">
<div class="card"><div class="num">{len(records)}</div><div class="lbl">Batches</div></div>
<div class="card"><div class="num">{mean_risk:.3f}</div><div class="lbl">Mean risk</div></div>
<div class="card"><div class="num">{max_risk:.3f}</div><div class="lbl">Max risk</div></div>
{cards}
</div>
<section><h2>Risk trend</h2>{_svg_chart(risks, bands)}</section>
<section><h2>Top risk drivers</h2><p>{_html.escape(driver_summary)}</p></section>
<section><h2>Batches</h2>
<table><thead><tr><th>Batch</th><th>Timestamp</th><th>𝓡</th><th>Band</th>{sig_headers}<th>Top driver</th></tr></thead>
<tbody>{''.join(rows) if rows else '<tr><td colspan="99">No records</td></tr>'}</tbody></table>
</section>
</main></body></html>"""
