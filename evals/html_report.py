"""The HTML report.

This is the artifact a reader actually looks at, and for an audio project the
non-negotiable part is that they can **hear** it. A results table alone asks
someone to trust that the measured improvement corresponds to something
audible, which is exactly the claim SPEC 10.5 says to treat with suspicion.

Audio is transcoded to 320 kbps MP3 for transport because a lossless page is
tens of megabytes. Every measurement in the tables is computed on the lossless
render, never on the transcode, and the page says so -- the alternative is a
reader reasonably wondering whether the numbers describe the files they just
heard.

No external scripts, fonts or stylesheets. The page is served from a static
directory and has to work offline, so the charts are inline SVG rather than a
charting library.
"""

from __future__ import annotations

import html
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from headroom.audio import AudioBuffer, save
from headroom.control.state import RunTrace

from .report import ResultsTable, aggregate, paired

MP3_BITRATE: Final[str] = "320k"

#: Systems in the order they should be presented: floors, then the real
#: competitor, then the bound.
SYSTEM_ORDER: Final[tuple[str, ...]] = (
    "null",
    "random",
    "hillclimb",
    "heuristic",
    "single_agent",
    "agent",
    "optimizer",
)

_CSS: Final[str] = """
:root {
  --bg: #fbfaf8; --panel: #ffffff; --ink: #16181d; --muted: #5c6270;
  --line: #e3e1dc; --accent: #1d4ed8; --good: #0f7b52; --bad: #b4232a;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
:root:not([data-theme="light"]) { }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #14161a; --panel: #1c1f25; --ink: #e9e8e4; --muted: #a0a6b4;
    --line: #2c313a; --accent: #7aa2ff; --good: #4ecb8f; --bad: #ff8a8a;
  }
}
:root[data-theme="dark"] {
  --bg: #14161a; --panel: #1c1f25; --ink: #e9e8e4; --muted: #a0a6b4;
  --line: #2c313a; --accent: #7aa2ff; --good: #4ecb8f; --bad: #ff8a8a;
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); margin: 0;
  font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.wrap { max-width: 1080px; margin: 0 auto; padding: 48px 24px 96px; }
h1 { font-size: 34px; letter-spacing: -0.02em; margin: 0 0 6px; }
h2 { font-size: 21px; letter-spacing: -0.01em; margin: 48px 0 12px;
  padding-bottom: 8px; border-bottom: 1px solid var(--line); }
h3 { font-size: 15px; margin: 28px 0 8px; }
.sub { color: var(--muted); margin: 0 0 4px; }
.lead { max-width: 68ch; }
code, .mono { font-family: var(--mono); font-size: 13px; }
.panel { background: var(--panel); border: 1px solid var(--line);
  border-radius: 10px; padding: 18px 20px; margin: 16px 0; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--line);
  white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-weight: 600; font-size: 12px;
  text-transform: uppercase; letter-spacing: 0.04em; }
tr.bound td { color: var(--muted); font-style: italic; }
tr.hero td { font-weight: 600; }
.pos { color: var(--good); } .neg { color: var(--bad); }
.players { display: grid; gap: 10px; }
.player { display: grid; grid-template-columns: 130px 1fr; gap: 12px;
  align-items: center; }
.player .name { font-family: var(--mono); font-size: 12px; color: var(--muted); }
audio { width: 100%; height: 34px; }
.chain { font-family: var(--mono); font-size: 12px; color: var(--muted);
  word-break: break-word; margin: 6px 0 0; }
.note { color: var(--muted); font-size: 13px; max-width: 68ch; }
.warn { border-left: 3px solid var(--bad); padding-left: 12px; }
.tag { display: inline-block; font-family: var(--mono); font-size: 11px;
  color: var(--muted); border: 1px solid var(--line); border-radius: 999px;
  padding: 1px 8px; margin-right: 6px; }
@media (max-width: 640px) {
  .player { grid-template-columns: 1fr; gap: 4px; }
}
"""


@dataclass(frozen=True, slots=True)
class Showcase:
    """One cell rendered for listening."""

    track_id: str
    degradation_kind: str
    degradation_seed: int
    degradation_describe: str
    original: AudioBuffer
    degraded: AudioBuffer
    outputs: dict[str, AudioBuffer]
    traces: dict[str, RunTrace]


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def write_audio(buf: AudioBuffer, path: Path) -> Path:
    """Write a listenable file, transcoding to MP3 when ffmpeg is available.

    Falls back to 16-bit WAV rather than failing: a heavier page still lets
    someone hear the result, which is the point.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = path.with_suffix(".wav")
    save(buf, wav, subtype="PCM_16")
    if not _have_ffmpeg():
        return wav
    mp3 = path.with_suffix(".mp3")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-b:a", MP3_BITRATE, str(mp3)],
            check=True,
            timeout=120,
        )
    except (subprocess.SubprocessError, OSError):
        return wav
    wav.unlink(missing_ok=True)
    return mp3


def _order_key(system: str) -> int:
    return SYSTEM_ORDER.index(system) if system in SYSTEM_ORDER else len(SYSTEM_ORDER)


def _recovery_html(value: float) -> str:
    css = "pos" if value > 0 else ("neg" if value < 0 else "")
    return f'<span class="{css}">{value:+.3f}</span>'


def _table(table: ResultsTable, kind: str | None = None) -> str:
    cells = (
        [c for c in table.by_kind if c.degradation_kind == kind] if kind else list(table.overall)
    )
    if not cells:
        return ""
    cells.sort(key=lambda c: _order_key(c.system))
    rows: list[str] = []
    for cell in cells:
        classes = []
        if cell.system == "optimizer":
            classes.append("bound")
        if cell.system in ("heuristic", "agent"):
            classes.append("hero")
        rows.append(
            f'<tr class="{" ".join(classes)}">'
            f"<td>{_esc(cell.system)}</td><td>{cell.n}</td>"
            f"<td>{_recovery_html(cell.recovery_median)}</td>"
            f"<td>{cell.recovery_q1:+.3f} to {cell.recovery_q3:+.3f}</td>"
            f"<td>{cell.converged_rate:.0%}</td>"
            f"<td>{cell.oscillation_rate:.0%}</td>"
            f"<td>{cell.regression_rate:.0%}</td>"
            f"<td>{cell.renders_median:.0f}</td>"
            f"<td>{cell.wall_median_s:.1f}</td>"
            f"<td>${cell.cost_total_usd:.4f}</td></tr>"
        )
    head = (
        "<tr><th>system</th><th>n</th><th>recovery</th><th>IQR</th>"
        "<th>converged</th><th>oscillated</th><th>regressed</th>"
        "<th>renders</th><th>wall s</th><th>cost</th></tr>"
    )
    return f'<div class="scroll"><table>{head}{"".join(rows)}</table></div>'


def _convergence_svg(traces: Sequence[RunTrace], width: int = 640, height: int = 220) -> str:
    """Distance against step, one line per system, shared axes.

    Inline SVG because the page must work offline from a static directory.
    """
    series = [
        (t.system, [s.distance_score for s in t.steps])
        for t in sorted(traces, key=lambda t: _order_key(t.system))
        if t.steps
    ]
    if not series:
        return ""
    max_len = max(len(v) for _, v in series)
    max_y = max(max(v) for _, v in series) or 1.0
    pad_l, pad_b, pad_t, pad_r = 46, 28, 12, 108

    palette = ["#8a8f9c", "#b4232a", "#c98a1b", "#1d4ed8", "#7b3fbf", "#0f7b52"]
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def x(i: int) -> float:
        return pad_l + (plot_w * i / max(max_len - 1, 1))

    def y(v: float) -> float:
        return pad_t + plot_h * (1.0 - min(v / max_y, 1.0))

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'role="img" aria-label="distance against step for each system">',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" '
        'stroke="currentColor" stroke-opacity="0.25"/>',
        f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" '
        f'y2="{pad_t + plot_h}" stroke="currentColor" stroke-opacity="0.25"/>',
        f'<text x="4" y="{pad_t + 10}" font-size="10" fill="currentColor" '
        f'fill-opacity="0.6">{max_y:.2f}</text>',
        f'<text x="4" y="{pad_t + plot_h}" font-size="10" fill="currentColor" '
        'fill-opacity="0.6">0</text>',
        f'<text x="{pad_l}" y="{height - 8}" font-size="10" fill="currentColor" '
        'fill-opacity="0.6">step</text>',
    ]
    for idx, (system, values) in enumerate(series):
        colour = palette[idx % len(palette)]
        points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
        parts.append(
            f'<polyline fill="none" stroke="{colour}" stroke-width="2" '
            f'stroke-linejoin="round" points="{points}"/>'
        )
        parts.append(
            f'<text x="{pad_l + plot_w + 8}" y="{pad_t + 12 + idx * 15}" font-size="11" '
            f'fill="{colour}">{_esc(system)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _showcase_html(showcase: Showcase, audio_dir: Path, rel: str) -> str:
    players = [
        ("original (target)", showcase.original, "the file the system is trying to match"),
        ("degraded (input)", showcase.degraded, showcase.degradation_describe),
    ]
    for system in sorted(showcase.outputs, key=_order_key):
        players.append((system, showcase.outputs[system], ""))

    safe = showcase.track_id.replace("/", "_")
    slug = f"{safe}_{showcase.degradation_kind}_{showcase.degradation_seed}"
    blocks: list[str] = []
    for label, buf, hint in players:
        filename = write_audio(buf, audio_dir / f"{slug}__{label.split(' ')[0]}")
        trace = showcase.traces.get(label)
        meta = ""
        if trace is not None:
            meta = (
                f'<div class="chain">recovery {trace.recovery_ratio:+.3f} '
                f"&middot; {trace.n_renders} renders &middot; "
                f"{_esc(trace.final_chain.describe())}</div>"
            )
        elif hint:
            meta = f'<div class="chain">{_esc(hint)}</div>'
        blocks.append(
            '<div class="player">'
            f'<div class="name">{_esc(label)}</div>'
            f'<div><audio controls preload="none" src="{rel}/{_esc(filename.name)}"></audio>'
            f"{meta}</div></div>"
        )

    return (
        '<div class="panel">'
        f"<h3>{_esc(showcase.track_id)} &middot; "
        f"{_esc(showcase.degradation_kind)}/{showcase.degradation_seed}</h3>"
        f'<p class="note">{_esc(showcase.degradation_describe)}</p>'
        f'<div class="players">{"".join(blocks)}</div>'
        f"{_convergence_svg(list(showcase.traces.values()))}"
        "</div>"
    )


def build(
    traces: Sequence[RunTrace],
    out_dir: str | Path,
    showcases: Sequence[Showcase] = (),
    *,
    title: str = "headroom results",
    corpus_note: str = "",
) -> Path:
    """Write the report to ``out_dir/index.html`` with audio alongside."""
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    audio_dir = destination / "audio"

    table = aggregate(traces)
    kinds = sorted({c.degradation_kind for c in table.by_kind})

    systems = sorted({t.system for t in traces} - {"heuristic"}, key=_order_key)
    paired_rows = "".join(
        (
            lambda r: (
                f"<tr><td>{_esc(r.system_a)}</td><td>{r.n}</td>"
                f"<td>{_recovery_html(r.median_difference)}</td>"
                f"<td>{r.a_wins}</td><td>{r.b_wins}</td><td>{r.ties}</td>"
                f"<td>{'—' if r.p_value is None else f'{r.p_value:.4f}'}</td></tr>"
            )
        )(paired(traces, system, "heuristic"))
        for system in systems
    )

    incomparable = (
        ""
        if table.comparable
        else (
            '<div class="panel warn"><strong>Not comparable.</strong> These traces come '
            f"from {len(table.config_hashes)} different metric configurations, so the "
            "rows below do not all measure the same thing.</div>"
        )
    )

    cost = sum(t.total_cost_usd for t in traces)
    versions = next((t.package_versions for t in traces if t.package_versions), {})
    provenance = " ".join(
        f'<span class="tag">{_esc(k)} {_esc(v)}</span>' for k, v in sorted(versions.items())
    )

    body = f"""<div class="wrap">
<h1>{_esc(title)}</h1>
<p class="sub">Mastering decisions graded by measurement, not by an LLM's opinion.</p>
<p class="lead note">Every system below runs inside the same loop, against the same
degraded input, with the same render budget, and is scored by the same
deterministic feature vector. <code>recovery</code> is
<code>1 - final/initial</code> distance: 1.0 is a perfect repair, 0 is no
progress, negative means the system made the audio worse.
<code>optimizer</code> is a measured upper bound rather than a competitor &mdash;
it gets hundreds of renders instead of {table.overall[0].renders_median:.0f} and is
seeded with the other systems' answers, so it says what was achievable, not what
is practical.</p>
{incomparable}

<h2>Overall</h2>
{_table(table)}

<h2>Paired against the heuristic</h2>
<p class="note">Same track, degradation and seed on both sides, so each row is a
per-cell difference rather than a difference of marginal medians. The test is
Wilcoxon signed-rank: recovery ratios are bounded above, unbounded below and not
normal, so a t-test would be measuring the wrong thing.</p>
<div class="scroll"><table>
<tr><th>system</th><th>n</th><th>median difference</th><th>wins</th><th>losses</th>
<th>ties</th><th>signed-rank p</th></tr>
{paired_rows}
</table></div>

<h2>By degradation</h2>
<p class="note">The useful question is not which system is better overall but which
cases each one wins. Single-feature degradations are where a proportional
controller should win; coupled ones are where coordinating moves across an
interacting system should start to earn its cost.</p>
{"".join(f"<h3>{_esc(k)}</h3>{_table(table, k)}" for k in kinds)}

<h2>Listen</h2>
<p class="note">Measured is not perceived. A system can hit every target and still
sound wrong, so the outputs are here to be heard rather than trusted. Players use
320&nbsp;kbps MP3 for transport; every number on this page is computed on the
lossless render.</p>
{
        "".join(_showcase_html(s, audio_dir, "audio") for s in showcases)
        or '<div class="panel note">No showcase cells were rendered.</div>'
    }

<h2>Provenance</h2>
<div class="panel">
<p class="note">{_esc(corpus_note) if corpus_note else "Corpus not recorded."}</p>
<p class="note">{len(traces)} traces &middot;
git <code>{_esc(sorted(table.git_shas)[0] if table.git_shas else "unknown")}</code> &middot;
metric config <code>{
        _esc(sorted(table.config_hashes)[0] if table.config_hashes else "unknown")
    }</code> &middot;
total API spend <code>${cost:.4f}</code></p>
<p>{provenance}</p>
<p class="note">Pinned versions matter: pedalboard, librosa and numpy can all shift
numeric output across releases, which would silently invalidate cached statistics
and stored results.</p>
</div>
</div>"""

    page = f"<title>{_esc(title)}</title>\n<style>{_CSS}</style>\n{body}\n"
    out = destination / "index.html"
    out.write_text(page)
    return out


def make_showcase(
    track_id: str,
    kind: str,
    seed: int,
    traces: Sequence[RunTrace],
    *,
    source: AudioBuffer,
    degradation_describe: str = "",
) -> Showcase:
    """Reproduce the audio for one cell so it can be listened to.

    Rebuilt from the traces rather than saved during the run: a trace holds the
    chain and the source hash, so the audio is a derived artifact. That keeps
    runs cheap and guarantees what is played is what the numbers describe.
    """
    from headroom.analysis.features import analyze
    from headroom.dsp.backends.pedalboard import render_chain

    from .degradations import DegradationKind, make_degradation

    cell = [
        t
        for t in traces
        if t.track_id == track_id and t.degradation_kind == kind and t.degradation_seed == seed
    ]
    if not cell:
        raise KeyError(f"no traces for {track_id} {kind}/{seed}")

    level = analyze(source).lufs_integrated
    degradation = make_degradation(
        DegradationKind(kind) if not isinstance(kind, str) else kind,  # type: ignore[arg-type]
        seed=seed,
        level_db=level,
    )
    degraded = render_chain(source, degradation.chain)

    outputs: dict[str, AudioBuffer] = {}
    by_system: dict[str, RunTrace] = {}
    for trace in sorted(cell, key=lambda t: _order_key(t.system)):
        outputs[trace.system] = render_chain(degraded, trace.final_chain)
        by_system[trace.system] = trace

    return Showcase(
        track_id=track_id,
        degradation_kind=kind,
        degradation_seed=seed,
        degradation_describe=degradation_describe or degradation.describe(),
        original=source,
        degraded=degraded,
        outputs=outputs,
        traces=by_system,
    )
