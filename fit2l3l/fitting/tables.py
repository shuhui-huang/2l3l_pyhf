"""Yield and systematic-uncertainty tables (HistFitter YieldsTable / SysTable).

Outputs plain text, LaTeX and JSON.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pyhf

from fit2l3l.fitting.common import parameter_group
from fit2l3l.utils import write_json

log = logging.getLogger(__name__)


def _fmt(value: float, unc: Optional[float] = None, digits: int = 2) -> str:
    if unc is None:
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} ± {unc:.{digits}f}"


def _tex(value: float, unc: Optional[float] = None, digits: int = 2) -> str:
    if unc is None:
        return f"${value:.{digits}f}$"
    return f"${value:.{digits}f} \\pm {unc:.{digits}f}$"


def _escape(text: str) -> str:
    return text.replace("_", r"\_").replace("#", r"\#").replace("%", r"\%")


# ---------------------------------------------------------------------------
# yields
# ---------------------------------------------------------------------------


def yields_table(
    prefit,
    postfit,
    data: Dict[str, np.ndarray],
    channels: Sequence[str],
    *,
    blinded: Sequence[str] = (),
    exclude_samples: Sequence[str] = (),
) -> Dict[str, Any]:
    """Collects per-channel observed / fitted / pre-fit yields (per sample and total)."""
    model = prefit.model
    samples = [s for s in model.config.samples if s not in exclude_samples]
    out: Dict[str, Any] = {"channels": list(channels), "samples": samples, "rows": {}}
    for channel in channels:
        ich = model.config.channels.index(channel)
        row: Dict[str, Any] = {
            "observed": None if channel in blinded else float(np.sum(data[channel])),
        }
        for label, pred in (("fitted", postfit), ("prefit", prefit)):
            if pred is None:
                continue
            yields = np.asarray(pred.model_yields[ich]).sum(axis=1)
            stdev = np.asarray(pred.total_stdev_model_channels[ich])
            per_sample = {}
            for i, sample in enumerate(model.config.samples):
                if sample in exclude_samples:
                    continue
                per_sample[sample] = (float(yields[i]), float(stdev[i]))
            idx = [i for i, s in enumerate(model.config.samples) if s not in exclude_samples]
            total = float(np.sum(yields[idx]))
            row[label] = {"total": (total, float(stdev[-1])), "samples": per_sample}
        out["rows"][channel] = row
    return out


def write_yields_table(table: Dict[str, Any], output: pathlib.Path, *, samples_meta: Dict[str, Any] = None,
                       title: str = "") -> List[pathlib.Path]:
    from tabulate import tabulate

    samples_meta = samples_meta or {}
    channels = table["channels"]
    rows = table["rows"]
    have_fit = any("fitted" in rows[c] for c in channels)
    lines = [["Observed events"] + [("blinded" if rows[c]["observed"] is None else f"{rows[c]['observed']:.0f}") for c in channels]]
    if have_fit:
        lines.append(["Fitted bkg events"] + [_fmt(*rows[c]["fitted"]["total"]) for c in channels])
        for s in table["samples"]:
            lines.append([f"  Fitted {s}"] + [_fmt(*rows[c]["fitted"]["samples"][s]) for c in channels])
    lines.append(["MC exp. SM events"] + [_fmt(*rows[c]["prefit"]["total"]) for c in channels])
    for s in table["samples"]:
        lines.append([f"  MC exp. {s}"] + [_fmt(*rows[c]["prefit"]["samples"][s]) for c in channels])
    txt = tabulate(lines, headers=["Region"] + list(channels), tablefmt="simple")
    output.parent.mkdir(parents=True, exist_ok=True)
    txt_path = output.with_suffix(".txt")
    txt_path.write_text((title + "\n" if title else "") + txt + "\n")

    # LaTeX, HistFitter-like layout
    col = "l" + "c" * len(channels)
    tex = [r"\begin{table}[htbp]", r"\centering", r"\small", rf"\begin{{tabular}}{{{col}}}", r"\hline\hline"]
    tex.append(r"{\bf Region} & " + " & ".join(_escape(c) for c in channels) + r" \\")
    tex.append(r"\hline")
    tex.append("Observed events & " + " & ".join(
        "--" if rows[c]["observed"] is None else f"${rows[c]['observed']:.0f}$" for c in channels) + r" \\")
    tex.append(r"\hline")
    if have_fit:
        tex.append("Fitted bkg events & " + " & ".join(_tex(*rows[c]["fitted"]["total"]) for c in channels) + r" \\")
        tex.append(r"\hline")
        for s in table["samples"]:
            name = samples_meta.get(s, {}).get("label", s)
            tex.append(f"Fitted {_escape(name) if '$' not in name else name} events & " + " & ".join(
                _tex(*rows[c]["fitted"]["samples"][s]) for c in channels) + r" \\")
        tex.append(r"\hline")
    tex.append("MC exp. SM events & " + " & ".join(_tex(*rows[c]["prefit"]["total"]) for c in channels) + r" \\")
    tex.append(r"\hline")
    for s in table["samples"]:
        name = samples_meta.get(s, {}).get("label", s)
        tex.append(f"MC exp. {_escape(name) if '$' not in name else name} events & " + " & ".join(
            _tex(*rows[c]["prefit"]["samples"][s]) for c in channels) + r" \\")
    tex += [r"\hline\hline", r"\end{tabular}", rf"\caption{{{_escape(title)}}}" if title else "", r"\end{table}"]
    tex_path = output.with_suffix(".tex")
    tex_path.write_text("\n".join(l for l in tex if l) + "\n")
    json_path = output.with_suffix(".json")
    write_json(table, json_path)
    return [txt_path, tex_path, json_path]


# ---------------------------------------------------------------------------
# systematic breakdown
# ---------------------------------------------------------------------------


def uncertainty_breakdown(
    model: pyhf.pdf.Model,
    parameters: np.ndarray,
    uncertainty: np.ndarray,
    corr_mat: np.ndarray,
    channels: Sequence[str],
    groups: Dict[str, str],
    *,
    exclude_samples: Sequence[str] = (),
) -> Dict[str, Any]:
    """Impact of each parameter (and group of parameters) on the total background yield.

    For every parameter i the yields are evaluated at ``theta_i +- sigma_i`` (all other
    parameters at their best-fit values); the symmetrised shift is the impact delta_i.
    The total uncertainty is sqrt(delta^T C delta) with the correlation matrix C,
    a group uncertainty is the same restricted to the parameters of the group
    (HistFitter SysTable, error propagation from the fit result).
    """
    parameters = np.asarray(parameters, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    corr = np.asarray(corr_mat, dtype=float)
    sample_mask = np.asarray([s not in exclude_samples for s in model.config.samples])
    slices = {c: model.config.channel_slices[c] for c in channels}

    def channel_totals(pars):
        by_sample = np.asarray(model.main_model.expected_data(pars, return_by_sample=True))
        totals = by_sample[sample_mask].sum(axis=0)
        return np.asarray([totals[slices[c]].sum() for c in channels]), by_sample

    nominal, nominal_by_sample = channel_totals(parameters)
    n = len(parameters)
    deltas = np.zeros((n, len(channels)))
    for i in range(n):
        if uncertainty[i] <= 0:
            continue
        up = parameters.copy()
        up[i] += uncertainty[i]
        down = parameters.copy()
        down[i] -= uncertainty[i]
        deltas[i] = 0.5 * (channel_totals(up)[0] - channel_totals(down)[0])

    names = list(model.config.par_names)

    out: Dict[str, Any] = {}
    group_names = sorted({parameter_group(nm, groups) for nm in names})
    for ic, channel in enumerate(channels):
        d = deltas[:, ic]
        total = float(np.sqrt(max(d @ corr @ d, 0.0)))
        params = {names[i]: float(d[i]) for i in range(n) if abs(d[i]) > 0}
        grp = {}
        for gname in group_names:
            idx = [i for i, nm in enumerate(names) if parameter_group(nm, groups) == gname]
            dg = d[idx]
            value = float(np.sqrt(max(dg @ corr[np.ix_(idx, idx)] @ dg, 0.0)))
            if value > 0:
                grp[gname] = value
        out[channel] = {
            "total_yield": float(nominal[ic]),
            "total_uncertainty": total,
            "parameters": dict(sorted(params.items(), key=lambda kv: -abs(kv[1]))),
            "groups": dict(sorted(grp.items(), key=lambda kv: -kv[1])),
        }
    return out


def write_systematics_table(breakdown: Dict[str, Any], output: pathlib.Path, *, title: str = "",
                            parameters: bool = True, groups: bool = True, threshold: float = 0.0) -> List[pathlib.Path]:
    from tabulate import tabulate

    channels = list(breakdown)
    written = []
    output.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    tex_blocks = []
    for kind, enabled in (("groups", groups), ("parameters", parameters)):
        if not enabled:
            continue
        keys = []
        for c in channels:
            for k in breakdown[c][kind]:
                if k not in keys:
                    keys.append(k)
        # sort by the largest impact over all channels
        keys.sort(key=lambda k: -max(abs(breakdown[c][kind].get(k, 0.0)) / max(breakdown[c]["total_yield"], 1e-12) for c in channels))
        rows = [["Total background expectation"] + [f"{breakdown[c]['total_yield']:.2f}" for c in channels]]
        rows.append(["Total uncertainty"] + [
            f"±{breakdown[c]['total_uncertainty']:.2f} [{100 * breakdown[c]['total_uncertainty'] / max(breakdown[c]['total_yield'], 1e-12):.1f}%]"
            for c in channels])
        for k in keys:
            vals = [abs(breakdown[c][kind].get(k, 0.0)) for c in channels]
            rel = [v / max(breakdown[c]["total_yield"], 1e-12) for v, c in zip(vals, channels)]
            if max(rel) < threshold:
                continue
            rows.append([k] + [f"±{v:.2f} [{100 * r:.1f}%]" for v, r in zip(vals, rel)])
        blocks.append(f"=== uncertainty breakdown ({kind}) ===\n" + tabulate(rows, headers=["Uncertainty"] + channels))
        col = "l" + "c" * len(channels)
        tex = [r"\begin{table}[htbp]", r"\centering", r"\small", rf"\begin{{tabular}}{{{col}}}", r"\hline\hline",
               r"{\bf Uncertainty} & " + " & ".join(_escape(c) for c in channels) + r" \\", r"\hline"]
        for row in rows:
            tex.append(_escape(row[0]) + " & " + " & ".join(
                cell.replace("±", r"$\pm$").replace("%", r"\%") for cell in row[1:]) + r" \\")
            if row[0] == "Total uncertainty":
                tex.append(r"\hline")
        tex += [r"\hline\hline", r"\end{tabular}", rf"\caption{{{_escape(title)} ({kind})}}", r"\end{table}"]
        tex_blocks.append("\n".join(tex))
    txt_path = output.with_suffix(".txt")
    txt_path.write_text((title + "\n" if title else "") + "\n\n".join(blocks) + "\n")
    tex_path = output.with_suffix(".tex")
    tex_path.write_text("\n\n".join(tex_blocks) + "\n")
    json_path = output.with_suffix(".json")
    write_json(breakdown, json_path)
    written += [txt_path, tex_path, json_path]
    return written


def plot_grouped_uncertainties(breakdown: Dict[str, Any], output: pathlib.Path, *, atlas="Internal",
                               lumi_fb=None, formats=("pdf", "png")):
    """Relative uncertainty per group and region (HistFitter PlotSystematicsGrouped)."""
    from fit2l3l.plotting import style
    from fit2l3l.plotting.style import plt

    channels = list(breakdown)
    groups = []
    for c in channels:
        for g in breakdown[c]["groups"]:
            if g not in groups:
                groups.append(g)
    x = np.arange(len(channels))
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(channels) + 3), 5))
    tot = [breakdown[c]["total_uncertainty"] / max(breakdown[c]["total_yield"], 1e-12) for c in channels]
    ax.step(np.append(x, x[-1] + 1) - 0.5, np.append(tot, tot[-1]), where="post", color="black", label="Total", linewidth=2)
    for i, g in enumerate(groups):
        vals = [breakdown[c]["groups"].get(g, 0.0) / max(breakdown[c]["total_yield"], 1e-12) for c in channels]
        ax.step(np.append(x, x[-1] + 1) - 0.5, np.append(vals, vals[-1]), where="post", label=g, linestyle="--" if i >= 10 else "-")
    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel(r"Relative uncertainty $\Delta N_\mathrm{bkg}/N_\mathrm{bkg}$")
    ax.set_ylim(0, max(tot) * 1.8 if tot else 1)
    style.atlas_label(ax, atlas, lumi_fb)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper right")
    return style.save(fig, output, formats)
