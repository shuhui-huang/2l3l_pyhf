"""Data / background plots per region and summary plots over many regions."""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from fit2l3l.plotting import style
from fit2l3l.plotting.style import plt

log = logging.getLogger(__name__)


def _sample_style(name: str, samples_meta: Dict[str, Any], i: int):
    meta = samples_meta.get(name, {})
    color = meta.get("color") or plt.get_cmap("tab20")(i % 20)
    label = style.tlatex(meta.get("label", name))
    return color, label


def data_per_channel(model, data) -> Dict[str, np.ndarray]:
    data = np.asarray(data, dtype=float)[: model.config.nmaindata]
    return {ch: data[model.config.channel_slices[ch]] for ch in model.config.channels}


def plot_region(
    prediction,
    data: Dict[str, np.ndarray],
    channel: str,
    region_meta: Dict[str, Any],
    samples_meta: Dict[str, Any],
    output: pathlib.Path,
    *,
    blinded: bool = False,
    atlas: str = "Internal",
    lumi_fb: Optional[float] = None,
    log_y: Optional[bool] = None,
    formats=("pdf", "png"),
):
    model = prediction.model
    ich = model.config.channels.index(channel)
    samples = model.config.samples
    yields = np.asarray(prediction.model_yields[ich])  # sample x bin
    total = yields.sum(axis=0)
    total_unc = np.asarray(prediction.total_stdev_model_bins[ich][-1])
    edges = np.asarray(region_meta.get("bins") or np.arange(len(total) + 1), dtype=float)
    if len(edges) != len(total) + 1:
        edges = np.arange(len(total) + 1, dtype=float)
    centers = 0.5 * (edges[1:] + edges[:-1])

    fig, (ax, rax) = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    order = np.argsort(yields.sum(axis=1))  # smallest at the bottom of the legend
    bottom = np.zeros_like(total)
    for i in order:
        if yields[i].sum() <= 0:
            continue
        color, label = _sample_style(samples[i], samples_meta, i)
        ax.stairs(bottom + yields[i], edges, baseline=bottom, fill=True, color=color, label=label, linewidth=0)
        bottom = bottom + yields[i]
    ax.stairs(total, edges, color="black", linewidth=1)
    ax.stairs(total + total_unc, edges, baseline=total - total_unc, fill=True, facecolor="none", hatch="////",
              edgecolor="gray", linewidth=0, label="Uncertainty")
    obs = np.asarray(data[channel])
    if not blinded:
        ax.errorbar(centers, obs, yerr=np.sqrt(np.maximum(obs, 0)), fmt="ko", markersize=4, label="Data")
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(total > 0, obs / total, np.nan)
            ratio_err = np.where(total > 0, np.sqrt(np.maximum(obs, 0)) / total, np.nan)
        rax.errorbar(centers, ratio, yerr=ratio_err, fmt="ko", markersize=4)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(total > 0, total_unc / total, 0)
    rax.stairs(1 + rel, edges, baseline=1 - rel, fill=True, facecolor="none", hatch="////", edgecolor="gray", linewidth=0)
    rax.axhline(1, color="black", linewidth=0.8, linestyle="--")
    rax.set_ylim(0.0, 2.0)
    rax.set_ylabel("Data / Bkg.")
    variable = region_meta.get("variable", "cuts")
    rax.set_xlabel(style.tlatex(region_meta.get("label", channel)) if variable != "cuts" else channel)
    if variable == "cuts":
        rax.set_xticks([])
    ax.set_ylabel("Events")
    ymax = max(float(np.max(total + total_unc)), float(np.max(obs)) if not blinded else 0.0, 1e-3)
    if log_y is None:
        log_y = len(total) > 1 and ymax / max(float(np.min(total[total > 0])) if np.any(total > 0) else 1, 1e-3) > 50
    if log_y:
        ax.set_yscale("log")
        ax.set_ylim(max(1e-2, 0.5 * float(np.min(total[total > 0])) if np.any(total > 0) else 1e-2), ymax * 200)
    else:
        ax.set_ylim(0, ymax * 1.6)
    ax.set_xlim(edges[0], edges[-1])
    extra = [f"{channel}, {prediction.label}" + (" (blinded)" if blinded else "")]
    style.atlas_label(ax, atlas, lumi_fb, extra)
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=1)
    return style.save(fig, output, formats)


def plot_summary(
    prediction,
    data: Dict[str, np.ndarray],
    channels: Sequence[str],
    samples_meta: Dict[str, Any],
    output: pathlib.Path,
    *,
    blinded: Sequence[str] = (),
    atlas: str = "Internal",
    lumi_fb: Optional[float] = None,
    formats=("pdf", "png"),
    title: str = "",
):
    """One entry per region (per bin for binned regions), with a significance panel."""
    model = prediction.model
    samples = model.config.samples
    labels, stack, totals, uncs, obs, is_blind = [], [], [], [], [], []
    for channel in channels:
        ich = model.config.channels.index(channel)
        y = np.asarray(prediction.model_yields[ich])
        unc = np.asarray(prediction.total_stdev_model_bins[ich][-1])
        for b in range(y.shape[1]):
            labels.append(channel if y.shape[1] == 1 else f"{channel}[{b}]")
            stack.append(y[:, b])
            totals.append(y[:, b].sum())
            uncs.append(unc[b])
            obs.append(data[channel][b])
            is_blind.append(channel in blinded)
    stack = np.asarray(stack).T  # sample x entry
    totals, uncs, obs = np.asarray(totals), np.asarray(uncs), np.asarray(obs)
    edges = np.arange(len(labels) + 1, dtype=float)
    centers = edges[:-1] + 0.5
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(max(6.4, 0.5 * len(labels) + 2), 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    bottom = np.zeros(len(labels))
    for i in np.argsort(stack.sum(axis=1)):
        if stack[i].sum() <= 0:
            continue
        color, label = _sample_style(samples[i], samples_meta, i)
        ax.stairs(bottom + stack[i], edges, baseline=bottom, fill=True, color=color, label=label, linewidth=0)
        bottom = bottom + stack[i]
    ax.stairs(totals + uncs, edges, baseline=np.maximum(totals - uncs, 1e-3), fill=True, facecolor="none",
              hatch="////", edgecolor="gray", linewidth=0, label="Uncertainty")
    mask = ~np.asarray(is_blind)
    ax.errorbar(centers[mask], obs[mask], yerr=np.sqrt(np.maximum(obs[mask], 0)), fmt="ko", markersize=4, label="Data")
    ax.set_yscale("log")
    ax.set_ylim(max(0.05, 0.3 * float(np.min(totals[totals > 0])) if np.any(totals > 0) else 0.05),
                float(np.max(np.maximum(totals + uncs, obs))) * 300)
    ax.set_ylabel("Events")
    with np.errstate(divide="ignore", invalid="ignore"):
        signif = (obs - totals) / np.sqrt(totals + uncs**2)
    signif[~mask] = np.nan
    rax.bar(centers, np.nan_to_num(signif), width=0.8, color="#d62728")
    rax.axhline(0, color="black", linewidth=0.8)
    rax.set_ylim(-3.5, 3.5)
    rax.set_ylabel(r"$(n_\mathrm{obs}-n_\mathrm{pred})/\sigma_\mathrm{tot}$", fontsize=9)
    rax.set_xticks(centers)
    rax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_xlim(0, len(labels))
    style.atlas_label(ax, atlas, lumi_fb, [title or prediction.label])
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    return style.save(fig, output, formats)
