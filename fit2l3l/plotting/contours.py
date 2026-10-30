"""Exclusion contours and 1D upper-limit plots from merged hypotest results.

Replaces HistFitter's ``GenerateJSONOutput.py`` / ``harvestToContours.py`` /
``multiplexJSON.py`` / ``plotContours.py`` chain:

* CLs values are converted to significances Z = Phi^-1(1 - CLs) and interpolated on a
  regular grid (``linear``, ``cubic`` or ``rbf``) inside the convex hull of the
  signal points and below the kinematic limit; the 95% CL contour is Z = 1.645
* observed (with the signal cross-section +-1 sigma theory lines), expected and
  the +-1 sigma expected band; previous limits from HEPData csv files
* ``BestSR``: per signal point the result with the best expected CLs among several
  signal regions is used (``multiplexJSON``), and a map of the chosen SR is drawn
* grey numbers: CLs or excluded cross-section per point
* 1D: upper limits on the cross-section vs mass with the theory prediction
* contour lines are also written as JSON (HEPData input)
"""

from __future__ import annotations

import logging
import math
import pathlib
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy import interpolate, spatial
from scipy.stats import norm

from fit2l3l.plotting import style
from fit2l3l.plotting.style import plt
from fit2l3l.utils import load_analysis, read_json, share_path, write_json

log = logging.getLogger(__name__)

Z95 = float(norm.isf(0.05))  # 1.645
CONTOUR_KEYS = {
    "observed": "CLs",
    "expected": "CLsexp",
    "expected_m1s": "clsd1s",
    "expected_p1s": "clsu1s",
    "expected_m2s": "clsd2s",
    "expected_p2s": "clsu2s",
    "observed_theory_up": "CLs_theory_up",
    "observed_theory_down": "CLs_theory_down",
}


def load_points(path) -> List[Dict[str, Any]]:
    points = read_json(path)
    return [p for p in points if not p.get("failedfit") and _finite(p.get("CLsexp"))]


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def best_of(results: Dict[str, List[Dict[str, Any]]], key: str = "CLsexp") -> List[Dict[str, Any]]:
    """Per mass point, takes the result with the lowest expected CLs (multiplexJSON)."""
    best: Dict[tuple, Dict[str, Any]] = {}
    for label, points in results.items():
        for p in points:
            mass = tuple(round(float(m), 6) for m in p.get("masses", [p["m0"], p["m12"]]))
            if not _finite(p.get(key)):
                continue
            if mass not in best or p[key] < best[mass][key]:
                best[mass] = dict(p, bestSR=label)
    return list(best.values())


def _significance(cls) -> np.ndarray:
    cls = np.clip(np.asarray(cls, dtype=float), 1e-12, 1 - 1e-12)
    return norm.isf(cls)


def _forbidden_mask(xx, yy, model_cfg: Dict[str, Any]) -> np.ndarray:
    forb = model_cfg.get("forbidden")
    if not forb:
        return np.zeros_like(xx, dtype=bool)
    function = forb.get("function")
    cut = float(forb.get("cut", 0.0))
    if function:
        try:
            limit = eval(function, {"__builtins__": {}}, {"x": xx, "np": np})  # noqa: S307 - plotting config
        except Exception:
            limit = xx - cut
        limit = np.broadcast_to(np.asarray(limit, dtype=float), xx.shape)
    else:
        limit = xx - cut
    return yy > limit


def interpolate_z(x, y, values, xx, yy, method="linear", smoothing=0.0, xy_scale=None):
    """Interpolates significances on the grid (NaN outside the convex hull)."""
    pts = np.column_stack([x, y]).astype(float)
    z = _significance(values)
    scale = xy_scale or (np.ptp(pts[:, 0]) or 1.0, np.ptp(pts[:, 1]) or 1.0)
    pts_s = pts / scale
    grid_s = np.column_stack([xx.ravel() / scale[0], yy.ravel() / scale[1]])
    if method in ("linear", "cubic", "nearest"):
        zz = interpolate.griddata(pts_s, z, grid_s, method=method)
    elif method == "rbf":
        rbf = interpolate.RBFInterpolator(pts_s, z, kernel="thin_plate_spline", smoothing=float(smoothing))
        zz = rbf(grid_s)
        hull = spatial.Delaunay(pts_s)
        zz[hull.find_simplex(grid_s) < 0] = np.nan
    else:
        raise ValueError(f"unknown interpolation '{method}'")
    return zz.reshape(xx.shape)


def contour_lines(xx, yy, zz, level=Z95) -> List[np.ndarray]:
    import contourpy

    gen = contourpy.contour_generator(xx, yy, np.ma.masked_invalid(zz))
    return [np.asarray(line) for line in gen.lines(level) if len(line) > 1]


def _model_cfg(meta: Dict[str, Any], name: Optional[str]) -> Dict[str, Any]:
    grid = name or meta.get("signal", {}).get("grid")
    try:
        analysis = load_analysis(meta.get("analysis", "ss3l"))
        return dict(analysis.plotting_models.get(grid, {}))
    except Exception as exc:  # pragma: no cover
        log.warning(f"cannot load plotting model {grid}: {exc}")
        return {}


def _previous_limits(ax, model_cfg, x_range, y_range, one_d=False):
    for prev in model_cfg.get("previous_limits", []):
        path = share_path(prev["csv"])
        if not path.exists():
            log.warning(f"previous limit {path} not found")
            continue
        data = np.genfromtxt(path, delimiter=",", comments="#")
        data = data[np.all(np.isfinite(data), axis=1)]
        if data.size == 0:
            continue
        label = style.tlatex(prev.get("label", ""))
        xs, ys = data[:, 0], data[:, 1] * float(prev.get("scale", 1.0))
        if prev.get("line") or one_d:
            ax.plot(xs, ys, color="gray", linestyle="-.", linewidth=1.5, label=label)
        else:
            poly_x = np.append(xs, [x_range[0], x_range[0]])
            poly_y = np.append(ys, [y_range[0], ys[0]])
            ax.fill(poly_x, poly_y, color="lightgray", alpha=0.5, label=label, zorder=0)


def _forbidden_line(ax, model_cfg, x_range, y_range):
    forb = model_cfg.get("forbidden")
    if not forb:
        return
    cut = float(forb.get("cut", 0.0))
    xs = np.linspace(x_range[0], x_range[1], 100)
    ys = xs - cut
    ok = (ys >= y_range[0]) & (ys <= y_range[1])
    if not np.any(ok):
        return
    ax.plot(xs[ok], ys[ok], color="gray", linestyle="--", linewidth=1)
    if forb.get("text"):
        p1 = ax.transData.transform((xs[ok][0], ys[ok][0]))
        p2 = ax.transData.transform((xs[ok][-1], ys[ok][-1]))
        angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
        ax.text(float(forb.get("label_x", xs[ok][len(xs[ok]) // 2])), float(forb.get("label_y", ys[ok][len(ys[ok]) // 2])),
                style.tlatex(forb["text"]), rotation=angle, rotation_mode="anchor", fontsize=8, color="gray")


def plot_contours(
    points: List[Dict[str, Any]],
    model_cfg: Dict[str, Any],
    output: pathlib.Path,
    *,
    x_key: str = "m0",
    y_key: str = "m12",
    interpolation: str = "linear",
    smoothing: float = 0.0,
    draw_cls: bool = False,
    draw_cls_exp: bool = False,
    draw_xs: bool = False,
    show_points: bool = True,
    atlas: str = "Internal",
    lumi_fb: Optional[float] = None,
    formats=("pdf", "png"),
    title: str = "",
    npix: int = 250,
) -> Dict[str, Any]:
    if len(points) < 3:
        raise ValueError("need at least 3 signal points for a contour")
    x = np.asarray([float(p[x_key]) for p in points])
    y = np.asarray([float(p[y_key]) for p in points])
    labels = model_cfg.get("labels", (x_key, y_key, ""))
    x_range = model_cfg.get("x_range", (x.min(), x.max()))
    y_range = model_cfg.get("y_range", (y.min(), y.max()))
    crange = model_cfg.get("contour_range")
    if crange and crange[1] > crange[0] and crange[3] > crange[2]:
        gx = np.linspace(crange[0], crange[1], npix)
        gy = np.linspace(crange[2], crange[3], npix)
    else:
        gx = np.linspace(min(x.min(), x_range[0]), max(x.max(), x_range[1]), npix)
        gy = np.linspace(min(y.min(), y_range[0]), max(y.max(), y_range[1]), npix)
    xx, yy = np.meshgrid(gx, gy)
    forbidden = _forbidden_mask(xx, yy, model_cfg)

    lines: Dict[str, List[np.ndarray]] = {}
    fields: Dict[str, np.ndarray] = {}
    for name, key in CONTOUR_KEYS.items():
        vals = [p.get(key) for p in points]
        mask = np.asarray([_finite(v) for v in vals])
        if mask.sum() < 3:
            continue
        zz = interpolate_z(x[mask], y[mask], np.asarray(vals, dtype=object)[mask].astype(float), xx, yy,
                           method=interpolation, smoothing=smoothing)
        zz[forbidden] = np.nan
        fields[name] = zz
        lines[name] = contour_lines(xx, yy, zz)

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    _previous_limits(ax, model_cfg, x_range, y_range)
    if "expected_m1s" in fields and "expected_p1s" in fields:
        zhi = np.fmax(fields["expected_m1s"], fields["expected_p1s"])
        zlo = np.fmin(fields["expected_m1s"], fields["expected_p1s"])
        # inside the band: excluded by the optimistic but not by the pessimistic edge;
        # a continuous field keeps the band edges smooth
        band = np.fmin(zhi - Z95, Z95 - zlo)
        ax.contourf(xx, yy, np.ma.masked_invalid(band), levels=[0.0, np.inf], colors=["#ffd700"], alpha=0.9, zorder=1)
        ax.fill_between([], [], color="#ffd700", label=r"Expected limit ($\pm 1 \sigma_\mathrm{exp}$)")
    for line in lines.get("expected", []):
        ax.plot(line[:, 0], line[:, 1], color="#1f3f8f", linestyle="--", linewidth=2, zorder=3)
    if "expected" in lines and "expected_m1s" not in fields:
        ax.plot([], [], color="#1f3f8f", linestyle="--", linewidth=2, label="Expected limit")
    elif "expected" in lines:
        ax.plot([], [], color="#1f3f8f", linestyle="--", linewidth=2)
    for line in lines.get("observed", []):
        ax.plot(line[:, 0], line[:, 1], color="#c00000", linewidth=2.5, zorder=4)
    for name in ("observed_theory_up", "observed_theory_down"):
        for line in lines.get(name, []):
            ax.plot(line[:, 0], line[:, 1], color="#c00000", linestyle=":", linewidth=1.5, zorder=4)
    if "observed" in lines:
        lab = r"Observed limit ($\pm 1 \sigma_\mathrm{theory}^\mathrm{SUSY}$)" if "observed_theory_up" in lines else "Observed limit"
        ax.plot([], [], color="#c00000", linewidth=2.5, label=lab)
    if show_points:
        ax.scatter(x, y, s=6, color="black", alpha=0.4, zorder=2)
    for flag, key, fmt in ((draw_cls, "CLs", "{:.2f}"), (draw_cls_exp, "CLsexp", "{:.2f}"), (draw_xs, "excludedXsec", "{:.2g}")):
        if not flag:
            continue
        for p in points:
            if _finite(p.get(key)):
                ax.text(float(p[x_key]), float(p[y_key]), fmt.format(float(p[key])), fontsize=5, color="gray",
                        ha="center", va="center", zorder=5)
    _forbidden_line(ax, model_cfg, x_range, y_range)
    ax.set_xlabel(style.tlatex(labels[0]))
    ax.set_ylabel(style.tlatex(labels[1]))
    extra = [labels[2]] if len(labels) > 2 and labels[2] else []
    if title:
        extra.append(title)
    style.atlas_label(ax, atlas, lumi_fb, [], y=0.97)
    if extra:
        ax.set_title(style.tlatex(extra[0]), fontsize=7, loc="left")
    ax.text(0.04, 0.80, "All limits at 95% CL", transform=ax.transAxes, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    written = style.save(fig, output, formats)
    write_json({k: [l.tolist() for l in v] for k, v in lines.items()}, output.with_name(output.name + "_contours.json"))
    return {"plots": written, "lines": {k: len(v) for k, v in lines.items()}}


def plot_best_sr_map(points, output, *, x_key="m0", y_key="m12", model_cfg=None, formats=("pdf", "png"), atlas="Internal"):
    model_cfg = model_cfg or {}
    srs = sorted({p["bestSR"] for p in points})
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    for i, sr in enumerate(srs):
        sel = [p for p in points if p["bestSR"] == sr]
        ax.scatter([float(p[x_key]) for p in sel], [float(p[y_key]) for p in sel], color=cmap(i % 10), label=sr, s=25)
    labels = model_cfg.get("labels", (x_key, y_key, ""))
    ax.set_xlabel(style.tlatex(labels[0]))
    ax.set_ylabel(style.tlatex(labels[1]))
    if "x_range" in model_cfg:
        ax.set_xlim(*model_cfg["x_range"])
        ax.set_ylim(*model_cfg["y_range"])
    style.atlas_label(ax, atlas)
    ax.legend(title="Best expected SR", fontsize=8, frameon=False, loc="upper right")
    return style.save(fig, output, formats)


def plot_1d(points, model_cfg, output, *, x_key="m0", atlas="Internal", lumi_fb=None, formats=("pdf", "png")):
    """Upper limit on the cross-section vs mass (HistFitter --draw1D)."""
    pts = sorted([p for p in points if _finite(p.get("upperLimit")) and p.get("xsec")], key=lambda p: float(p[x_key]))
    if not pts:
        raise ValueError("no points with upper limits and cross-sections (run with UpperLimit enabled)")
    labels = model_cfg.get("labels", (x_key, r"$\sigma$ [fb]", ""))
    unit = 1e-3 if "pb" in labels[1] else 1.0  # results are in fb
    x = np.asarray([float(p[x_key]) for p in pts])
    xs = np.asarray([float(p["xsec"]) for p in pts]) * unit
    obs = np.asarray([float(p["upperLimit"]) for p in pts]) * xs
    exp = {k: np.asarray([float(p.get(k, np.nan)) for p in pts]) * xs for k in (
        "expectedUpperLimitMinus2Sig", "expectedUpperLimitMinus1Sig", "expectedUpperLimit",
        "expectedUpperLimitPlus1Sig", "expectedUpperLimitPlus2Sig")}
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.fill_between(x, exp["expectedUpperLimitMinus2Sig"], exp["expectedUpperLimitPlus2Sig"], color="#ffd700", label=r"Expected $\pm 2\sigma$")
    ax.fill_between(x, exp["expectedUpperLimitMinus1Sig"], exp["expectedUpperLimitPlus1Sig"], color="#00cc00", label=r"Expected $\pm 1\sigma$")
    ax.plot(x, exp["expectedUpperLimit"], "k--", label="Expected limit")
    ax.plot(x, obs, "k-", marker="o", markersize=3, label="Observed limit")
    up = np.asarray([float(p.get("xsec_rel_up") or 0) for p in pts])
    dn = np.asarray([float(p.get("xsec_rel_down") or 0) for p in pts])
    theory_label = style.tlatex(model_cfg.get("theory_legend") or "Theory")
    ax.plot(x, xs, color="#1f77b4", label=theory_label)
    if np.any(up > 0):
        ax.fill_between(x, xs * (1 - dn), xs * (1 + up), color="#1f77b4", alpha=0.3)
    _previous_limits(ax, model_cfg, model_cfg.get("x_range", (x.min(), x.max())), model_cfg.get("y_range", (0, 1)), one_d=True)
    if model_cfg.get("log_y", True):
        ax.set_yscale("log")
    if "x_range" in model_cfg:
        ax.set_xlim(*model_cfg["x_range"])
    if "y_range" in model_cfg:
        ax.set_ylim(*model_cfg["y_range"])
    ax.set_xlabel(style.tlatex(labels[0]))
    ax.set_ylabel(style.tlatex(labels[1]))
    style.atlas_label(ax, atlas, lumi_fb, [model_cfg.get("extra_text", "")])
    if len(labels) > 2:
        ax.set_title(style.tlatex(labels[2]), fontsize=7, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    return style.save(fig, output, formats)


def plot_from_config(cfg: Dict[str, Any], *, merged_file, meta: Dict[str, Any], outdir: pathlib.Path,
                     general: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Entry point used by the fit runner / CLI (``Contours`` block of the fit YAML)."""
    cfg = dict(cfg or {})
    general = general or {}
    formats = general.get("PlotFormats", ["pdf", "png"])
    atlas = general.get("ATLASLabel", "Internal")
    model_cfg = _model_cfg(meta, cfg.get("Model"))
    model_cfg.update(cfg.get("ModelOverrides", {}) or {})
    lumi = meta.get("lumi_fb")
    x_key, y_key = cfg.get("XKey", "m0"), cfg.get("YKey", "m12")
    name = cfg.get("Name") or meta.get("signal", {}).get("grid", "contour")
    xsec_info = {p["name"]: p for p in meta.get("signal", {}).get("points", [])}
    points = load_points(merged_file)
    for p in points:
        info = xsec_info.get(p["name"], {})
        for key in ("xsec_rel_up", "xsec_rel_down"):
            p.setdefault(key, info.get(key))
    out: Dict[str, Any] = {}
    best = cfg.get("BestSR")
    if best:
        results = {cfg.get("Label", meta.get("name", "this")): points}
        for label, path in best.items():
            results[label] = load_points(path)
        points = best_of(results)
        write_json(points, pathlib.Path(outdir) / f"{name}_best_hypotest_list.json")
        plot_best_sr_map(points, pathlib.Path(outdir) / f"{name}_bestSR", x_key=x_key, y_key=y_key,
                         model_cfg=model_cfg, formats=formats, atlas=atlas)
        name += "_best"
    if cfg.get("OneD"):
        out["1d"] = plot_1d(points, model_cfg, pathlib.Path(outdir) / f"{name}_upperlimit_1D", x_key=x_key,
                            atlas=atlas, lumi_fb=lumi, formats=formats)
    else:
        out["2d"] = plot_contours(
            points, model_cfg, pathlib.Path(outdir) / f"{name}_contour", x_key=x_key, y_key=y_key,
            interpolation=cfg.get("Interpolation", "linear"), smoothing=float(cfg.get("Smoothing", 0.0)),
            draw_cls=cfg.get("DrawCLs", False), draw_cls_exp=cfg.get("DrawCLsExp", False), draw_xs=cfg.get("DrawXS", False),
            show_points=cfg.get("ShowPoints", True), atlas=atlas, lumi_fb=lumi, formats=formats,
        )
    log.info(f"contour plots written to {outdir}")
    return out
