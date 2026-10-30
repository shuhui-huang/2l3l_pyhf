"""Plot styling helpers (ATLAS-like matplotlib style, TLatex -> mathtext)."""

from __future__ import annotations

import pathlib
import re
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# longest first, so that e.g. "#Deltam" -> "\Delta m"
_TLATEX_COMMANDS = sorted(
    [
        "tilde", "bar", "hat", "chi", "pm", "mp", "rightarrow", "leftarrow", "to", "Delta", "delta",
        "sigma", "tau", "nu", "mu", "alpha", "beta", "gamma", "Gamma", "eta", "phi", "pi", "ell",
        "approx", "sim", "geq", "leq", "times", "sqrt", "bf", "it", "rm", "infty", "lambda", "Lambda",
        "epsilon", "varepsilon", "theta", "omega", "rho", "kappa", "xi", "Upsilon",
    ],
    key=len,
    reverse=True,
)
_MPL_NAME = {"bf": "mathbf", "it": "mathit", "rm": "mathrm", "ell": "ell"}


def _split_splitline(text: str) -> str:
    while "#splitline{" in text:
        start = text.index("#splitline{")
        i = start + len("#splitline")
        groups = []
        for _ in range(2):
            depth, j = 0, i
            while j < len(text):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            groups.append(text[i + 1 : j])
            i = j + 1
        text = text[:start] + groups[0] + "\n" + groups[1] + text[i:]
    return text


def tlatex(text: Optional[str]) -> str:
    """Converts ROOT TLatex markup (``#tilde{g}``, ``m_{t}``) to matplotlib mathtext.

    Strings already containing ``$`` are returned unchanged.
    """
    if not text:
        return ""
    if "$" in text or ("#" not in text and "_{" not in text and "^{" not in text):
        return text
    text = _split_splitline(text)
    out = []
    i, n = 0, len(text)
    in_math = False
    depth = 0
    while i < n:
        c = text[i]
        starts_markup = c == "#" or (c in "_^" and i + 1 < n and text[i + 1] == "{")
        if starts_markup and not in_math:
            out.append("$")
            in_math = True
        if in_math:
            if c == "#":
                cmd = next((k for k in _TLATEX_COMMANDS if text.startswith(k, i + 1)), None)
                if cmd is None:
                    m = re.match(r"[A-Za-z]+", text[i + 1 :])
                    cmd = m.group(0) if m else ""
                out.append("\\" + _MPL_NAME.get(cmd, cmd))
                i += 1 + len(cmd)
                nxt = text[i] if i < n else ""
                if nxt.isalpha():
                    out.append(" ")
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            if c == "%":
                out.append(r"\%")
                i += 1
                continue
            if depth == 0 and c not in "{}_^#" and not c.isalnum() and c not in "*'":
                # end of the markup sequence
                if c == " " and i + 1 < n and (text[i + 1] == "#"):
                    out.append(r"\ ")
                    i += 1
                    continue
                out.append("$")
                in_math = False
                out.append(c)
                i += 1
                continue
            out.append(c)
            i += 1
            continue
        out.append(c)
        i += 1
    if in_math:
        out.append("$")
    result = "".join(out)
    # text inside math mode after a closing brace, e.g. "m_{t}" handled; clean "$$"
    return result.replace("$$", "")


def atlas_label(ax, label: str = "Internal", lumi_fb: Optional[float] = None, extra: Iterable[str] = (),
                x: float = 0.04, y: float = 0.95, fontsize: float = 12) -> None:
    text = r"$\mathbf{ATLAS}$" + (f" {label}" if label else "")
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fontsize, va="top", ha="left")
    lines = []
    if lumi_fb:
        lines.append(rf"$\sqrt{{s}}$ = 13 TeV, {lumi_fb:.1f} fb$^{{-1}}$")
    lines += [tlatex(e) for e in extra if e]
    for i, line in enumerate(lines):
        ax.text(x, y - 0.065 * (i + 1), line, transform=ax.transAxes, fontsize=fontsize - 2, va="top", ha="left")


def save(fig, path: pathlib.Path, formats: Iterable[str] = ("pdf", "png")) -> list:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=150 if fmt == "png" else None, bbox_inches="tight")
        written.append(out)
    plt.close(fig)
    return written
