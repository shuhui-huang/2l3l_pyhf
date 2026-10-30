import numpy as np

from fit2l3l.fitting import toys
from fit2l3l.plotting.style import tlatex


def test_tlatex():
    assert tlatex("m(#tilde{b}_{1}) [GeV]") == r"m($\tilde{b}_{1}$) [GeV]"
    assert tlatex("plain") == "plain"
    assert "\n" in tlatex("#splitline{a}{b}")


def test_interpolate_limit():
    pois = np.linspace(0, 4, 9)
    cls = np.exp(-pois)  # crosses 0.05 at -ln(0.05) = 2.996
    assert abs(toys.interpolate_limit(pois, cls) - 2.9957) < 1e-3


def test_toy_cls_conventions():
    rng = np.random.default_rng(0)
    b = rng.chisquare(1, 20000)
    sb = rng.chisquare(1, 20000) * 0.3
    cls, clsb, clb, exp = toys._cls(2.0, sb, b)
    assert 0 < cls < 1 and clsb <= clb
    assert exp == sorted(exp)
