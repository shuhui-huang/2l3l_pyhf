import numpy as np
import pytest

from fit2l3l.workspace import expressions as E

ARR = {
    "nSigLep": np.array([2, 3, 3, 3]),
    "mSFOS": np.array([90e3, 90e3, 70e3, 110e3]),
    "met": np.array([1.0, 2.0, 3.0, 4.0]),
    "meff": np.array([10.0, 1.0, 10.0, 10.0]),
    "isSS15": np.array([1, 0, 1, 1], dtype=np.int32),
}


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("(nSigLep>=3)&& !(mSFOS>81000 && mSFOS<101000)&& (met/meff >=0.2)", [False, False, True, True]),
        ("nSigLep==3 && isSS15", [False, False, True, True]),
        ("!isSS15", [False, True, False, False]),
        ("81000<mSFOS<101000", [True, True, False, False]),
        ("(nSigLep>=3) & (met>1)", [False, True, True, True]),
        ("!(nSigLep==3) || met>3.5", [True, False, False, True]),
        ("TMath::Abs(met-2)<1.5", [True, True, True, False]),
    ],
)
def test_cpp_selections(expr, expected):
    assert E.evaluate(expr, ARR, 4).tolist() == expected


def test_values_and_symbols():
    assert E.evaluate("1", ARR, 4).tolist() == [1, 1, 1, 1]
    assert E.evaluate("met if nSigLep==3 else 0", ARR, 4).tolist() == [0, 2, 3, 4]
    assert E.free_symbols("met/ht_lept > 0.3 && sqrt(meff) > 1") == {"met", "ht_lept", "meff"}


def test_product_and_and():
    assert E.product(["w1", "lumiScaling", "1"]) == "w1*lumiScaling"
    assert E.and_("a>1", "b<2 || c") == "(a>1) && (b<2 || c)"


@pytest.mark.parametrize("bad", ['__import__("os")', "met.sum()", "[met]"])
def test_unsafe_expressions_rejected(bad):
    with pytest.raises(E.ExpressionError):
        E.evaluate(bad, ARR, 4)
