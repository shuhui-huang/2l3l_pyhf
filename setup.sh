#!/bin/bash
# Usage:  source setup.sh
#
# Creates (the first time) and activates a python virtual environment with fit2l3l,
# cabinetry and pyhf installed.  Works with any python >= 3.9 (e.g. lxplus EL9).
# Set FIT2L3L_VENV to put the environment somewhere else (e.g. on EOS/AFS work).

FIT2L3L_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export FIT2L3L_DIR
VENV="${FIT2L3L_VENV:-${FIT2L3L_DIR}/.venv}"
PYTHON="${FIT2L3L_PYTHON:-python3}"

if [ ! -f "${VENV}/bin/activate" ]; then
    echo "fit2l3l: creating virtual environment in ${VENV}"
    "${PYTHON}" -m venv "${VENV}" || return 1
    "${VENV}/bin/python" -m pip install --upgrade pip --quiet
    "${VENV}/bin/python" -m pip install -e "${FIT2L3L_DIR}[test]" --quiet || return 1
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
echo "fit2l3l: environment ready ($(python -c 'import cabinetry, pyhf; print("cabinetry", cabinetry.__version__, "/ pyhf", pyhf.__version__)'))"
echo "fit2l3l: run 'fit2l3l --help' to get started"
