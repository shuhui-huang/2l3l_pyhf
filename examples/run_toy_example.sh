#!/bin/bash
# End-to-end example on toy ntuples (no real inputs needed), ~10 minutes on one core.
#   source setup.sh && bash examples/run_toy_example.sh
set -e
cd "$(dirname "$0")/.."

# 1) toy ntuples in the HistFitter input format (only EG/MET kinematic trees)
fit2l3l toy-inputs --output InputTrees --grid Mc16SusyGG2StepWZ --schema SP:4,LSP:7 \
    --points 376776,376777,376778,376779,376780,376781,376786,376787,376788,376790,376791,376792 \
    --syst-groups EG,MET

SYST="--set Systematics=EG,MET,FT,PU,TRIG,FAKES,CHFLIP,TH"

# 2) cabinetry: workspaces (results/<Name>/ws.json)
fit2l3l build configs/workspace/bkgonly_RpcSSwz.yaml $SYST
fit2l3l build configs/workspace/disc_RpcGGwz2.yaml $SYST
fit2l3l build configs/workspace/excl_GG2StepWZ_RpcGGwz.yaml $SYST \
    --points 376776,376777,376778,376779,376780,376781,376786,376787,376788,376790,376791,376792
fit2l3l inspect results/bkgonly_RpcSSwz

# 3) pyhf: fits on ws.json (results/<Name>/fits/<fit>/)
fit2l3l fit configs/fits/bkgonly.yaml
fit2l3l fit configs/fits/discovery.yaml
fit2l3l fit configs/fits/exclusion.yaml --set 'UpperLimit={Enabled: false}'

# 4) contours again, with grey CLs numbers
fit2l3l contours results/excl_GG2StepWZ_RpcGGwz2/fits/exclusion/hypotest_list.json \
    --workspace results/excl_GG2StepWZ_RpcGGwz2 --config configs/fits/exclusion.yaml --draw-cls
