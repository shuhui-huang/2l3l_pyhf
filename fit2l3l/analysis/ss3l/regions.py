"""Region definitions of the SS/3L analysis (ported from ``fitConfigSS3L.py``).

* ``CUTS``      : region name -> selection.  The strings are copied verbatim from
  ``configMgr.cutsDict`` of the HistFitter config (C++ syntax is understood).
  Events in the ntuples are assumed to pass the SS/3L pre-selection.
* ``BINNINGS``  : region name -> {variable: (nbins, low, high)} for binned fits
  (formerly the ``binning`` block of ``hardCodedYields.json``).  The single-bin
  ``"cuts"`` binning is always available and does not need to be listed.
  Explicit bin edges can be given as a list, e.g. ``'met': [100e3, 200e3, 400e3]``.

Units follow the ntuples (MeV).

To add a region: add an entry to ``CUTS`` (and ``BINNINGS`` if you want to fit a
distribution), then list it in the ``Regions`` block of a workspace YAML.
"""

from fit2l3l.analysis.base import Region

CUTS = {
    # validation/control region
    'CRWZ2j': '(nSigLep==3 && NlepBL==3 && nBJets20==0 && (nJets25==2 || nJets25==3) && met>30000 && met<150000 && meff<1500000 && ht_lept>130000 && mSFOS>81000 && mSFOS<101000 && isSS15)',
    'VRWZ2j': '(nSigLep==3 && NlepBL==3 && nBJets20==0 && (nJets25==2 || nJets25==3) && met>30000 && met<150000 && meff<1500000 && ht_lept>130000 && mSFOS>81000 && mSFOS<101000 && isSS15)',
    'VRWZge4j': '(nSigLep==3 && NlepBL==3 && nBJets20==0 && nJets25>=4 && met>30000 && met<150000 && meff<1500000 && ht_lept>130000 && mSFOS>81000 && mSFOS<101000 && isSS15)',
    'VRWZ4j': '(nSigLep==3 && NlepBL==3 && nBJets20==0 && nJets25>=4 && meff>600000 && meff<1500000 && met>30000 && met<250000 && mSFOS>81000 && mSFOS<101000 && met/meff<0.2)',
    'VRWZ6j': '(nSigLep==3 && NlepBL==3 && nBJets20==0 && nJets25>=6 && meff>400000 && meff<1500000 && met>30000 && met<250000 && mSFOS>81000 && mSFOS<101000 && met/meff<0.15)',
    'VRTTV': '(nSigLep>=2 && NlepBL>=2 && nBJets20>=1 && nJets40>=3 && meff>600000 && meff<1500000 && met>30000 && met<250000 && isSS30 && dRl1j>1.1 && (ht_bjet/ht_jet)>0.4 && (met/meff)>0.1 && Pt_subl>30000)',
    'VRTTW': '(nSigLep==2 && NlepBL==2 && SSChannel==1 && nBJets20>=2 && nJets25>=2 && meff<1500000 && met>30000 && met<250000 && Pt_l>40000 && Pt_subl>25000)',
    'VRTTW3j': '(nSigLep==2 && NlepBL==2 && SSChannel==3 && nBJets20>=2 && nJets25>=3 && meff<1500000 && met>30000 && met<250000 && Pt_l>25000 && Pt_subl>25000)',
    # extra VRs
    'VRTTV1b6j': '(nSigLep>=2 && NlepBL>=2 && nBJets20>=1 && nJets40>=6 && meff<1500000 && met>30000 && met<250000 && isSS30 && (met/meff)<0.15)',
    'VRTTV6j': '(nSigLep>=2 && NlepBL>=2 && nJets40>=6 && meff<1200000 && met>30000 && met<250000 && isSS30 && met/meff<0.15)',
    'VRTTV2b': '(nSigLep>=2 && NlepBL>=2 && nBJets20>=2 && nJets40>=3 && meff>600000 && meff<1500000 && met>30000 && met<250000 && isSS30 && dRl1j>1 && (ht_bjet/ht_jet)>0.3 && (met/meff)>0.12)',
    'VRTTW1b': '(nSigLep==2 && NlepBL==2 && nBJets20>=1 && nJets40>=3 && meff>550000 && meff<1500000 && met>30000 && met<250000 && Pt_l>30000 && Pt_subl>30000 && dRl1j>1 && (ht_bjet/ht_jet)>0.42 && (met/meff)>0.15)',
    'VRTTZ': '(nSigLep>=3 && NlepBL>=3 && nBJets20>=1 && nJets40>=4 && meff>500000 && meff<1500000 && met>30000 && met<250000 && Pt_l>20000 && Pt_subl>20000 && (ht_bjet/ht_jet)>0.13 && mSFOS>81000 && mSFOS<101000 && met/meff<0.2)',
    'VRWW': '(nSigLep==2 && NlepBL==2 && nBJets20==0 && nJets50>=2 && meff>650000 && meff<1500000 && met>55000 && met<250000 && (!(mllSS>81000 && SSChannel==2 && mllSS<101000)) && dRl1j>0.7 && dRl2j>0.7 && dRSS>1.3)',
    'VRWZ2j1b': '(nSigLep==3 && NlepBL==3 && nBJets20>=1 && nJets25==2 && met>30000 && met<250000 && meff<1500000 && mSFOS>81000 && mSFOS<101000 && Pt_l>20000 && Pt_subl>20000 && Pt_thirdl>20000)',
    # Full Run-2 strong SRs:
    # RPV LQD
    'RpvLQD': '(nSigLep==2)&&(nJets50>=5)&&(nBJets20>=0)&&(meff>=2600000)',
    # RPV UDD
    'RpvUDD1b': '(nSigLep==2)&&(nJets50>=6)&&(nBJets20==1)&&(ht_jet>=1600000)',
    'RpvUDD2b': '(nSigLep==2)&&(nJets25>=2)&&(nBJets20==2)&&(ht_jet>=1700000)',
    'RpvUDDge2b': '(nSigLep==2)&&(nJets50>=5)&&(nBJets20>=2)&&(ht_jet>=1600000)',
    'RpvUDDge3b': '(nSigLep==2)&&(nJets50>=4)&&(nBJets20>=3)&&(meff>=1600000)',
    # RPC GG 2-step via slep
    'RpcGGslep1': '(nSigLep>=3 && nJets40 >=4 && nBJets20==0 && (mSFOS < 81000 || mSFOS > 101000) && met > ht_jet*0.4 && Pt_subl > 30*1000 && met > ht_lept*1.4)',
    'RpcGGslep2': '(nSigLep>=3 && nJets40 >=4 && nBJets20==0 && (mSFOS < 81000 || mSFOS > 101000) && met/ht_jet > 0.3 && Pt_subl > 70000 && dPhiLLmet > 0.7 && met> 150000)',
    'RpcGGslep3': '(nSigLep>=3 && nJets40 >=4 && nBJets20==0 && (mSFOS < 81000 || mSFOS > 101000) &&  ht_jet > 1200000 && met > 100000)',
    # RPC GG 2-step via WZ
    'RpcGGwz1': '(nSigLep>=2 && nBJets20==0 && nJets40>=6 && met>150000 && meff>2100000)',
    'RpcGGwz2': '(nSigLep>=2 && nBJets20==0 && nJets40>=6 && met>190000 && meff>1300000 && dPhiLLmet>0.8)',
    'RpcGGwz3': '(nSigLep>=2 && nBJets20==0 && nJets25>=6 && met>200000 && dPhiLLmet>0.2 && met_Sig>6 && meff/ht_lept>8 && NlepBL>=3)',
    # RPC SS 2-step via slep
    'RpcSSslep1': '(nSigLep==3 && NlepBL==3 && Pt_l>40000 && Pt_subl>40000 && Pt_thirdl>40000 && nBJets20==0 && nJets25>=2 && Pt_jet>60000 && Pt_subjet>60000 && met>=200000 && meff>=2000000 && dPhiLLmet>=0.3 && dRll>=0.5 && !(mSFOS>81000 && mSFOS<101000))',
    'RpcSSslep1excl': '(nSigLep==3 && NlepBL==3 && Pt_l>40000 && Pt_subl>40000 && Pt_thirdl>40000 && nBJets20==0 && nJets25>=2 && Pt_jet>60000 && Pt_subjet>60000 && met>=200000 && meff>=1000000 && dPhiLLmet>=0.3 && dRll>=0.5 && !(mSFOS>81000 && mSFOS<101000))',
    'RpcSSslep2': '(nSigLep==3 && NlepBL==3 && Pt_l>40000 && Pt_subl>40000 && Pt_thirdl>40000 && nBJets20==0 && nJets25>=2 && Pt_jet>60000 && Pt_subjet>60000 && met>=200000 && meff>=1000000 && met/ht_lept>=0.7 && dPhiLLmet>=0.5 && dRll>=0.2 && !(mSFOS>81000 && mSFOS<101000))',
    'RpcSSslep3': '(nSigLep==3 && NlepBL==3 && Pt_l<=60000 && nBJets20==0 && nJets25>=3 && Pt_jet>60000 && Pt_subjet>60000 && met>=100000 && meff>=600000 && ht_lept/ht_jet<=0.6 && dPhiLLmet>=1.4 && !(mSFOS>81000 && mSFOS<101000))',
    'RpcSSslep4': '(nSigLep==3 && NlepBL==3 && Pt_l>30000 && Pt_subl>30000 && Pt_thirdl>30000 && nBJets20==0 && nJets25>=3 && Pt_jet>60000 && Pt_subjet>60000 && met>=100000 && meff>=700000 && met/ht_lept>=0.7 && ht_lept/ht_jet<=0.6 && dPhiLLmet>=1.4 && !(mSFOS>81000 && mSFOS<101000))',
    # RPC SS 2-step via WZ
    'RpcSSwz1': '(nSigLep>=3)&& !(mSFOS>81000 && mSFOS<101000)&& (nBJets20==0)&& (meff>=0)&& (nJets25>=4)&& (met/meff >=0.2)&& (ht_lept/ht_jet <=0.2)',
    'RpcSSwz2': '(nSigLep>=3)&& (mSFOS>81000 && mSFOS<101000)&& (nBJets20==0)&& (meff>=800000)&& (met>=150000)&& (nJets25>=6)&& (met/ht_lept >=1.2)&& (ht_lept/ht_jet <=0.3)',
    'RpcSSwz3': '(nSigLep>=3)&& (mSFOS>81000 && mSFOS<101000)&& (nBJets20==0)&& (meff>=900000)&& (met>=200000)&& (nJets40>=5)&& (met/ht_lept >=1.1)&& (ht_lept/ht_jet <=0.4)',
    'RpcSSwz4': '(nSigLep>=3)&& (nBJets20==0)&& (meff>=1500000)&& (met>=250000)&& (nJets40>=5)&& (met/ht_lept >=0.3)&& (ht_lept/ht_jet <=0.7)',
    # N-1 plots for unblinding
    # RPC GG 2-step via WZ
    'RpcGGwz1LooseMeff': '(nSigLep>=2 && nBJets20==0 && nJets40>=6 && met>150000 && meff>600000)',
    'RpcGGwz1LooseNJets': '(nSigLep>=2 && nBJets20==0 && nJets40>=2 && met>150000 && meff>2100000)',
    'RpcGGwz2LooseNJets': '(nSigLep>=2 && nBJets20==0 && nJets40>=2 && met>190000 && meff>1300000 && dPhiLLmet>0.8)',
    'RpcGGwz3LooseNJets': '(nSigLep>=2 && nBJets20==0 && nJets25>=2 && met>200000 && dPhiLLmet>0.2 && met_Sig>6 && meff/ht_lept>8 && NlepBL>=3)',
    # RPC SS 2-step via WZ
    'RpcSSwz3Loose1': '(nSigLep>=3)&& (mSFOS>81000 && mSFOS<101000)&& (nBJets20==0)&& (meff>=800000)&& (met>=200000)&& (nJets40>=2)&& (met/ht_lept >=1.1)',
    'RpcSSwz3Loose2': '(nSigLep>=3)&& (mSFOS>81000 && mSFOS<101000)&& (nBJets20==0)&& (meff>=900000)&& (met>=200000)&& (nJets40>=2)&& (met/ht_lept >=1.1)&& (ht_lept/ht_jet <=0.7)',
    'RpcSSwz4Loose1': '(nSigLep>=3)&& (nBJets20==0)&& (meff>=800000)&& (met>=200000)&& (nJets40>=2)&& (met/ht_lept >=0.3)&& (ht_lept/ht_jet <=1)',
    'RpcSSwz4Loose2': '(nSigLep>=3)&& (nBJets20==0)&& (meff>=800000)&& (met>=200000)&& (nJets40>=2)&& (met/ht_lept >=0.3)&& (ht_lept/ht_jet <=0.7)',
    # SR N-1 plots for CONF note
    # RPV LQD
    'RpvLQD_noMeff': '(nSigLep==2)&&(nJets50>=5)&&(nBJets20>=0)',
    # RPV UDD
    'RpvUDD1b_noHtjet': '(nSigLep==2)&&(nJets50>=6)&&(nBJets20==1)',
    'RpvUDD2b_noHtjet': '(nSigLep==2)&&(nJets25>=2)&&(nBJets20==2)',
    'RpvUDDge2b_noHtjet': '(nSigLep==2)&&(nJets50>=5)&&(nBJets20>=2)',
    'RpvUDDge3b_noMeff': '(nSigLep==2)&&(nJets50>=4)&&(nBJets20>=3)',
    # RPC GG 2-step via sleptons
    'RpcGGslep1_noMetOverHtlep': '(nSigLep>=3 && nJets40 >=4 && nBJets20==0 && (mSFOS < 81000 || mSFOS > 101000) && met > ht_jet*0.4 && Pt_subl > 30*1000)',
    'RpcGGslep2_noMet': '(nSigLep>=3 && nJets40 >=4 && nBJets20==0 && (mSFOS < 81000 || mSFOS > 101000) && met/ht_jet > 0.3 && Pt_subl > 70000 && dPhiLLmet > 0.7)',
    'RpcGGslep3_noMet': '(nSigLep>=3 && nJets40 >=4 && nBJets20==0 && (mSFOS < 81000 || mSFOS > 101000) &&  ht_jet > 1200000)',
    # RPC GG 2-step via WZ
    'RpcGGwz1_noMet': '(nSigLep>=2 && nBJets20==0 && nJets40>=6 && meff>2100000)',
    'RpcGGwz1_noMeff': '(nSigLep>=2 && nBJets20==0 && nJets40>=6 && met>150000)',
    'RpcGGwz2_noMeff': '(nSigLep>=2 && nBJets20==0 && nJets40>=6 && met>190000 && dPhiLLmet>0.8)',
    'RpcGGwz3_noMet': '(nSigLep>=2 && nBJets20==0 && nJets25>=6 && dPhiLLmet>0.2 && met_Sig>6 && meff/ht_lept>8 && NlepBL>=3)',
    # RPC SS 2-step via WZ
    'RpcSSwz1_noMet': '(nSigLep>=3)&& !(mSFOS>81000 && mSFOS<101000)&& (nBJets20==0)&& (meff>=0)&& (nJets25>=4)&& (met/meff >=0.2)&& (ht_lept/ht_jet <=0.2)',
    'RpcSSwz2_noMet': '(nSigLep>=3)&& (mSFOS>81000 && mSFOS<101000)&& (nBJets20==0)&& (meff>=800000)&& (met>=0)&& (nJets25>=6)&& (met/ht_lept >=1.2)&& (ht_lept/ht_jet <=0.3)',
    'RpcSSwz3_noMeff': '(nSigLep>=3)&& (mSFOS>81000 && mSFOS<101000)&& (nBJets20==0)&& (meff>=0)&& (met>=200000)&& (nJets40>=5)&& (met/ht_lept >=1.1)&& (ht_lept/ht_jet <=0.4)',
    'RpcSSwz4_noMeff': '(nSigLep>=3)&& (nBJets20==0)&& (meff>=0)&& (met>=250000)&& (nJets40>=5)&& (met/ht_lept >=0.3)&& (ht_lept/ht_jet <=0.7)',
    # RPC SS 2-step via sleptons
    'RpcSSslep1excl_noMeff': '(nSigLep==3 && NlepBL==3 && Pt_l>40000 && Pt_subl>40000 && Pt_thirdl>40000 && nBJets20==0 && nJets25>=2 && Pt_jet>60000 && Pt_subjet>60000 && met>=200000 && meff>=0 && dPhiLLmet>=0.3 && dRll>=0.5 && !(mSFOS>81000 && mSFOS<101000))',
    'RpcSSslep2_noMeff': '(nSigLep==3 && NlepBL==3 && Pt_l>40000 && Pt_subl>40000 && Pt_thirdl>40000 && nBJets20==0 && nJets25>=2 && Pt_jet>60000 && Pt_subjet>60000 && met>=200000 && meff>=0 && met/ht_lept>=0.7 && dPhiLLmet>=0.5 && dRll>=0.2 && !(mSFOS>81000 && mSFOS<101000))',
    'RpcSSslep3_noMeff': '(nSigLep==3 && NlepBL==3 && Pt_l<=60000 && nBJets20==0 && nJets25>=3 && Pt_jet>60000 && Pt_subjet>60000 && met>=100000 && meff>=0 && ht_lept/ht_jet<=0.6 && dPhiLLmet>=1.4 && !(mSFOS>81000 && mSFOS<101000))',
    'RpcSSslep4_noMeff': '(nSigLep==3 && NlepBL==3 && Pt_l>30000 && Pt_subl>30000 && Pt_thirdl>30000 && nBJets20==0 && nJets25>=3 && Pt_jet>60000 && Pt_subjet>60000 && met>=100000 && meff>=0 && met/ht_lept>=0.7 && ht_lept/ht_jet<=0.6 && dPhiLLmet>=1.4 && !(mSFOS>81000 && mSFOS<101000))',
}

BINNINGS = {
    'RpvLQD': {'meff': (5, 2600000, 3600000)},
    'RpvUDD1b': {'ht_jet': (5, 1600000, 3100000)},
    'RpvUDD2b': {},
    'RpvUDDge2b': {'ht_jet': (5, 1600000, 3100000)},
    'RpvUDDge3b': {},
    'RpcGGslep1': {'met/ht_lept': (2, 1.4, 2.6)},
    'RpcGGslep2': {'met': (2, 150000, 350000)},
    'RpcGGslep3': {'met': (2, 100000, 400000)},
    'RpcGGwz1': {},
    'RpcGGwz2': {'meff': (2, 1300000, 2300000)},
    'RpcGGwz3': {},
    'RpcSSslep1': {},
    'RpcSSslep1excl': {'meff': (5, 1000000, 2250000)},
    'RpcSSslep2': {'meff': (4, 1000000, 2000000)},
    'RpcSSslep3': {'meff': (4, 600000, 1000000)},
    'RpcSSslep4': {'meff': (3, 700000, 1300000)},
    'RpcSSwz1': {'met': (3, 100000, 400000)},
    'RpcSSwz2': {'met': (3, 150000, 450000)},
    'RpcSSwz3': {'meff': (4, 900000, 1900000)},
    'RpcSSwz4': {'meff': (4, 1500000, 2300000)},
    'CRWZ2j': {'nJets25': (2, 1.5, 3.5), 'meff': (5, 200000, 1200000), 'met': (4, 30000, 150000)},
    'VRWZ2j': {'nJets25': (2, 1.5, 3.5), 'meff': (5, 200000, 1200000), 'met': (4, 30000, 150000)},
    'VRWZ4j': {'nJets25': (3, 3.5, 6.5), 'meff': (3, 600000, 1200000), 'met': (5, 30000, 250000)},
    'VRWZge4j': {'nJets25': (3, 3.5, 6.5)},
    'VRWZ6j': {'nJets25': (2, 5.5, 7.5), 'meff': (4, 400000, 1200000), 'met': (5, 30000, 250000)},
    'VRTTV': {},
    'VRTTW': {},
    'VRTTW3j': {},
    'VRTTV1b6j': {},
    'VRTTV6j': {},
    'VRTTV2b': {},
    'VRTTW1b': {'meff': (3, 400000, 1000000)},
    'VRTTZ': {},
    'VRWW': {'nJets25': (5, 1.5, 6.5)},
    'VRWZ2j1b': {},
    'RpcGGwz1LooseMeff': {'meff': (6, 600000, 2400000)},
    'RpcGGwz1LooseNJets': {'nJets40': (5, 1.5, 6.5)},
    'RpcGGwz2LooseNJets': {'nJets40': (5, 1.5, 6.5)},
    'RpcGGwz3LooseNJets': {'nJets25': (5, 1.5, 6.5)},
    'RpcSSwz3Loose1': {'nJets40': (4, 1.5, 5.5)},
    'RpcSSwz3Loose2': {'nJets40': (4, 1.5, 5.5)},
    'RpcSSwz4Loose1': {'nJets40': (4, 1.5, 5.5)},
    'RpcSSwz4Loose2': {'nJets40': (4, 1.5, 5.5)},
    'RpcSSslep1excl_noMeff': {'meff': (7, 500000, 2250000)},
    'RpcSSslep2_noMeff': {'meff': (6, 500000, 2000000)},
    'RpcSSslep3_noMeff': {'meff': (7, 300000, 1000000)},
    'RpcSSslep4_noMeff': {'meff': (4, 500000, 1300000)},
    'RpcSSwz2_noMet': {'met': (4, 50000, 450000)},
    'RpcSSwz3_noMeff': {'meff': (5, 650000, 1900000)},
    'RpcSSwz4_noMeff': {'meff': (7, 900000, 2300000)},
    'RpcGGslep1_noMetOverHtlep': {'met/ht_lept': (4, 0.2, 2.6)},
    'RpcGGslep2_noMet': {'met': (3, 50000, 350000)},
    'RpcGGslep3_noMet': {'met': (3, -50000, 400000)},
    'RpcGGwz1_noMet': {'met': (4, 0, 200000)},
    'RpcGGwz1_noMeff': {'meff': (4, 1500000, 2300000)},
    'RpcGGwz2_noMeff': {'meff': (3, 800000, 2300000)},
    'RpcGGwz3_noMet': {'met': (2, 150000, 250000)},
    'RpvLQD_noMeff': {'meff': (8, 2000000, 3600000)},
    'RpvUDD1b_noHtjet': {'ht_jet': (8, 700000, 3100000)},
    'RpvUDD2b_noHtjet': {'ht_jet': (4, 1100000, 1900000)},
    'RpvUDDge2b_noHtjet': {'ht_jet': (8, 700000, 3100000)},
    'RpvUDDge3b_noMeff': {'meff': (4, 1000000, 1800000)},
}

#: pretty labels of fit variables (plots / tables)
VARIABLE_LABELS = {
    "met": r"$E_\mathrm{T}^\mathrm{miss}$ [MeV]",
    "meff": r"$m_\mathrm{eff}$ [MeV]",
    "ht_jet": r"$H_\mathrm{T}^\mathrm{jet}$ [MeV]",
    "met/ht_lept": r"$E_\mathrm{T}^\mathrm{miss}/H_\mathrm{T}^\mathrm{lep}$",
    "nJets25": r"$N_\mathrm{jets}^{25}$",
    "nJets40": r"$N_\mathrm{jets}^{40}$",
}


def build_regions():
    """Returns {name: Region} for all regions with a selection."""
    regions = {}
    for name, cut in CUTS.items():
        regions[name] = Region(
            name=name,
            cut=cut,
            binnings=BINNINGS.get(name, {}),
            variable_labels=VARIABLE_LABELS,
        )
    return regions
