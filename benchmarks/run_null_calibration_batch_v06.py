from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from gsrfs import GSRSelector
from gsrfs.synthetic import null_independent

ap=argparse.ArgumentParser(); ap.add_argument('--start',type=int,required=True); ap.add_argument('--stop',type=int,required=True); args=ap.parse_args()
out=Path(__file__).resolve().parent/'results'/'v06'; out.mkdir(parents=True,exist_ok=True)
path=out/'auto_calibration_v06_null_200_raw.csv'
if path.exists():
    df=pd.read_csv(path)
else:
    seed0=pd.read_csv(out/'auto_calibration_v06_null_signal_raw.csv')
    df=seed0[seed0.scenario=='null_independent'].copy()
for seed in range(args.start,args.stop):
    if seed in set(df.seed.astype(int)): continue
    X,_,_,name=null_independent(seed=seed)
    s=GSRSelector(n_features='auto',n_permutations=100,alpha=.05,max_pairs=5000,pair_sampler='balanced',residual_power=.5,allow_empty=True,support_engine='vectorized',residual_engine='batched_cd',random_state=seed).fit(X)
    idx=s.get_support(indices=True)
    row=pd.DataFrame([dict(scenario=name,seed=seed,n_selected=s.n_features_selected_,any_selected=int(s.n_features_selected_>0),selected=','.join(map(str,idx.tolist())),auto_threshold=s.auto_threshold_,max_observed_effective_support=float(np.max(s.effective_support_)),group_coverage=np.nan,relevant_precision=np.nan,redundancy_rate=np.nan)])
    df=pd.concat([df,row],ignore_index=True)
    # checkpoint every seed to make long calibration resumable
    df.sort_values('seed').to_csv(path,index=False)
print('stored',len(df),'null runs; false selections',int(df.any_selected.sum()),'rate',float(df.any_selected.mean()))
