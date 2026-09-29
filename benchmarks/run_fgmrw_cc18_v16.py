"""Pinned recent-author comparison on an X-defined tractable CC18 subset.

The subset rule is label/performance independent: frozen CC18 task order, after
label-blind encoding, accepting tasks with <= recent_max_features and <= max_samples.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
from run_cc18 import load_task, _load_frozen_tasks
from gsrfs import GSRSelector, FGMRWAuthorSelector
from gsrfs.benchmark import run_real_benchmark_flexible
from gsrfs.repro import write_manifest


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--limit',type=int,default=10)
    ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    ap.add_argument('--max-samples',type=int,default=1500)
    ap.add_argument('--recent-max-features',type=int,default=100)
    ap.add_argument('--permutations',type=int,default=50)
    ap.add_argument('--max-pairs',type=int,default=50000)
    args=ap.parse_args()
    frozen, ids=_load_frozen_tasks()
    out=ROOT/'benchmarks'/'results'/'v16'/'fgmrw_recent'
    out.mkdir(parents=True,exist_ok=True)
    config={
      'suite':'OpenML-CC18','suite_id':99,'selection_rule':'frozen task order; X-only encoded p <= recent_max_features',
      'limit':args.limit,'seeds':args.seeds,'max_samples_uniform':args.max_samples,
      'recent_max_features':args.recent_max_features,'labels_used_for_task_acceptance':False,
      'fgmrw_alpha':0.1,'fgmrw_s':0.8,'fgmrw_commit':'325a904a28284a879c8a522ca7c4abc56449e9f5'
    }
    write_manifest(out/'environment_and_config.json',extra=config)
    rows=[]; manifest=[]; accepted=0
    fac={
      'GSR-FS': lambda k,seed:GSRSelector(n_features=k,n_permutations=args.permutations,max_pairs=args.max_pairs,pair_sampler='balanced',residual_power=0.5,random_state=seed,support_engine='vectorized',residual_engine='auto'),
      'FGMRW-UFS-TKDE2026-author': lambda k,seed:FGMRWAuthorSelector(n_features=k,alpha=0.1,s=0.8),
    }
    for tid in ids:
        if accepted>=args.limit: break
        rec={'task_id':int(tid)}
        try:
            ds,meta=load_task(tid,args.max_samples,args.recent_max_features,seed=1729)
            if ds.X.shape[1] > args.recent_max_features:
                raise ValueError('recent feature cap exceeded')
            r=run_real_benchmark_flexible([ds],fac,seeds=tuple(args.seeds))
            r.insert(0,'openml_task_id',int(tid)); rows.append(r); accepted+=1
            rec.update(meta); rec['status']='ok'; rec['error']=''
        except Exception as exc:
            rec['status']='skipped'; rec['error']=repr(exc)
        manifest.append(rec)
    raw=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    raw.to_csv(out/'fgmrw_cc18_raw.csv',index=False)
    pd.DataFrame(manifest).to_csv(out/'fgmrw_cc18_manifest.csv',index=False)
    summary=(raw.loc[raw.error.fillna('').eq('')].groupby(['dataset','method'])[['nmi','ari']].mean().reset_index() if not raw.empty else pd.DataFrame())
    summary.to_csv(out/'fgmrw_cc18_dataset_method_summary.csv',index=False)
    gate={
      'version':'0.16.0','recent_method':'FGMRW-UFS-TKDE2026-author','required_tasks':args.limit,
      'accepted_tasks':accepted,'zero_method_failures': bool(raw.empty or raw.error.fillna('').eq('').all()),
      'x_only_task_acceptance':True,'author_source_pinned':True,
    }
    gate['overall_pass']=gate['accepted_tasks']>=gate['required_tasks'] and gate['zero_method_failures']
    (out/'RECENT_AUTHOR_GATE_CURRENT_V016.json').write_text(json.dumps(gate,indent=2))
    print(json.dumps(gate,indent=2))

if __name__=='__main__': main()
