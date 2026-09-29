"""Finalize the v0.16 official OpenML-CC18 completeness/statistics gate."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results-dir', type=Path, required=True)
    ap.add_argument('--min-tasks', type=int, default=30)
    args=ap.parse_args()
    d=args.results_dir
    raw=pd.read_csv(d/'cc18_v15_raw.csv')
    man=pd.read_csv(d/'cc18_v15_dataset_manifest.csv')
    audit=json.loads((d/'source_integrity_audit.json').read_text())
    ok_tasks=set(man.loc[man.status.eq('ok'),'task_id'].astype(int))
    eligible=set(audit.get('eligible_datasets', []))
    # Source audit dataset names use task<id>_<name>; require one eligible record per ok task.
    eligible_task_ids=set()
    for name in eligible:
        if str(name).startswith('task'):
            try: eligible_task_ids.add(int(str(name).split('_',1)[0][4:]))
            except Exception: pass
    counted=ok_tasks & eligible_task_ids
    methods=sorted(raw.loc[raw.error.fillna('').eq(''),'method'].unique().tolist())
    canonical=[m for m in methods if 'scikit-feature' in m]
    failure_rows=int(raw.error.fillna('').ne('').sum())
    gate={
      'version':'0.16.0','suite':'OpenML-CC18','suite_id':99,
      'minimum_tasks_required':int(args.min_tasks),'accepted_ok_tasks':len(ok_tasks),
      'source_integrity_pass_tasks':len(counted),'methods':methods,
      'canonical_methods':canonical,'method_failure_rows':failure_rows,
      'checks':{
        'task_count':len(counted)>=args.min_tasks,
        'canonical_comparators':len(canonical)>=2,
        'zero_method_failures':failure_rows==0,
        'source_integrity':len(counted)==len(ok_tasks) and len(counted)>=args.min_tasks,
      }
    }
    gate['overall_pass']=all(gate['checks'].values())
    (d/'STANDARD_BENCHMARK_GATE_CURRENT_V016.json').write_text(json.dumps(gate,indent=2))

    clean=raw.loc[raw.error.fillna('').eq('')].copy()
    metric_rows=[]
    for metric in ['nmi','ari']:
        per=(clean.groupby(['dataset','method'])[metric].mean().unstack('method'))
        per=per.dropna(axis=0, how='any')
        if per.shape[0]>=2 and per.shape[1]>=2:
            ranks=per.apply(lambda r: pd.Series(rankdata(-r.to_numpy()),index=r.index),axis=1)
            mean_ranks=ranks.mean().sort_values()
            stat,p=friedmanchisquare(*[per[c].to_numpy() for c in per.columns])
            metric_rows.append({'metric':metric,'n_complete_datasets':len(per),'friedman_stat':float(stat),'friedman_p':float(p)})
            mean_ranks.rename('mean_rank').to_csv(d/f'{metric}_mean_ranks.csv')
            per.mean().sort_values(ascending=False).rename('mean').to_csv(d/f'{metric}_method_means.csv')
    pd.DataFrame(metric_rows).to_csv(d/'cc18_omnibus.csv',index=False)
    print(json.dumps(gate,indent=2))

if __name__=='__main__': main()
