import os, subprocess, sys
import pandas as pd

PS = [500, 1000, 2500, 5000, 10000]
REPS = 3

rows = []
for p in PS:
    for rep in range(REPS):
        subprocess.run([sys.executable, 'run_r2_highdim_single.py', str(p), str(rep)], check=True)
        f = f'r2_highdim_p{p}_rep{rep}.csv'
        rows.append(pd.read_csv(f).iloc[0].to_dict())

raw = pd.DataFrame(rows).sort_values(['p','rep'])
raw.to_csv('r2_highdim_scalability_raw.csv', index=False)

summary = raw.groupby('p', as_index=False).agg(
    n=('n','first'),
    B=('B','first'),
    max_pairs=('max_pairs','first'),
    reps=('rep','size'),
    median_seconds=('fit_seconds','median'),
    mean_seconds=('fit_seconds','mean'),
    sd_seconds=('fit_seconds','std'),
    median_peak_rss_mb=('peak_rss_mb','median'),
    max_peak_rss_mb=('peak_rss_mb','max'),
    median_incremental_mb=('incremental_peak_mb','median'),
    selected_k=('selected_k','mean'),
    completion_rate=('completed','mean')
)
summary.to_csv('r2_highdim_scalability_summary.csv', index=False)
print(summary.to_string(index=False))
