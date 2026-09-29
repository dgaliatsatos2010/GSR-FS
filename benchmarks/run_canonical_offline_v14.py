from pathlib import Path
import json, os, sys, time
import pandas as pd

from gsrfs.realdata import load_offline_real_panel
from gsrfs.downstream import (
    run_foldwise_downstream_validation,
    canonical_offline_method_factories,
    summarize_foldwise_downstream,
)

OUT = Path(__file__).resolve().parent / 'results' / 'v14'
OUT.mkdir(parents=True, exist_ok=True)

datasets = [d for d in load_offline_real_panel(include_digits=True) if d.name != 'digits']
methods = canonical_offline_method_factories(gsr_permutations=12, max_pairs=8000)
raw = run_foldwise_downstream_validation(
    datasets,
    method_factories=methods,
    seeds=(0,),
    n_splits=3,
)
raw.to_csv(OUT / 'canonical_offline_foldwise_raw.csv', index=False)
overall, extension = summarize_foldwise_downstream(raw)
overall.to_csv(OUT / 'canonical_offline_foldwise_summary.csv')
extension.to_csv(OUT / 'canonical_offline_extension_summary.csv')

prov = {
    'benchmark': 'v0.14 canonical scikit-feature fold-wise offline panel',
    'datasets': [d.name for d in datasets],
    'excluded_from_primary_run': {
        'digits': 'canonical SPEC/MCFS use dense n-by-n spectral decompositions; handled as separate scalability stress case'
    },
    'scikit_feature_snapshot_env': os.environ.get('SCIKIT_FEATURE_SNAPSHOT', ''),
    'selectors_receive_y': False,
    'mcfs_n_clusters': 5,
    'laplacian_workaround': 'upstream construct_W(X) passed explicitly as W because upstream lap_score default call raises KeyError(W)',
}
(OUT / 'canonical_offline_provenance.json').write_text(json.dumps(prov, indent=2))
print(overall)
print('\nErrors by method:')
print(raw.assign(failed=raw.error.fillna('').str.len().gt(0)).groupby('method').failed.mean())
