from pathlib import Path
import pandas as pd
from gsrfs.realdata import load_offline_real_panel
from gsrfs.downstream import run_foldwise_downstream_validation, summarize_foldwise_downstream

OUT = Path(__file__).resolve().parent / "results" / "v08"
OUT.mkdir(parents=True, exist_ok=True)

datasets = load_offline_real_panel(include_digits=True)
raw = run_foldwise_downstream_validation(
    datasets,
    seeds=(0,),
    n_splits=3,
    gsr_permutations=12,
    max_pairs=8000,
)
raw.to_csv(OUT / "offline_panel_downstream_raw.csv", index=False)
overall, extension = summarize_foldwise_downstream(raw)
overall.to_csv(OUT / "offline_panel_downstream_summary.csv")
extension.to_csv(OUT / "offline_extension_downstream_summary.csv")
print(overall)
print("\nOffline extension only:\n", extension)
