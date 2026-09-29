from pathlib import Path
from gsrfs.benchmark import run_synthetic_benchmark, summarize

out = Path(__file__).resolve().parent / "results"
out.mkdir(exist_ok=True)

df = run_synthetic_benchmark(seeds=(0, 1, 2), gsr_permutations=20, max_pairs=12000)
df.to_csv(out / "synthetic_raw.csv", index=False)
summary = summarize(df)
summary.to_csv(out / "synthetic_summary.csv")
print(summary)
