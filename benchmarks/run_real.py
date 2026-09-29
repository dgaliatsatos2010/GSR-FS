from pathlib import Path
from gsrfs.benchmark import run_real_benchmark, summarize_real

out = Path(__file__).resolve().parent / "results"
out.mkdir(exist_ok=True)

df = run_real_benchmark(seeds=(0, 1), gsr_permutations=20, max_pairs=10000)
df.to_csv(out / "real_builtin_raw.csv", index=False)
summary = summarize_real(df)
summary.to_csv(out / "real_builtin_summary.csv")
print(summary)
