import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("benchmark_trace_summary.csv")
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
for ax, (t, g) in zip(axs.ravel(), df.groupby("MatrixType", sort=False)):
    for meth, gg in g.groupby("Method", sort=False):
        ax.semilogy(gg.m, gg.MeanRelError, marker="o", label=meth)
    ax.set_title(t); ax.set_xlabel("Matrix-vector products m"); ax.set_ylabel("Mean relative error")
axs[0, 0].legend(); plt.tight_layout(); plt.show()