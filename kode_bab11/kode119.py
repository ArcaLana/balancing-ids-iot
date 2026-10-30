# Konsistensi antarseed: rata-rata, simpangan baku, uji Wilcoxon berpasangan terhadap Baseline, dan uji Friedman
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, friedmanchisquare
hs = pd.read_csv("output/hasil_seed.csv")
ringkas = hs.groupby(["model", "sampler"])[["f1_macro", "gmean"]].agg(["mean", "std"]).round(4)
print(ringkas.to_string())

baris = []
for m in ("RF", "XGB"):
    w = hs[hs.model == m].pivot(index="seed", columns="sampler", values="f1_macro")
    for s in [c for c in w.columns if c != "Baseline"]:
        beda = w[s] - w["Baseline"]
        p = wilcoxon(w[s], w["Baseline"]).pvalue
        baris.append({"model": m, "sampler": s, "selisih_rata2": beda.mean(), "menang_dari_baseline": int((beda > 0).sum()),
                      "dari_seed": len(beda), "p_wilcoxon": p})
    fr = friedmanchisquare(*[w[c] for c in w.columns])
    print(m, "Friedman: statistik =", round(fr.statistic, 3), "| p =", round(fr.pvalue, 5))
uji = pd.DataFrame(baris); uji.to_csv("output/uji_statistik.csv", index=False)
print(uji.round(4).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for a, m in zip(ax, ("RF", "XGB")):
    w = hs[hs.model == m].pivot(index="seed", columns="sampler", values="f1_macro")
    a.boxplot([w[c] for c in w.columns], tick_labels=list(w.columns), showmeans=True)
    a.set_title(m); a.tick_params(axis="x", labelrotation=25, labelsize=8); a.grid(axis="y", alpha=0.25)
ax[0].set_ylabel("F1 macro (data uji tiap seed)")
plt.tight_layout(); plt.savefig("output/g_seed_boxplot.png", dpi=200); plt.show()
