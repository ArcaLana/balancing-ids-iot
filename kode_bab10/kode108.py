import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

urutan = ["Baseline", "RUS", "ROS", "SMOTE", "Borderline-SMOTE", "ADASYN",
          "SMOTE-Tomek", "SMOTE-ENN", "class_weight", "sample_weight"]
warna_model = (("RF", "#2E9C93", -0.14), ("XGB", "#E07B39", 0.14))
y = np.arange(len(urutan))

# Grafik titik (bukan batang): sumbu dipotong agar selisih kecil terlihat; garis putus = skor Baseline
fig, ax = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
for a, (nama, t) in zip(ax, (("Klasifikasi biner", tabel_biner), ("Klasifikasi multikelas", tabel_multi))):
    for model, warna, dy in warna_model:
        d = t[(t["status"] == "ok") & (t["model"] == model)].set_index("sampler").reindex(urutan)
        a.scatter(d["f1_macro"], y + dy, color=warna, s=40, label=model, zorder=3)
        a.axvline(d.loc["Baseline", "f1_macro"], color=warna, ls="--", lw=0.9, alpha=0.7)
    a.set_title(nama); a.set_xlabel("F1 macro pada data uji (sumbu dipotong)")
    a.set_yticks(y); a.set_yticklabels(urutan); a.grid(axis="y", alpha=0.25)
ax[0].invert_yaxis()
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles[:2], labels[:2], loc="lower center", ncol=2, frameon=False)
plt.tight_layout(rect=(0, 0.06, 1, 1))
plt.savefig("output/g_f1_macro.png", dpi=200)
plt.show()
