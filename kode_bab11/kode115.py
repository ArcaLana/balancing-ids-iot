# Laporan per kelas: F1 tiap kelas untuk empat konfigurasi (heatmap) dan tabel presisi/recall/F1 untuk satu konfigurasi
from sklearn.metrics import precision_recall_fscore_support
konfig = [("RF", "Baseline"), ("XGB", "Baseline"), ("XGB", "Borderline-SMOTE"), ("XGB", "SMOTE-ENN")]
f1_kelas = {}
for m, s in konfig:
    yp = np.load(f"output/prediksi/multikelas_{m}_{s}.npz")["y_pred"]
    f1_kelas[f"{m} {s}"] = precision_recall_fscore_support(y_test, yp, labels=labels, zero_division=0)[2]
f1_df = pd.DataFrame(f1_kelas, index=nama_kelas)
f1_df["sampel_uji"] = np.bincount(y_test, minlength=len(labels))
f1_df.round(4).to_csv("output/f1_per_kelas.csv")
print(f1_df.round(3).to_string())

fig, ax = plt.subplots(figsize=(8, 4.8))
nilai = f1_df.drop(columns="sampel_uji").to_numpy()
im = ax.imshow(nilai, cmap="Blues", vmin=0.8, vmax=1.0, aspect="auto")
for i in range(nilai.shape[0]):
    for j in range(nilai.shape[1]):
        ax.text(j, i, f"{nilai[i, j]:.3f}".replace(".", ","), ha="center", va="center", fontsize=7,
                color="white" if nilai[i, j] > 0.94 else "black")
ax.set_xticks(range(nilai.shape[1])); ax.set_xticklabels(f1_df.columns[:-1], fontsize=7)
ax.set_yticks(range(len(nama_kelas))); ax.set_yticklabels([f"{n} ({c})" for n, c in zip(nama_kelas, f1_df['sampel_uji'])], fontsize=7)
fig.colorbar(im, ax=ax, fraction=0.04, label="F1 (skala 0,8 sampai 1,0)")
plt.tight_layout(); plt.savefig("output/g_f1_per_kelas.png", dpi=200); plt.show()
