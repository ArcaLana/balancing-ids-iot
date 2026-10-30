import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_norm = cm / cm.sum(axis=1, keepdims=True)                   # tiap baris dijumlahkan menjadi 1 (= recall pada diagonal)
fig, ax = plt.subplots(figsize=(9.5, 8))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
for i in labels:
    for j in labels:
        if cm[i, j] > 0:
            ax.text(j, i, f"{cm[i, j]:,}".replace(",", "."), ha="center", va="center", fontsize=7,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
ax.set_xticks(labels); ax.set_xticklabels(nama_kelas, rotation=60, ha="right", fontsize=7)
ax.set_yticks(labels); ax.set_yticklabels(nama_kelas, fontsize=7)
ax.set_xlabel("Kelas prediksi"); ax.set_ylabel("Kelas sebenarnya")
fig.colorbar(im, ax=ax, fraction=0.04, label="proporsi per baris")
plt.tight_layout(); plt.savefig("output/g_confusion_matrix.png", dpi=200); plt.show()
print("Sel di luar diagonal yang tidak nol:", int(((cm > 0) & ~np.eye(len(labels), dtype=bool)).sum()))
