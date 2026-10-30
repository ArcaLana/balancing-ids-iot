from sklearn.metrics import roc_curve, precision_recall_curve, auc
pilih = ["DOS_SYN_Hping", "Wipro_bulb", "Metasploit_Brute_Force_SSH", "NMAP_FIN_SCAN"]
warna = ["#1f77b4", "#E07B39", "#2E9C93", "#9467bd"]
tabel_auc = []
fig_roc, a1 = plt.subplots(figsize=(5.2, 4.4)); fig_pr, a2 = plt.subplots(figsize=(5.2, 4.4))
for n, w in zip(pilih, warna):
    k = nama_kelas.index(n)
    fpr_, tpr_, _ = roc_curve(Y[:, k], proba[:, k]); pr_, rc_, _ = precision_recall_curve(Y[:, k], proba[:, k])
    a1.plot(fpr_, tpr_, color=w, label=n); a2.plot(rc_, pr_, color=w, label=n)
    tabel_auc.append({"kelas": n, "sampel_uji": int(Y[:, k].sum()), "ROC_AUC": auc(fpr_, tpr_),
                      "PR_AUC": average_precision_score(Y[:, k], proba[:, k])})
a1.plot([0, 1], [0, 1], "k:", lw=0.8); a1.set_xlabel("False Positive Rate"); a1.set_ylabel("True Positive Rate")
a2.set_xlabel("Recall"); a2.set_ylabel("Presisi")
for a in (a1, a2): a.legend(fontsize=7, loc="lower right" if a is a1 else "lower left"); a.grid(alpha=0.25)
fig_roc.tight_layout(); fig_roc.savefig("output/g_roc.png", dpi=200)
fig_pr.tight_layout(); fig_pr.savefig("output/g_pr.png", dpi=200)
tabel_auc = pd.DataFrame(tabel_auc); tabel_auc.to_csv("output/auc_kelas_terpilih.csv", index=False)
print(tabel_auc.round(4).to_string(index=False))
