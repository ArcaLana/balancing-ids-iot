# Analisis kesalahan: (a) pasangan kelas yang paling sering tertukar (XGB baseline, multikelas)
#                     (b) alarm palsu dan serangan lolos pada skenario biner
asal, tujuan = np.where(~np.eye(len(labels), dtype=bool) & (cm > 0))
pasangan = sorted(zip(cm[asal, tujuan], asal, tujuan), reverse=True)[:8]
for n, i, j in pasangan:
    print(f"{n:3d} sampel {nama_kelas[i]} -> diprediksi {nama_kelas[j]}")

print("\nBiner (0 = normal, 1 = serangan); FP = alarm palsu, FN = serangan lolos")
baris = []
for m in ("RF", "XGB"):
    for s in ("Baseline", "RUS", "SMOTE", "SMOTE-ENN"):
        yp = np.load(f"output/prediksi/biner_{m}_{s}.npz")["y_pred"]
        tn, fp, fn, tp = confusion_matrix(yb_test, yp).ravel()
        baris.append({"model": m, "sampler": s, "FP": fp, "FN": fn, "FPR": fp / (fp + tn), "FNR": fn / (fn + tp)})
galat = pd.DataFrame(baris); galat.to_csv("output/galat_biner.csv", index=False)
print("Jumlah normal pada data uji:", int((yb_test == 0).sum()), "| serangan:", int((yb_test == 1).sum()))
print(galat.round(5).to_string(index=False))
