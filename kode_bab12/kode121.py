# Tabel ringkasan studi kasus: metrik multikelas (termasuk ROC-AUC), FPR/FNR biner, ukuran data latih, waktu
from sklearn.metrics import roc_auc_score, confusion_matrix
ukuran_latih = {"Baseline": 82545, "RUS": 13888, "ROS": 92657, "SMOTE": 92657, "Borderline-SMOTE": 89484,
                "SMOTE-Tomek": 92597, "SMOTE-ENN": 92101, "class_weight": 82545, "sample_weight": 82545}   # dari Bab 9
multi = pd.read_csv("output/hasil_multikelas.csv"); multi = multi[multi["status"] == "ok"]
biner = pd.read_csv("output/hasil_biner.csv"); biner = biner[biner["status"] == "ok"].set_index(["model", "sampler"])
baris = []
for r in multi.itertuples():
    nama = f"output/prediksi/multikelas_{r.model}_{r.sampler}.npz".replace(" ", "_")
    proba = np.load(nama)["proba"].astype("float64"); proba /= proba.sum(axis=1, keepdims=True)
    auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
    yp = np.load(f"output/prediksi/biner_{r.model}_{r.sampler}.npz")["y_pred"]
    tn, fp, fn, tp = confusion_matrix(yb_test, yp).ravel()
    baris.append({"model": r.model, "sampler": r.sampler, "f1_macro": r.f1_macro, "gmean": r.gmean, "roc_auc_macro": auc,
                  "FPR_biner": fp / (fp + tn), "FNR_biner": fn / (fn + tp),
                  "ukuran_latih_multi": ukuran_latih[r.sampler], "fit_multi_dtk": r.waktu_fit_dtk,
                  "fit_biner_dtk": biner.loc[(r.model, r.sampler), "waktu_fit_dtk"]})
studi = pd.DataFrame(baris); studi.to_csv("output/ringkasan_studi_kasus.csv", index=False)
print(studi[["model", "sampler", "f1_macro", "gmean", "roc_auc_macro", "FPR_biner", "FNR_biner", "fit_multi_dtk"]].round(4).to_string(index=False))
