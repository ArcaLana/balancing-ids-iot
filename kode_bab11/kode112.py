# Metrik lengkap untuk satu konfigurasi (XGB tanpa sampler, multikelas), dari prediksi yang tersimpan di Bab 10
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score,
                             average_precision_score)
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score
from sklearn.preprocessing import label_binarize

d = np.load("output/prediksi/multikelas_XGB_Baseline.npz")
y_pred, proba = d["y_pred"], d["proba"].astype("float64")
proba = proba / proba.sum(axis=1, keepdims=True)               # float32 -> jumlah baris tepat 1
labels = np.arange(len(nama_kelas))

print(classification_report(y_test, y_pred, labels=labels, target_names=nama_kelas, digits=4, zero_division=0))
print("G-Mean (imblearn)      :", round(geometric_mean_score(y_test, y_pred, average="multiclass"), 4))
print("ROC-AUC OvR macro      :", round(roc_auc_score(y_test, proba, multi_class="ovr", average="macro"), 4))
Y = label_binarize(y_test, classes=labels)
print("PR-AUC (AP) macro      :", round(float(np.mean([average_precision_score(Y[:, k], proba[:, k]) for k in labels])), 4))
