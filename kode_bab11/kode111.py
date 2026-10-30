# Contoh ILUSTRASI (angka rekaan, bukan hasil penelitian): 3 kelas, baris = kelas sebenarnya, kolom = prediksi
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
nama3 = ["Normal", "Serangan A", "Serangan B"]
cm = np.array([[940, 40, 20],
               [ 10, 80, 10],
               [  5,  5, 40]])
y_true = np.repeat([0, 1, 2], cm.sum(axis=1))                  # susun label dari matriks agar fungsi sklearn bisa dipakai
y_pred = np.concatenate([np.repeat([0, 1, 2], cm[i]) for i in range(3)])
assert (confusion_matrix(y_true, y_pred) == cm).all()

TP = np.diag(cm); FP = cm.sum(axis=0) - TP; FN = cm.sum(axis=1) - TP; TN = cm.sum() - TP - FP - FN
presisi, recall, f1, dukungan = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
spesifisitas = TN / (TN + FP); fpr = FP / (FP + TN); fnr = FN / (FN + TP)
for i, n in enumerate(nama3):
    print(f"{n:11s} TP={TP[i]:4d} FP={FP[i]:3d} FN={FN[i]:3d} TN={TN[i]:4d} | presisi={presisi[i]:.4f} recall={recall[i]:.4f} "
          f"F1={f1[i]:.4f} spes={spesifisitas[i]:.4f} FPR={fpr[i]:.4f} FNR={fnr[i]:.4f}")
print("Akurasi    :", round(TP.sum() / cm.sum(), 4))
print("F1 macro   :", round(f1.mean(), 4), "| recall macro:", round(recall.mean(), 4))
print("G-Mean (3 kelas)       :", round(float(np.prod(recall) ** (1 / 3)), 4))
print("Weighted F1            :", round(float((f1 * dukungan).sum() / dukungan.sum()), 4))
