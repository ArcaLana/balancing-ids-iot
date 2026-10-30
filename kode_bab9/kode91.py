import json, time
from collections import Counter
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import sklearn, imblearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42

# Muat hasil Bab 8. Di bab ini HANYA data latih yang dipakai.
X_train = pd.read_csv("output/X_train.csv")
y_train = pd.read_csv("output/y_train.csv")["y"].to_numpy()
nama_kelas = pd.read_csv("output/pemetaan_label.csv")["label"].tolist()

# Sampler berbasis jarak memerlukan fitur numerik berskala (keputusan bab ini; lihat teks)
kolom_num = X_train.select_dtypes(include="number").columns.tolist()
praproses = Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())])
Z_train = praproses.fit_transform(X_train[kolom_num])    # dipasang pada data latih saja
print("Fitur numerik:", len(kolom_num), "| bentuk Z_train:", Z_train.shape)

def distribusi(y):
    c = Counter(y)
    return {nama_kelas[k]: c[k] for k in sorted(c)}

hasil, waktu = {"Asli": distribusi(y_train)}, {}

def jalankan(nama, sampler):
    """Jalankan sampler HANYA pada data latih; catat distribusi dan waktu."""
    t0 = time.perf_counter()
    Zr, yr = sampler.fit_resample(Z_train, y_train)
    waktu[nama] = time.perf_counter() - t0
    hasil[nama] = distribusi(yr)
    print(f"{nama}: total {len(yr)} sampel (asli {len(y_train)}), {waktu[nama]:.2f} dtk")
    return Zr, yr

def gambar(nama):
    d = pd.DataFrame({"Asli": hasil["Asli"], nama: hasil[nama]})
    ax = d.plot.barh(figsize=(9, 5), logx=True, color=["#9CA3AF", "#2E9C93"])
    ax.invert_yaxis()
    ax.set_xlabel("Jumlah sampel (skala logaritmik)")
    plt.tight_layout()
    plt.savefig(f"output/g_{nama.replace(' ', '_')}.png", dpi=200)
    plt.show()

jumlah = Counter(y_train)
print("Distribusi latih asli:", hasil["Asli"])
