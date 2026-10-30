import os, json, time, platform, warnings
from collections import Counter
import numpy as np, pandas as pd
import sklearn, imblearn, xgboost
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.pipeline import Pipeline            # pipeline yang menerima sampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
os.makedirs("output/prediksi", exist_ok=True)

# Data dari Bab 8: X_train/X_test asli (tanpa balancing), label multikelas dan biner
X_train = pd.read_csv("output/X_train.csv");  X_test = pd.read_csv("output/X_test.csv")
y_train = pd.read_csv("output/y_train.csv")["y"].to_numpy()
y_test = pd.read_csv("output/y_test.csv")["y"].to_numpy()
yb_train = pd.read_csv("output/y_train_biner.csv")["y_biner"].to_numpy()
yb_test = pd.read_csv("output/y_test_biner.csv")["y_biner"].to_numpy()
nama_kelas = pd.read_csv("output/pemetaan_label.csv")["label"].tolist()

kolom_num = X_train.select_dtypes(include="number").columns.tolist()
kolom_kat = [c for c in X_train.columns if c not in kolom_num]
print("Latih:", X_train.shape, "| Uji:", X_test.shape)
print("Fitur numerik:", len(kolom_num), "| kategorikal (tidak dipakai di bab ini):", kolom_kat)

def praproses():
    """Imputasi median dan scaling untuk fitur numerik. Dipasang di dalam pipeline,
    sehingga statistiknya hanya dipelajari dari data latih (atau lipatan latih)."""
    return ColumnTransformer(
        [("num", SkPipeline([("imputer", SimpleImputer(strategy="median")),
                             ("scaler", StandardScaler())]), kolom_num)],
        remainder="drop")                      # kolom kategorikal dibuang (subbab 10.1)
