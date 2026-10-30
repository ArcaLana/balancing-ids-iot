from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

kolom_numerik = [c for c in X_train.columns if c not in kolom_kategorikal]

# Semua langkah yang "belajar dari data" berada di dalam satu pipeline
praproses = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), kolom_numerik),
    ("kat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
     kolom_kategorikal),
])

pipa = Pipeline([
    ("praproses", praproses),
    ("sampler", RandomUnderSampler(random_state=RANDOM_STATE)),   # contoh; diganti di Bab 9
    ("model", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
])

# Validasi silang pada DATA LATIH: sampler hanya aktif pada lipatan latih
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
skor_cv = cross_val_score(pipa, X_train, y_train, cv=cv, scoring="f1_macro")
print("F1-macro per lipatan:", skor_cv.round(3), "| rata-rata:", skor_cv.mean().round(3))
