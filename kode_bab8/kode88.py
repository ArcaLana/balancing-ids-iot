# Ilustrasi kebocoran data pada DATA SINTETIS (bukan RT-IoT2022)
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
X, y = make_classification(n_samples=4000, n_features=10, n_informative=4,
    n_redundant=2, weights=[0.95, 0.05], class_sep=0.5, flip_y=0.02,
    random_state=RANDOM_STATE)

def skor(model, X_uji, y_uji):
    p = model.predict(X_uji)
    return recall_score(y_uji, p), f1_score(y_uji, p)

# SALAH: SMOTE pada seluruh data, baru dibagi
X_s, y_s = SMOTE(random_state=RANDOM_STATE).fit_resample(X, y)
Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(
    X_s, y_s, test_size=0.30, stratify=y_s, random_state=RANDOM_STATE)
m_salah = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
m_salah.fit(Xa_tr, ya_tr)
r1, f1 = skor(m_salah, Xa_te, ya_te)

# BENAR: dibagi dulu, SMOTE hanya pada data latih, uji tetap asli
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
Xb_bal, yb_bal = SMOTE(random_state=RANDOM_STATE).fit_resample(Xb_tr, yb_tr)
m_benar = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
m_benar.fit(Xb_bal, yb_bal)
r2, f2 = skor(m_benar, Xb_te, yb_te)

print(f"Data uji SALAH : n={len(ya_te)}, minoritas={int(ya_te.sum())} ({ya_te.mean():.0%})")
print(f"Data uji BENAR : n={len(yb_te)}, minoritas={int(yb_te.sum())} ({yb_te.mean():.1%})")
print(f"SALAH (SMOTE sebelum split): recall minoritas={r1:.3f}  F1 minoritas={f1:.3f}")
print(f"BENAR (split dulu)          : recall minoritas={r2:.3f}  F1 minoritas={f2:.3f}")
