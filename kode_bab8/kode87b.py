# Ilustrasi: kelas dengan 1 sampel tidak dapat distratifikasi (DATA SINTETIS)
import numpy as np
from sklearn.model_selection import train_test_split

y_kecil = np.array([0] * 98 + [1] * 1 + [2] * 1)
X_kecil = np.arange(100).reshape(-1, 1)
try:
    train_test_split(X_kecil, y_kecil, test_size=0.3, stratify=y_kecil, random_state=42)
except ValueError as e:
    print("ValueError:", str(e)[:140])
