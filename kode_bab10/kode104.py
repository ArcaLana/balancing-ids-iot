from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

def model_xgb():
    return XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                         tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE)

# Fungsi tujuan dipilih otomatis dari jumlah kelas pada label (diuji pada data sintetis)
for k in (2, 12):
    Xd, yd = make_classification(n_samples=1500, n_informative=8, n_classes=k, random_state=0)
    m = model_xgb(); m.set_params(n_estimators=5).fit(Xd, yd)
    print(f"{k} kelas -> objective =", m.get_xgb_params()["objective"])

# Pembanding tanpa sampling: bobot sampel seimbang (XGBoost) dan class_weight (Random Forest)
bobot = compute_sample_weight("balanced", y_train)
print("Bobot sampel multikelas: terkecil", bobot.min().round(3), "| terbesar", bobot.max().round(1))
