from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

# SMOTE dahulu, lalu Tomek Links pada SEMUA kelas (bawaan SMOTETomek)
st = SMOTETomek(smote=SMOTE(sampling_strategy=target_ros, k_neighbors=K,
                            random_state=RANDOM_STATE),
                random_state=RANDOM_STATE)
_ = jalankan("SMOTE-Tomek", st)

# Sampel yang dibuang tahap Tomek = selisih terhadap SMOTE parsial (SMOTE-nya identik)
dibuang = {k: hasil["SMOTE parsial"][k] - hasil["SMOTE-Tomek"][k] for k in nama_kelas}
print("Sampel dibuang tahap Tomek per kelas:", dibuang)
print("Total dibuang:", sum(dibuang.values()))
gambar("SMOTE-Tomek")
