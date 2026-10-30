from imblearn.over_sampling import BorderlineSMOTE, ADASYN

# Borderline-SMOTE (varian 1): sampel sintetis hanya dari sampel "bahaya"
bl = BorderlineSMOTE(kind="borderline-1", k_neighbors=K, m_neighbors=10,
                     sampling_strategy=target_ros, random_state=RANDOM_STATE)
_ = jalankan("Borderline-SMOTE", bl)
bahaya = {nama_kelas[k]: len(v) for k, v in bl.in_danger_indices.items()}
print("Sampel 'bahaya' per kelas yang dinaikkan:", bahaya)

# ADASYN: jumlah akhir tidak selalu tepat sama dengan target; dapat gagal
try:
    _ = jalankan("ADASYN", ADASYN(sampling_strategy=target_ros, n_neighbors=K,
                                  random_state=RANDOM_STATE))
except (RuntimeError, ValueError) as e:
    print("ADASYN gagal:", e)

for nama in ("SMOTE parsial", "Borderline-SMOTE", "ADASYN"):
    if nama in hasil:
        selisih = {k: hasil[nama][k] - target_ros[i]
                   for i, k in enumerate(nama_kelas) if i in target_ros}
        print(nama, "| selisih jumlah akhir terhadap target per kelas:", selisih)
gambar("Borderline-SMOTE")
