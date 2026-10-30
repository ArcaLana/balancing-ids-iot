from imblearn.over_sampling import SMOTE

# k_neighbors harus lebih kecil dari jumlah sampel kelas terkecil (subbab 7.7)
jumlah_min = min(jumlah.values())
K = max(1, min(5, jumlah_min - 1))
print("Sampel kelas terkecil:", jumlah_min, "| k_neighbors dipakai:", K)

# a) Bawaan: semua kelas selain yang terbesar dinaikkan ke jumlah kelas terbesar
_ = jalankan("SMOTE bawaan", SMOTE(k_neighbors=K, random_state=RANDOM_STATE))

# b) Target parsial (TARGET dan target_ros sudah didefinisikan pada Kode 9.3)
_ = jalankan("SMOTE parsial", SMOTE(sampling_strategy=target_ros, k_neighbors=K,
                                    random_state=RANDOM_STATE))
gambar("SMOTE parsial")
