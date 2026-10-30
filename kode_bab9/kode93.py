from imblearn.over_sampling import RandomOverSampler

# a) Bawaan: semua kelas selain yang terbesar dinaikkan ke jumlah kelas terbesar
_ = jalankan("ROS bawaan", RandomOverSampler(random_state=RANDOM_STATE))

# b) Target parsial: tiap kelas dinaikkan paling tidak sampai TARGET sampel
TARGET = 2000                                  # CEK
target_ros = {k: max(v, TARGET) for k, v in jumlah.items()}
Zr, yr = jalankan("ROS parsial", RandomOverSampler(sampling_strategy=target_ros,
                                                   random_state=RANDOM_STATE))
# ROS menggandakan baris asli: jumlah baris unik tidak bertambah
print("Baris unik setelah ROS parsial:", len(np.unique(Zr, axis=0)),
      "| baris unik data latih asli:", len(np.unique(Z_train, axis=0)))
gambar("ROS parsial")
