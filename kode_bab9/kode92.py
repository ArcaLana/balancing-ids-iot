from imblearn.under_sampling import RandomUnderSampler

# a) Pengaturan bawaan: semua kelas selain yang terkecil diturunkan ke jumlah kelas terkecil
_ = jalankan("RUS bawaan", RandomUnderSampler(random_state=RANDOM_STATE))

# b) Target parsial: tiap kelas dibatasi paling banyak BATAS sampel
BATAS = 2000                                   # CEK: sesuaikan dengan distribusi Anda
target_rus = {k: min(v, BATAS) for k, v in jumlah.items()}
_ = jalankan("RUS parsial", RandomUnderSampler(sampling_strategy=target_rus,
                                               random_state=RANDOM_STATE))
print(hasil["RUS bawaan"])
print(hasil["RUS parsial"])
gambar("RUS parsial")
