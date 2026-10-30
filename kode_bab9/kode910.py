import os
os.makedirs("output/hasil_balancing", exist_ok=True)

# Simpan satu hasil sebagai array terkompresi. Konvensi nama: metode_target_seed.npz
Zr, yr = SMOTE(sampling_strategy=target_ros, k_neighbors=K,
               random_state=RANDOM_STATE).fit_resample(Z_train, y_train)
nama_file = f"output/hasil_balancing/smote_target{TARGET}_seed{RANDOM_STATE}.npz"
np.savez_compressed(nama_file, Z=Zr, y=yr, kolom=np.array(kolom_num))
print("Tersimpan:", nama_file, "| ukuran:", round(os.path.getsize(nama_file) / 1e6, 2), "MB")

# Cek bahwa berkas dapat dibaca kembali
muat = np.load(nama_file)
print("Dibaca ulang:", muat["Z"].shape, muat["y"].shape)
