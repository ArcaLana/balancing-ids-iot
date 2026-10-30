# ---- Skenario MULTIKELAS: target sama dengan Bab 9 (BATAS dan TARGET = 2000 per kelas)
jumlah = Counter(y_train)
BATAS = 2000; TARGET = 2000                                    # CEK: sama dengan Bab 9
K = max(1, min(5, min(jumlah.values()) - 1))
target_rus = {k: min(v, BATAS) for k, v in jumlah.items()}
target_ros = {k: max(v, TARGET) for k, v in jumlah.items()}

hasil_multi = []
jalankan_semua("multikelas", y_train, y_test,
               konfigurasi(target_rus, target_ros, K), hasil_multi)
tabel_multi = pd.DataFrame(hasil_multi)
tabel_multi.to_csv("output/hasil_multikelas.csv", index=False)
rk = pd.DataFrame(recall_kelas)
rk["kelas_nama"] = rk.apply(lambda r: nama_kelas[r["kelas"]] if r["skenario"] == "multikelas"
                            else ["normal", "serangan"][r["kelas"]], axis=1)
rk.to_csv("output/recall_per_kelas.csv", index=False)
print(tabel_multi[["model", "sampler", "status", "f1_macro", "gmean", "waktu_fit_dtk"]].round(3).to_string(index=False))
