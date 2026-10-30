# Dokumentasi eksperimen: konfigurasi, versi, spesifikasi mesin, dan log gabungan
konfigurasi_eksp = {
    "versi": {lib.__name__: lib.__version__ for lib in (np, pd, sklearn, imblearn, xgboost)},
    "random_state": RANDOM_STATE,
    "mesin": {"cpu_inti": os.cpu_count(), "platform": platform.platform()},
    "fitur": f"{len(kolom_num)} numerik berskala; kategorikal tidak dipakai",
    "model": {"RF": model_rf().get_params(), "XGB": {k: v for k, v in model_xgb().get_params().items()
                                                       if k in ("n_estimators", "max_depth", "learning_rate", "tree_method", "random_state")}},
    "sampler_multikelas": {"BATAS": BATAS, "TARGET": TARGET, "k_neighbors": int(K)},
    "sampler_biner": "sampling_strategy=auto (kedua kelas disamakan)",
    "evaluasi": "latih pada X_train; nilai pada X_test asli; waktu prediksi = median 5 kali",
}
with open("output/konfigurasi_bab10.json", "w") as f:
    json.dump(konfigurasi_eksp, f, indent=2, ensure_ascii=False, default=str)

log = pd.concat([tabel_biner, tabel_multi, tabel_kat]).reset_index(drop=True)
log.to_csv("output/log_eksperimen_bab10.csv", index=False)
print("Konfigurasi dan log tersimpan; jumlah baris log:", len(log))
print("Konfigurasi gagal:", log[log["status"] != "ok"][["skenario", "model", "sampler", "status"]].values.tolist())
