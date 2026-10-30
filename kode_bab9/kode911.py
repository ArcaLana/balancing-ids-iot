# Dokumentasi: konfigurasi, versi, seed, dan waktu eksekusi disimpan sebagai JSON dan CSV
konfigurasi = {
    "versi": {lib.__name__: lib.__version__ for lib in (np, pd, sklearn, imblearn)},
    "random_state": RANDOM_STATE,
    "fitur": "numerik berskala (imputer median dan StandardScaler dipasang pada data latih)",
    "k_neighbors": int(K),
    "BATAS_rus": BATAS,
    "TARGET_oversampling": TARGET,
    "waktu_detik": {k: round(v, 3) for k, v in waktu.items()},
    "distribusi": hasil,
}
with open("output/konfigurasi_bab9.json", "w") as f:
    json.dump(konfigurasi, f, indent=2, ensure_ascii=False, default=int)

log = pd.DataFrame({"waktu_dtk": pd.Series(waktu).round(3),
                    "total_sampel": pd.DataFrame(hasil).drop(columns="Asli").sum().astype(int)})
log["random_state"] = RANDOM_STATE
log["k_neighbors"] = K
print(log)
log.to_csv("output/log_eksperimen_bab9.csv")
