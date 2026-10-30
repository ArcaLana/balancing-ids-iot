from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN

def konfigurasi(target_rus="auto", target_ros="auto", K=5):
    """Satu kamus nama -> sampler. Semua sampler memakai RANDOM_STATE yang sama."""
    r = RANDOM_STATE
    smote = lambda: SMOTE(sampling_strategy=target_ros, k_neighbors=K, random_state=r)
    return {
        "Baseline": None,
        "RUS": RandomUnderSampler(sampling_strategy=target_rus, random_state=r),
        "ROS": RandomOverSampler(sampling_strategy=target_ros, random_state=r),
        "SMOTE": smote(),
        "Borderline-SMOTE": BorderlineSMOTE(sampling_strategy=target_ros, k_neighbors=K, random_state=r),
        "ADASYN": ADASYN(sampling_strategy=target_ros, n_neighbors=K, random_state=r),
        "SMOTE-Tomek": SMOTETomek(smote=smote(), random_state=r),
        "SMOTE-ENN": SMOTEENN(smote=smote(), random_state=r),
    }

def jalankan_semua(skenario, y_tr, y_te, sampler_dict, hasil):
    """Seluruh kombinasi model x sampler; konfigurasi sama kecuali sampler."""
    for model_nama, buat in (("RF", model_rf), ("XGB", model_xgb)):
        for s_nama, sampler in sampler_dict.items():
            langkah = [("praproses", praproses())]
            if sampler is not None:
                langkah.append(("sampler", sampler))
            langkah.append(("model", buat()))
            try:
                hasil.append(evaluasi(skenario, model_nama, s_nama, Pipeline(langkah),
                                      X_train, y_tr, X_test, y_te))
            except Exception as e:                           # mis. ADASYN gagal pada data ini
                hasil.append({"skenario": skenario, "model": model_nama, "sampler": s_nama,
                              "status": f"gagal: {type(e).__name__}"})
                print("GAGAL", skenario, model_nama, s_nama, "->", str(e)[:90])
            print(skenario, model_nama, s_nama, "selesai", flush=True)
        # Pembanding tanpa sampling: pembobotan kelas
        if model_nama == "RF":
            hasil.append(evaluasi(skenario, "RF", "class_weight", Pipeline(
                [("praproses", praproses()), ("model", model_rf(class_weight="balanced"))]),
                X_train, y_tr, X_test, y_te))
        else:
            w = compute_sample_weight("balanced", y_tr)
            hasil.append(evaluasi(skenario, "XGB", "sample_weight", Pipeline(
                [("praproses", praproses()), ("model", model_xgb())]),
                X_train, y_tr, X_test, y_te, fit_kwargs={"model__sample_weight": w}))

# ---- Skenario BINER: normal (0) vs serangan (1); target "auto" menyamakan kedua kelas
hasil_biner = []
jalankan_semua("biner", yb_train, yb_test, konfigurasi(), hasil_biner)
tabel_biner = pd.DataFrame(hasil_biner)
tabel_biner.to_csv("output/hasil_biner.csv", index=False)
print(tabel_biner[["model", "sampler", "status", "f1_macro", "gmean", "waktu_fit_dtk"]].round(3).to_string(index=False))
