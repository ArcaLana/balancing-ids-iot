# Uji kepekaan kedua: apakah nomor port membantu model "menghafal sesi"? Multikelas, tanpa sampler.
port = ["id.orig_p", "id.resp_p"]
varian_port = {"port semua fitur": kolom_num,
               "port tanpa orig_p": [c for c in kolom_num if c != "id.orig_p"],
               "port tanpa kedua": [c for c in kolom_num if c not in port]}
hasil_port = []
for nama, kolom in varian_port.items():
    pra = ColumnTransformer([("num", SkPipeline([("imputer", SimpleImputer(strategy="median")),
                                                 ("scaler", StandardScaler())]), kolom)], remainder="drop")
    for model_nama, buat in (("RF", model_rf), ("XGB", model_xgb)):
        hasil_port.append(evaluasi("multikelas", model_nama, nama,
                                   Pipeline([("praproses", pra), ("model", buat())]),
                                   X_train, y_train, X_test, y_test))
        print(nama, model_nama, "selesai", flush=True)
tabel_port = pd.DataFrame(hasil_port)
tabel_port["jumlah_fitur"] = tabel_port["sampler"].map({n: len(k) for n, k in varian_port.items()})
tabel_port.to_csv("output/hasil_kepekaan_port.csv", index=False)
print(tabel_port[["model", "sampler", "jumlah_fitur", "f1_macro", "gmean"]].round(4).to_string(index=False))
