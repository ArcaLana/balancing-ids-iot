# Uji kepekaan: bagaimana bila kolom kategorikal (proto, service) ikut? Hanya Random Forest, multikelas.
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTENC

def praproses_kat():
    return ColumnTransformer([
        ("num", SkPipeline([("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler())]), kolom_num),
        ("kat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), kolom_kat)])

# Setelah ColumnTransformer, kolom kategorikal berada di posisi paling kanan
idx_kat = list(range(len(kolom_num), len(kolom_num) + len(kolom_kat)))
smote_kat = dict(sampling_strategy=target_ros, k_neighbors=K, random_state=RANDOM_STATE)

varian = {
    "numerik saja (Bab 10.7)": None,
    "kat. Baseline": [],
    "kat. SMOTE (kategori dianggap angka)": [("sampler", SMOTE(**smote_kat))],
    "kat. SMOTENC": [("sampler", SMOTENC(categorical_features=idx_kat, **smote_kat))],
}
hasil_kat = []
for nama, langkah in varian.items():
    if langkah is None:                                   # salin hasil numerik-saja dari tabel_multi
        r = tabel_multi[(tabel_multi.model == "RF") & (tabel_multi.sampler == "SMOTE")].iloc[0].to_dict()
        r["sampler"] = nama + " SMOTE"; hasil_kat.append(r); continue
    pipa = Pipeline([("praproses", praproses_kat())] + langkah +
                    [("model", model_rf())])
    hasil_kat.append(evaluasi("multikelas", "RF", nama, pipa, X_train, y_train, X_test, y_test))
    print(nama, "selesai", flush=True)
tabel_kat = pd.DataFrame(hasil_kat)
tabel_kat.to_csv("output/hasil_kepekaan_kategorikal.csv", index=False)
print(tabel_kat[["sampler", "f1_macro", "gmean", "waktu_fit_dtk"]].round(3).to_string(index=False))
