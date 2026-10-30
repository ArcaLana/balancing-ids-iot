# Uji konsistensi: ulangi pembagian data dan pelatihan dengan 10 seed (multikelas); simpan tiap hasil segera
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score
import csv, time

X_all = pd.concat([X_train, X_test], ignore_index=True)           # data bersih Bab 8 (setelah duplikat dibuang)
y_all = np.concatenate([y_train, y_test])
SEEDS = list(range(42, 52))
SAMPLER_RF = ["Baseline", "RUS", "ROS", "SMOTE", "Borderline-SMOTE"]
SAMPLER_XGB = ["Baseline", "RUS", "SMOTE", "Borderline-SMOTE"]     # hybrid dan ADASYN tidak diulang (lihat teks)
kelas_langka = [nama_kelas.index("NMAP_FIN_SCAN"), nama_kelas.index("Metasploit_Brute_Force_SSH")]

def buat_sampler(nama, jumlah, seed):
    batas = {k: min(v, 2000) for k, v in jumlah.items()}
    target = {k: max(v, 2000) for k, v in jumlah.items()}
    K = max(1, min(5, min(jumlah.values()) - 1))
    return {"Baseline": None,
            "RUS": RandomUnderSampler(sampling_strategy=batas, random_state=seed),
            "ROS": RandomOverSampler(sampling_strategy=target, random_state=seed),
            "SMOTE": SMOTE(sampling_strategy=target, k_neighbors=K, random_state=seed),
            "Borderline-SMOTE": BorderlineSMOTE(sampling_strategy=target, k_neighbors=K, random_state=seed)}[nama]

berkas = "output/hasil_seed.csv"
kolom_csv = ["seed", "model", "sampler", "f1_macro", "gmean", "recall_FIN", "recall_Metasploit", "waktu_fit_dtk"]
with open(berkas, "w", newline="") as f:
    csv.writer(f).writerow(kolom_csv)
for seed in SEEDS:
    RANDOM_STATE = seed                                              # dibaca oleh model_rf dan model_xgb
    Xa, Xb, ya, yb = train_test_split(X_all, y_all, test_size=0.30, stratify=y_all, random_state=seed)
    jumlah = Counter(ya)
    for model_nama, buat, daftar in (("RF", model_rf, SAMPLER_RF), ("XGB", model_xgb, SAMPLER_XGB)):
        for s_nama in daftar:
            s = buat_sampler(s_nama, jumlah, seed)
            langkah = [("praproses", praproses())] + ([("sampler", s)] if s is not None else []) + [("model", buat())]
            t0 = time.perf_counter(); pipa = Pipeline(langkah).fit(Xa, ya); wf = time.perf_counter() - t0
            yp = pipa.predict(Xb)
            r = recall_score(yb, yp, labels=kelas_langka, average=None, zero_division=0)
            with open(berkas, "a", newline="") as f:
                csv.writer(f).writerow([seed, model_nama, s_nama, f1_score(yb, yp, average="macro"),
                                        gmean(yb, yp, np.unique(yb)), r[0], r[1], wf])
    print("seed", seed, "selesai", flush=True)
hasil_seed = pd.read_csv(berkas)
print(hasil_seed.groupby(["model", "sampler"])[["f1_macro", "gmean"]].agg(["mean", "std"]).round(4).to_string())
