from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
X_all = pd.concat([X_train, X_test], ignore_index=True); y_all = np.concatenate([y_train, y_test])
seed = 45; RANDOM_STATE = seed
Xa, Xb, ya, yb = train_test_split(X_all, y_all, test_size=0.30, stratify=y_all, random_state=seed)
jum = Counter(ya); tg = {k: max(v, 2000) for k, v in jum.items()}
res = {}
for nm, s in (("base", None), ("smote", SMOTE(sampling_strategy=tg, k_neighbors=5, random_state=seed))):
    pipa = Pipeline([("praproses", praproses())] + ([("s", s)] if s else []) + [("model", model_rf())]).fit(Xa, ya)
    res[nm] = pipa.predict(Xb)
for nama in ("NMAP_FIN_SCAN", "Metasploit_Brute_Force_SSH"):
    k = nama_kelas.index(nama); idx = np.where(yb == k)[0]
    for nm in res:
        salah = [(int(i), nama_kelas[res[nm][i]]) for i in idx if res[nm][i] != k]
        print(nama, nm, "salah:", salah)
