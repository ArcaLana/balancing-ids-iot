from sklearn.preprocessing import LabelEncoder

X = df.drop(columns=[LABEL])
y_teks = df[LABEL]

# Label multikelas: teks -> angka 0..K-1 (diperlukan XGBoost). Simpan pemetaannya.
le = LabelEncoder()
y = le.fit_transform(y_teks)
pemetaan = pd.DataFrame({"kode": range(len(le.classes_)), "label": le.classes_})
pemetaan.to_csv("output/pemetaan_label.csv", index=False)
print(pemetaan)

# Label biner: normal (0) vs serangan (1), dibentuk dari label multikelas.
# Nama kelas normal belum pasti, jadi kelas serangan dikenali dari kata kunci.
KATA_KUNCI_SERANGAN = ("DOS", "DDOS", "ARP", "NMAP", "Metasploit", "Brute")   # CEK
serangan = [k for k in le.classes_ if any(s.lower() in k.lower() for s in KATA_KUNCI_SERANGAN)]
normal = [k for k in le.classes_ if k not in serangan]
print("Dianggap serangan:", serangan)     # PERIKSA: harus sesuai daftar serangan
print("Dianggap normal  :", normal)       # PERIKSA: harus lalu lintas normal
y_biner = y_teks.isin(serangan).astype(int).to_numpy()
print("Biner (0 = normal, 1 = serangan):", {int(k): int(v) for k, v in zip(*np.unique(y_biner, return_counts=True))})
