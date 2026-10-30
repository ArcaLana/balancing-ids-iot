from sklearn.model_selection import train_test_split

# Periksa dulu: stratifikasi memerlukan minimal 2 sampel per kelas
jumlah_min = pd.Series(y).value_counts().min()
assert jumlah_min >= 2, "Ada kelas dengan kurang dari 2 sampel; lihat pembahasan kelas langka"

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
# Label biner dibagi dengan indeks yang sama agar konsisten dengan label multikelas
y_train_biner = y_biner[X_train.index]
y_test_biner = y_biner[X_test.index]

# Verifikasi proporsi tiap kelas pada data latih dan data uji
cek = pd.DataFrame({
    "latih": pd.Series(y_train).value_counts().sort_index(),
    "uji": pd.Series(y_test).value_counts().sort_index(),
})
cek["persen_latih"] = (cek["latih"] / cek["latih"].sum() * 100).round(3)
cek["persen_uji"] = (cek["uji"] / cek["uji"].sum() * 100).round(3)
cek.index = le.classes_[cek.index]
print(cek)
cek.to_csv("output/proporsi_latih_uji.csv")
print("Sampel kelas terkecil pada data latih:", int(cek["latih"].min()))
