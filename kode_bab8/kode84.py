# a) Buang kolom bukan-fitur lebih dulu. Kolom indeks membuat setiap baris
#    tampak unik, sehingga duplikat tidak akan terdeteksi bila kolom itu dibiarkan.
KOLOM_DIBUANG = list(dict.fromkeys(unik_semua + kolom_indeks + konstan))   # CEK sebelum dijalankan
df = df.drop(columns=KOLOM_DIBUANG)
print("Kolom dibuang:", KOLOM_DIBUANG, "| bentuk sekarang:", df.shape)

# b) Nilai kosong
kosong = df.isna().sum()
print("Kolom dengan nilai kosong:")
print(kosong[kosong > 0])
print("Total nilai kosong:", int(kosong.sum()))

# c) Duplikasi
n_dup = int(df.duplicated().sum())
print(f"Baris duplikat: {n_dup} ({n_dup / len(df):.2%})")

# d) Duplikat yang berlabel berbeda (pola fitur sama, label berlainan)
fitur = [c for c in df.columns if c != LABEL]
label_per_pola = df.groupby(fitur, dropna=False)[LABEL].nunique()
print("Pola fitur dengan lebih dari satu label:", int((label_per_pola > 1).sum()))

# e) Tipe data
print(df.dtypes.value_counts())
kolom_kategorikal = [c for c in df.select_dtypes(include=["object", "category"]).columns
                     if c != LABEL]
print("Kolom kategorikal (selain label):", kolom_kategorikal)

# f) Nilai pengganti yang bukan NaN (mis. "-") tidak terdeteksi isna(); lihat nilai teratas
for c in kolom_kategorikal:
    print(c, df[c].value_counts().head(5).to_dict())
