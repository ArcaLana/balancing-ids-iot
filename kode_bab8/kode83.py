LABEL = "Attack_type"     # CEK: nama kolom label menurut halaman dataset
assert LABEL in df.columns, "Kolom label tidak ditemukan; periksa df.columns"

print("Kolom label  :", LABEL)
print("Jumlah kelas :", df[LABEL].nunique())
print("Daftar label  :", sorted(df[LABEL].unique()))

# Kandidat kolom yang bukan fitur:
#   - nilainya unik di setiap baris (indeks atau identitas)
#   - nilainya konstan (tidak membedakan apa pun)
unik_semua = [c for c in df.columns if c != LABEL and df[c].nunique() == len(df)]
konstan = [c for c in df.columns if c != LABEL and df[c].nunique() <= 1]
print("Unik di setiap baris:", unik_semua)
print("Konstan              :", konstan)

# Kolom "Unnamed: n" muncul bila CSV disimpan bersama indeks pandas. Ia belum tentu unik di
# setiap baris (mis. indeks yang mulai dari 0 lagi di tiap kelas), jadi periksa per kelas.
kolom_indeks = [c for c in df.columns if str(c).startswith("Unnamed")]
for c in kolom_indeks:
    print(f"Kolom indeks '{c}': nilai per kelas")
    print(df.groupby(LABEL)[c].agg(["min", "max", "count", "nunique"]))
