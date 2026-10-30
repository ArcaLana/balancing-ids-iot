# Simpan distribusi sebelum duplikat dibuang, lalu buang duplikat SEBELUM pembagian data
dist_sebelum = df[LABEL].value_counts()
df = df.drop_duplicates().reset_index(drop=True)
print("Bentuk setelah pembersihan:", df.shape)

# Distribusi kelas, persentase, dan imbalance ratio (IR) per kelas
dist = df[LABEL].value_counts()
tabel = pd.DataFrame({
    "jumlah": dist,
    "persen": (dist / len(df) * 100).round(3),
    "IR": (dist.max() / dist).round(2),
    "dibuang_duplikat": (dist_sebelum - dist).reindex(dist.index),
})
print(tabel)
print("IR keseluruhan (terbesar / terkecil):", round(dist.max() / dist.min(), 2))
tabel.to_csv("output/dist_kelas.csv")

# Grafik: skala linear dan logaritmik berdampingan
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
dist.plot.barh(ax=ax[0], color="#2E9C93");  ax[0].set_title("Skala linear")
dist.plot.barh(ax=ax[1], color="#E07B39");  ax[1].set_title("Skala logaritmik")
ax[1].set_xscale("log")
for a in ax:
    a.invert_yaxis()
    a.set_xlabel("Jumlah sampel")
plt.tight_layout()
plt.savefig("output/g_distribusi.png", dpi=200)
plt.show()
