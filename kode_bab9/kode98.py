# Tabel distribusi semua metode (baris = kelas, kolom = metode)
tabel = pd.DataFrame(hasil).fillna(0).astype(int)
print(tabel)
tabel.to_csv("output/dist_sebelum_sesudah.csv")

# Ringkasan per metode: total, kelas terbesar dan terkecil, IR, waktu
ringkas = pd.DataFrame({
    "total": tabel.sum(),
    "terbesar": tabel.max(),
    "terkecil": tabel.min(),
})
ringkas["IR"] = (ringkas["terbesar"] / ringkas["terkecil"]).round(2)
ringkas["faktor_ukuran"] = (ringkas["total"] / ringkas.loc["Asli", "total"]).round(2)
ringkas["waktu_dtk"] = pd.Series(waktu).round(3)
print(ringkas)
ringkas.to_csv("output/ringkasan_metode.csv")

# Grafik gabungan: peta panas jumlah per kelas (baris) untuk tiap metode (kolom).
# Warna memakai log10 agar kelas besar dan kecil terbaca bersama; angka asli ditulis di sel.
nilai = tabel.to_numpy()
fig, ax = plt.subplots(figsize=(8, 5.2))
im = ax.imshow(np.log10(nilai + 1), cmap="YlGnBu", aspect="auto")
ax.set_xticks(range(tabel.shape[1]))
ax.set_xticklabels(tabel.columns, rotation=40, ha="right", fontsize=7)
ax.set_yticks(range(tabel.shape[0]))
ax.set_yticklabels(tabel.index, fontsize=7)
for i in range(nilai.shape[0]):
    for j in range(nilai.shape[1]):
        warna = "white" if np.log10(nilai[i, j] + 1) > 3.3 else "black"
        ax.text(j, i, f"{nilai[i, j]:,}".replace(",", "."), ha="center", va="center",
                fontsize=5.5, color=warna)
fig.colorbar(im, ax=ax, label="log10 (jumlah sampel + 1)", fraction=0.04, pad=0.02)
plt.tight_layout()
plt.savefig("output/g_perbandingan_metode.png", dpi=200)
plt.show()
