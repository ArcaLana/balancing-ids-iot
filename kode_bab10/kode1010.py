# Waktu latih (fit; termasuk praproses dan sampling) dan waktu prediksi seluruh data uji
print("CPU:", os.cpu_count(), "inti |", platform.platform())
gabung = pd.concat([tabel_biner, tabel_multi])
gabung = gabung[gabung["status"] == "ok"].copy()
gabung["ms_per_1000_baris"] = gabung["waktu_prediksi_dtk"] / len(X_test) * 1000 * 1000
waktu = gabung[["skenario", "model", "sampler", "waktu_fit_dtk", "waktu_prediksi_dtk", "ms_per_1000_baris"]]
waktu.to_csv("output/waktu.csv", index=False)
print(waktu.round(3).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
d = gabung[gabung["skenario"] == "multikelas"]
for a, kolom, judul, tick in ((ax[0], "waktu_fit_dtk", "Waktu latih (detik, skala log)", [1, 2, 5, 10, 20, 50]),
                              (ax[1], "ms_per_1000_baris", "Waktu prediksi (ms per 1.000 baris, skala log)", [3, 4, 5, 6, 8, 10])):
    for model, warna, dy in warna_model:
        s = d[d["model"] == model].set_index("sampler").reindex(urutan)
        a.scatter(s[kolom], y + dy, color=warna, s=40, label=model, zorder=3)
    a.set_xscale("log"); a.set_xticks(tick); a.xaxis.set_major_formatter(ScalarFormatter())
    a.minorticks_off(); a.set_title(judul, fontsize=10); a.grid(axis="y", alpha=0.25)
ax[0].set_yticks(y); ax[0].set_yticklabels(urutan); ax[0].invert_yaxis()
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles[:2], labels[:2], loc="lower center", ncol=2, frameon=False)
plt.tight_layout(rect=(0, 0.06, 1, 1))
plt.savefig("output/g_waktu.png", dpi=200)
plt.show()
