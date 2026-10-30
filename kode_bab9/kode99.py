# Eksperimen target jumlah sampel: 25%, 50%, 100% dari kelas mayoritas
maj = max(jumlah.values())
BATAS_UKURAN = 1_000_000        # CEK: batas jumlah baris yang masih muat di memori dan waktu Anda
baris = []
for frac in (0.25, 0.50, 1.00):
    target = {k: max(v, int(frac * maj)) for k, v in jumlah.items()}
    ukuran = sum(target.values())
    if ukuran > BATAS_UKURAN:
        print(f"Target {frac:.0%} dilewati: {ukuran} baris melebihi BATAS_UKURAN")
        baris.append({"target": f"{frac:.0%} mayoritas", "total": ukuran, "dijalankan": False})
        continue
    t0 = time.perf_counter()
    Zr, yr = SMOTE(sampling_strategy=target, k_neighbors=K,
                   random_state=RANDOM_STATE).fit_resample(Z_train, y_train)
    baris.append({"target": f"{frac:.0%} mayoritas", "total": len(yr),
                  "faktor_ukuran": round(len(yr) / len(y_train), 2),
                  "terkecil": min(Counter(yr).values()),
                  "waktu_dtk": round(time.perf_counter() - t0, 2), "dijalankan": True})
tabel_target = pd.DataFrame(baris)
print(tabel_target)
tabel_target.to_csv("output/dampak_target.csv", index=False)

ax = tabel_target.plot.bar(x="target", y="total", legend=False, color="#E07B39", figsize=(6, 4))
ax.set_ylabel("Jumlah baris data latih setelah balancing")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("output/g_dampak_target.png", dpi=200)
plt.show()
