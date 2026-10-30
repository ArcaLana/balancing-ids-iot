# Mengapa F1 macro berubah? Hitung prediksi dan prediksi benar untuk tiga kelas terkecil (multikelas)
langka = ["Metasploit_Brute_Force_SSH", "NMAP_FIN_SCAN", "Wipro_bulb"]
baris = []
for model in ("RF", "XGB"):
    for sampler in ("Baseline", "RUS", "SMOTE", "Borderline-SMOTE", "SMOTE-ENN"):
        yp = np.load(f"output/prediksi/multikelas_{model}_{sampler}.npz")["y_pred"]
        for nama in langka:
            k = nama_kelas.index(nama)
            baris.append({"model": model, "sampler": sampler, "kelas": nama,
                          "sampel_uji": int((y_test == k).sum()),
                          "diprediksi": int((yp == k).sum()),
                          "benar": int(((yp == k) & (y_test == k)).sum())})
langka_df = pd.DataFrame(baris)
langka_df["presisi"] = (langka_df["benar"] / langka_df["diprediksi"]).round(2)
langka_df["recall"] = (langka_df["benar"] / langka_df["sampel_uji"]).round(2)
langka_df.to_csv("output/kelas_langka_prediksi.csv", index=False)

langka_df["benar/diprediksi"] = langka_df["benar"].astype(str) + "/" + langka_df["diprediksi"].astype(str)
for model in ("RF", "XGB"):
    p = langka_df[langka_df["model"] == model].pivot(index="kelas", columns="sampler", values="benar/diprediksi")
    print(model, "(benar/diprediksi; sampel uji: Metasploit 11, FIN 8, Wipro 66)")
    print(p[["Baseline", "RUS", "SMOTE", "Borderline-SMOTE", "SMOTE-ENN"]].to_string())
