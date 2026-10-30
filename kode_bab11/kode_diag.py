Xa_ = pd.concat([X_train, X_test], ignore_index=True)[kolom_num]; ya_ = np.concatenate([y_train, y_test])
h = pd.util.hash_pandas_object(Xa_, index=False)
df_ = pd.DataFrame({"h": h, "y": ya_})
multi = df_.groupby("h")["y"].nunique(); multi = multi[multi > 1].index
for nama in ("NMAP_FIN_SCAN", "Metasploit_Brute_Force_SSH"):
    k = nama_kelas.index(nama); sub = df_[df_.y == k]
    print(nama, "jumlah:", len(sub), "| baris dengan fitur identik pada kelas lain:", int(sub.h.isin(multi).sum()),
          "| baris unik (fitur):", sub.h.nunique())
print("pola fitur dengan lebih dari satu label:", len(multi))
print("kelas yang terlibat:", sorted({nama_kelas[k] for k in df_[df_.h.isin(multi)].y.unique()}))
