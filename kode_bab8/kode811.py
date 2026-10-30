# Simpan data hasil pembagian agar dapat dipakai ulang di Bab 9 sampai Bab 12
X_train.to_csv("output/X_train.csv", index=False)
X_test.to_csv("output/X_test.csv", index=False)
pd.Series(y_train, name="y").to_csv("output/y_train.csv", index=False)
pd.Series(y_test, name="y").to_csv("output/y_test.csv", index=False)
pd.Series(y_train_biner, name="y_biner").to_csv("output/y_train_biner.csv", index=False)
pd.Series(y_test_biner, name="y_biner").to_csv("output/y_test_biner.csv", index=False)

# Catat lingkungan dan keputusan agar eksperimen dapat diulang
with open("output/lingkungan.txt", "w") as f:
    f.write(f"Python {sys.version.split()[0]}\n")
    for lib in (np, pd, sklearn, imblearn, xgboost):
        f.write(f"{lib.__name__} {lib.__version__}\n")
    f.write(f"RANDOM_STATE = {RANDOM_STATE}\n")
    f.write(f"Kolom dibuang: {KOLOM_DIBUANG}\n")
    f.write(f"Kolom kategorikal: {kolom_kategorikal}\n")
    f.write(f"Baris setelah pembersihan: {len(df)}\n")

print("Berkas di folder output:", sorted(os.listdir("output")))
