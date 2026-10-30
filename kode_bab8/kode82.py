NAMA_FILE = "data/RT_IOT2022.csv"   # CEK: sesuaikan dengan nama berkas hasil unduhan Anda

df = pd.read_csv(NAMA_FILE)
print("Bentuk data (baris, kolom):", df.shape)
print(df.head().T)        # .T agar puluhan kolom terbaca ke bawah
df.info()
print(df.describe().T)
