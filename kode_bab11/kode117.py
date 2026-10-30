# Sumber daya model baseline multikelas: ukuran model, memori proses, dan latensi prediksi satu baris
import psutil, pickle, gc
proses = psutil.Process()
hasil_sd = []
for nama, buat in (("RF", model_rf), ("XGB", model_xgb)):
    gc.collect(); rss0 = proses.memory_info().rss
    pipa = Pipeline([("praproses", praproses()), ("model", buat())]).fit(X_train, y_train)
    rss1 = proses.memory_info().rss
    ukuran_mb = len(pickle.dumps(pipa)) / 1e6                  # ukuran model terserialisasi
    satu = X_test.iloc[[0]]; lat = []
    for i in range(200):                                       # 200 prediksi satu baris
        t0 = time.perf_counter(); pipa.predict(X_test.iloc[[i]]); lat.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter(); pipa.predict(X_test); batch = (time.perf_counter() - t0) / len(X_test) * 1000
    hasil_sd.append({"model": nama, "ukuran_pickle_MB": ukuran_mb, "tambahan_RSS_MB": (rss1 - rss0) / 1e6,
                     "latensi_1_baris_ms_median": float(np.median(lat)), "latensi_per_baris_batch_ms": batch})
sd = pd.DataFrame(hasil_sd); sd.to_csv("output/sumber_daya.csv", index=False)
print(sd.round(3).to_string(index=False))
