from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

recall_kelas = []          # recall per kelas tiap konfigurasi (dipakai Bab 11)

def gmean(y_true, y_pred, labels):
    """Rata-rata geometrik recall tiap kelas. Bernilai 0 bila ada kelas yang tidak terdeteksi."""
    r = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return float(np.prod(r) ** (1 / len(r)))

def evaluasi(skenario, model_nama, sampler_nama, pipa, X_tr, y_tr, X_te, y_te, fit_kwargs=None):
    """Latih pipeline pada data latih, nilai pada data uji ASLI; catat metrik dan waktu."""
    labels = np.unique(y_te)
    t0 = time.perf_counter(); pipa.fit(X_tr, y_tr, **(fit_kwargs or {}))
    waktu_fit = time.perf_counter() - t0                   # termasuk praproses dan sampling
    tp = []
    for _ in range(5):                                      # ulangi agar waktu prediksi stabil
        t0 = time.perf_counter(); y_pred = pipa.predict(X_te); tp.append(time.perf_counter() - t0)
    proba = pipa.predict_proba(X_te).astype("float32")
    np.savez_compressed(f"output/prediksi/{skenario}_{model_nama}_{sampler_nama}.npz".replace(" ", "_"),
                        y_pred=y_pred, proba=proba)
    for k, r in zip(labels, recall_score(y_te, y_pred, labels=labels, average=None, zero_division=0)):
        recall_kelas.append({"skenario": skenario, "model": model_nama, "sampler": sampler_nama,
                             "kelas": int(k), "recall": float(r)})
    return {"skenario": skenario, "model": model_nama, "sampler": sampler_nama, "status": "ok",
            "akurasi": accuracy_score(y_te, y_pred),
            "presisi_macro": precision_score(y_te, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_te, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_te, y_pred, average="macro", zero_division=0),
            "gmean": gmean(y_te, y_pred, labels),
            "waktu_fit_dtk": waktu_fit, "waktu_prediksi_dtk": float(np.median(tp))}
