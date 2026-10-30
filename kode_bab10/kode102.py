# Demonstrasi pada data sintetis kecil: sklearn Pipeline menolak sampler, imblearn Pipeline menerimanya
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

Xs, ys = make_classification(n_samples=500, weights=[0.9, 0.1], random_state=0)
try:
    SkPipeline([("sampler", SMOTE(random_state=0)), ("model", LogisticRegression())]).fit(Xs, ys)
except TypeError as e:
    print("sklearn Pipeline ->", type(e).__name__, ":", str(e).split("'")[0].strip())

pipa = Pipeline([("sampler", SMOTE(random_state=0)), ("model", LogisticRegression())])
pipa.fit(Xs, ys)                                   # sampler aktif saat fit
print("Label sebelum sampler:", {int(k): v for k, v in Counter(ys).items()}, "| setelah sampler (dilihat model):",
      {int(k): v for k, v in Counter(SMOTE(random_state=0).fit_resample(Xs, ys)[1]).items()})
print("predict pada 7 baris ->", pipa.predict(Xs[:7]).shape, "(sampler dilewati)")
