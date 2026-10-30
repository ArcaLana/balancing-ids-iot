# Memeriksa nilai bawaan parameter pada imbalanced-learn yang terpasang (bahan Tabel 9.3)
import inspect, imblearn
from collections import Counter
import numpy as np
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN, SMOTENC
from imblearn.combine import SMOTETomek, SMOTEENN
print("imbalanced-learn", imblearn.__version__)
for c, ps in ((RandomUnderSampler,("sampling_strategy","replacement")),(RandomOverSampler,("sampling_strategy","shrinkage")),
              (SMOTE,("sampling_strategy","k_neighbors")),(BorderlineSMOTE,("sampling_strategy","k_neighbors","m_neighbors","kind")),
              (ADASYN,("sampling_strategy","n_neighbors")),(EditedNearestNeighbours,("sampling_strategy","n_neighbors","kind_sel")),
              (SMOTETomek,("sampling_strategy",)),(SMOTEENN,("sampling_strategy",))):
    sig=inspect.signature(c.__init__).parameters
    print(c.__name__, {p: sig[p].default for p in ps})
X,y=make_classification(n_samples=3000,n_features=6,n_informative=4,n_classes=3,weights=[.8,.15,.05],random_state=1)
print("asli",dict(sorted(Counter(y).items())))
for n,s in (("RUS",RandomUnderSampler(random_state=0)),("ROS",RandomOverSampler(random_state=0)),("SMOTE",SMOTE(random_state=0)),
            ("Borderline",BorderlineSMOTE(random_state=0)),("ADASYN",ADASYN(random_state=0))):
    print(n,dict(sorted(Counter(s.fit_resample(X,y)[1]).items())))
