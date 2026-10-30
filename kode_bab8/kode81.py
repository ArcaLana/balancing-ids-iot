!pip install -q imbalanced-learn xgboost

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn, imblearn, xgboost

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Struktur folder kerja
for folder in ("data", "output"):
    os.makedirs(folder, exist_ok=True)

# Catat versi (untuk bagian reprodusibilitas)
print("Python", sys.version.split()[0])
for lib in (np, pd, sklearn, imblearn, xgboost):
    print(lib.__name__, lib.__version__)
