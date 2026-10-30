from sklearn.ensemble import RandomForestClassifier

def model_rf(class_weight=None):
    return RandomForestClassifier(n_estimators=100, max_depth=None,
                                  class_weight=class_weight,
                                  n_jobs=-1, random_state=RANDOM_STATE)

p = model_rf().get_params()
print("Random Forest:", p["n_estimators"], "pohon | max_depth =", p["max_depth"], "| class_weight =", p["class_weight"])
