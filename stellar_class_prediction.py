import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

# -------------------------
# LOAD AND CLEAN TRAIN
# -------------------------
df = pd.read_csv("train.csv")
df.drop(["id", "alpha", "delta"], axis=1, inplace=True)

encoder = LabelEncoder()
df["class"] = encoder.fit_transform(df["class"])
# encoder.classes_ == ['GALAXY', 'QSO', 'STAR'] — 3 classes, no missing labels

# -------------------------
# FEATURE ENGINEERING
# -------------------------
df["u-g"] = df["u"] - df["g"]
df["g-r"] = df["g"] - df["r"]
df["r-i"] = df["r"] - df["i"]
df["i-z"] = df["i"] - df["z"]

# Distance from known stellar-locus and QSO-locus color-color regions
df["stellar_locus_dist"] = np.sqrt((df["g-r"] - 0.52) ** 2 + (df["r-i"] - 0.25) ** 2)
df["qso_locus_dist"] = np.sqrt((df["g-r"] - 0.24) ** 2 + (df["r-i"] - 0.15) ** 2)

# Redshift binned into physically meaningful zones
df["redshift_zone"] = pd.cut(
    df["redshift"], bins=[-999, 0.01, 0.15, 0.267, 999], labels=[0, 1, 2, 3]
).astype(int)

FEATURES = ["u", "g", "r", "i", "z", "redshift", "u-g", "g-r", "r-i", "i-z",
            "stellar_locus_dist", "qso_locus_dist", "redshift_zone"]

x = df[FEATURES]
y = df["class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

# -------------------------
# MODEL
# -------------------------
model = XGBClassifier(n_estimators=350, max_depth=8, learning_rate=0.10, random_state=42)
model.fit(x_train, y_train)

predictions = model.predict(x_test)
print(f"Test accuracy: {accuracy_score(y_test, predictions):.4f}")
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

cv_scores = cross_val_score(model, x_train, y_train, cv=5)
print(f"CV mean (train fold only): {cv_scores.mean():.4f}")

# -------------------------
# PROCESS KAGGLE TEST FILE AND PREDICT
# -------------------------
test_df_raw = pd.read_csv("test.csv")
test_df = test_df_raw.copy()
test_df.drop(["id", "alpha", "delta"], axis=1, inplace=True)

test_df["u-g"] = test_df["u"] - test_df["g"]
test_df["g-r"] = test_df["g"] - test_df["r"]
test_df["r-i"] = test_df["r"] - test_df["i"]
test_df["i-z"] = test_df["i"] - test_df["z"]
test_df["stellar_locus_dist"] = np.sqrt((test_df["g-r"] - 0.52) ** 2 + (test_df["r-i"] - 0.25) ** 2)
test_df["qso_locus_dist"] = np.sqrt((test_df["g-r"] - 0.24) ** 2 + (test_df["r-i"] - 0.15) ** 2)
test_df["redshift_zone"] = pd.cut(
    test_df["redshift"], bins=[-999, 0.01, 0.15, 0.267, 999], labels=[0, 1, 2, 3]
).astype(int)

# Defensive: median-impute rather than drop, in case a future test set has gaps
# (verified zero missing values on the actual competition test set — see README)
test_features = test_df[FEATURES].fillna(test_df[FEATURES].median())

test_predictions = model.predict(test_features)

submission = pd.DataFrame({
    "id": test_df_raw["id"],
    "class": encoder.inverse_transform(test_predictions),
})
submission.to_csv("submission.csv", index=False)
print("Submission saved.")