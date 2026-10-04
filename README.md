# Stellar Object Classification (Galaxy / QSO / Star)

An XGBoost classifier for the Kaggle Playground Series S6E6 stellar classification task — predicting whether an astronomical observation is a galaxy, quasar (QSO), or star from photometric and redshift data.

## Results

- **Local held-out test accuracy: 95.37%** (confirmed via a fresh top-to-bottom run of the exact script in this repo)
- **Kaggle leaderboard accuracy: 93.48%**

Note: accuracy has varied by roughly 0.03–0.04 percentage points (95.37%–95.41%) across different runs of this same script with the same `random_state`. This is expected — XGBoost's histogram-based tree building isn't always bit-exact reproducible run-to-run due to multi-threaded floating-point summation order, even with a fixed seed. It is not a sign of a bug; the model, features, and hyperparameters are identical across runs.

## Approach

### Data
- Dropped `alpha`/`delta` (sky coordinates) — position shouldn't causally determine object type, so these were excluded rather than treated as predictive signal
- Target (`class`: GALAXY / QSO / STAR) label-encoded

### Feature engineering
- **Color indices**: `u-g`, `g-r`, `r-i`, `i-z` — differences between photometric bands, standard in astronomical classification since color relates directly to an object's spectral properties
- **Locus distances**: `stellar_locus_dist` and `qso_locus_dist` — Euclidean distance in color-color space from the well-known stellar locus and QSO locus regions. This approach was adapted from community solutions to similar stellar classification problems, then implemented and validated independently (see misclassification analysis below) rather than derived from scratch
- **Redshift zone**: `redshift` binned into four physically meaningful ranges, since redshift behaves very differently for nearby stars (near-zero) versus distant galaxies and quasars

### Model
- Single `XGBClassifier` (n_estimators=350, max_depth=8, learning_rate=0.10) — no ensembling; a single well-tuned XGBoost model outperformed the ensembling approaches tried during development
- Evaluated via held-out test accuracy and 5-fold cross-validation on the training split

## How to run

```bash
pip install pandas numpy scikit-learn xgboost
```

Place `train.csv` and `test.csv` in the working directory and run the script. It trains the model, prints test accuracy and a confusion matrix, and writes `submission.csv` in the competition's expected format.

## Known limitations

- **Train/test split is not stratified.** Given real class imbalance in the data (galaxies substantially outnumber quasars and stars), an unstratified split risks a test fold that doesn't perfectly reflect the true class distribution. In practice, results have been consistent across multiple runs, suggesting this isn't materially affecting the reported accuracy — but it's a rigor gap worth flagging rather than hiding.
- **Some exploratory feature ideas (one-hot encoded categorical features, redshift-interaction terms) were tried during development and ultimately not used in the final feature set** — kept out of this script entirely rather than left in as dead code.
- Exploratory data analysis (correlation plots, distribution comparisons across classes, misclassification analysis) was done separately during development and isn't included in this pipeline script, which is kept focused on the reproducible training/prediction path. This included checking suspect low-redshift QSO predictions and comparing color indices between low-redshift galaxies and stars, to validate that the adapted locus-distance features were behaving as expected on this dataset.


