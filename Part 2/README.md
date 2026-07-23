# Part 2 — Supervised Machine Learning Model — Build, Train, and Evaluate

## What the code does

`part2_model.py` (or equivalent notebook) loads `cleaned_data.csv` from
Part 1 and builds two supervised models on the Ames Housing data:

1. **A regression model** predicting `SalePrice` (continuous)
2. **A classification model** predicting whether a home sells above or
   below the median price (binary)

It encodes categorical features correctly, splits/scales without leakage,
trains and evaluates both models, and runs a bootstrap significance test
comparing two regularization strengths.

## Label Definitions

- **`y_reg`**: `SalePrice`, used as-is (continuous)
- **`y_clf`**: `(SalePrice > SalePrice.median()).astype(int)` — 1 if the
  home sold above the median price, 0 otherwise

## How to run

```bash
pip install pandas numpy matplotlib scikit-learn
python part2_model.py
```
Requires `cleaned_data.csv` (from Part 1) in the same folder.

**Outputs produced automatically:**
```
outputs/
├── roc_curve_baseline.png
├── ols_vs_ridge.csv
├── threshold_sensitivity.csv
├── regularization_comparison.csv
├── class_balance_before_after.csv
└── bootstrap_auc_ci.csv
```

## Encoding Strategy

- **Ordinal columns** (18 total, e.g. `Exter_Qual`, `Kitchen_Qual`,
  `Bsmt_Exposure`, `Functional`) were label-encoded with an explicit integer
  mapping that preserves their natural order (e.g.
  `Po=0 < Fa=1 < TA=2 < Gd=3 < Ex=4` for quality scales). This is valid
  because the categories genuinely have a meaningful rank.
- **Nominal columns** (19 total, e.g. `Neighborhood`, `House_Style`,
  `Sale_Type`) have no natural order, so they were **one-hot encoded**
  (`drop_first=True` to avoid multicollinearity). Label-encoding these
  would have implied a false ordinal relationship — e.g. it would tell the
  model `Neighborhood=5` is "more" than `Neighborhood=2` in some meaningful
  sense, which isn't true for a city name.

Final encoded feature matrix: **197 columns** (up from the original ~55
raw feature columns).

## Leak-Free Split & Scaling

Data was split 80/20 (`train_test_split(..., random_state=42)`) **before**
any scaling. `StandardScaler` was fit **only on `X_train`**, then used to
transform both `X_train` and `X_test`. Fitting the scaler on the full
dataset (train+test combined) would leak test-set mean/variance statistics
into the training process — the model would be implicitly "aware" of the
test distribution before ever seeing test labels, inflating evaluation
metrics beyond what would be seen on genuinely new data.

## Regression Results

| Model | MSE | R² |
|---|---|---|
| Linear Regression (OLS) | 934,586,924 | 0.8830 |
| Ridge (alpha=1.0) | 934,158,935 | 0.8835 |

**Coefficient interpretation:** the top-3 features by absolute coefficient
size are printed in the console output. A large **positive** coefficient
means a one-standard-deviation increase in that (scaled) feature is
associated with a proportional increase in predicted `SalePrice`; a large
**negative** coefficient means the opposite — that feature drives price
down as it increases.

**Ridge vs. OLS:** results are nearly identical here (R² differs by only
0.0005), meaning multicollinearity/overfitting wasn't a major issue for
plain OLS on this dataset. `alpha` in Ridge controls the strength of the
L2 penalty on coefficient magnitudes — higher alpha shrinks coefficients
more aggressively toward zero, trading a little bias for lower variance.
Since OLS and Ridge perform almost the same here, the data doesn't suffer
from severe overfitting that regularization would need to fix.

## Classification Results

### Class balance (imbalance handling)

```
BEFORE any imbalance handling (y_clf_train): 0 → 1186, 1 → 1158 (49.4% minority)
AFTER imbalance handling decision:            0 → 1186, 1 → 1158 (unchanged)
```

Since the target was constructed by splitting at the median, classes are
already ~50/50 by design — no resampling (e.g. SMOTE) was needed.
`class_weight='balanced'` was applied anyway as a light-touch safeguard.
The counts are identical before/after because `class_weight` re-weights
the loss function during training; it does not alter the training rows
themselves (unlike SMOTE, which would change these counts).

### Logistic Regression (C=1.0) — baseline

- **AUC: 0.9798**
- Confusion matrix, full classification report (precision/recall/F1 per
  class), and ROC curve are in the console output / `outputs/roc_curve_baseline.png`

**Precision** = TP / (TP + FP) — of all homes predicted "above median,"
what fraction actually were?
**Recall** = TP / (TP + FN) — of all homes actually "above median," what
fraction did the model catch?

For a home-price classifier used to flag "premium" listings for a sales
team, **recall** arguably matters slightly more — missing a genuinely
above-median home (a false negative) means a lost sales opportunity,
whereas a false positive just means one extra listing gets reviewed and
discarded. Either way, both are already high here (~0.92–0.94), so this
model performs strongly regardless of which metric is prioritized.

**AUC = 0.9798** means the model has excellent ability to rank a randomly
chosen above-median home higher than a randomly chosen below-median home
— a near-perfect separation between the two classes.

### Threshold sensitivity (0.30–0.70)

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | — | — | **0.9435 (highest)** |
| 0.40 | 0.9377 | 0.9377 | 0.9377 |
| 0.50 | 0.9398 | 0.9213 | 0.9305 |
| 0.60 | 0.9517 | 0.9049 | 0.9277 |
| 0.70 | 0.9609 | 0.8852 | 0.9215 |

*(exact 0.30-row precision/recall values are in the saved
`threshold_sensitivity.csv`)*

**F1-maximizing threshold: 0.30.** Since classes are balanced and both
precision/recall matter roughly equally here, a domain-appropriate choice
would stick close to the default 0.5 or the F1-optimal 0.30 — lowering the
threshold trades some precision for higher recall (catches more
above-median homes, at the cost of a few more false positives), which is
a reasonable choice if the business cost of *missing* a premium listing
is higher than reviewing a false one.

### Regularization Experiment (C=1.0 vs. C=0.01)

| Model | Precision | Recall | AUC |
|---|---|---|---|
| C=1.0 | 0.9398 | 0.9213 | 0.9798 |
| **C=0.01** | **0.9561** | **0.9279** | **0.9885** |

`C` is the **inverse** of regularization strength: small `C` = strong L2
penalty (simpler, more constrained model), large `C` = weak penalty
(closer to unregularized MLE). **Reducing C to 0.01 improved performance**
on every metric here — precision, recall, and AUC all rose slightly. This
suggests the baseline (C=1.0) model, with 197 features, was mildly
overfitting the training data, and the added regularization improved its
generalization to the held-out test set.

### Bootstrap Confidence Interval for AUC Difference

500 bootstrap resamples of the test set were drawn (with replacement).
For each, the AUC difference (C=1.0 minus C=0.01) was computed.

- **Mean AUC difference: −0.0088**
- **95% CI: [−0.0148, −0.0035]**
- **Interval excludes zero: Yes**

Since the interval is entirely **below** zero, this means **C=0.01
reliably outperforms C=1.0** on AUC across resamples of the test set — not
just a lucky single split. This confirms the regularization experiment's
finding above is statistically consistent, not sampling noise: the
stronger-regularized model's advantage holds up.

## Files in this folder

```
part2/
├── PART_2_ModelBuild.ipynb   (outputs cleared)
├── cleaned_data.csv
├── outputs/
│   ├── roc_curve_baseline.png
│   ├── ols_vs_ridge.csv
│   ├── threshold_sensitivity.csv
│   ├── regularization_comparison.csv
│   ├── class_balance_before_after.csv
│   └── bootstrap_auc_ci.csv
└── README.md
```
