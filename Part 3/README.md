# Part 3 — Advanced Modeling — Ensembles, Tuning, and Full ML Pipeline

## What the code does

`part3_ensembles.py` (or equivalent notebook) reloads `cleaned_data.csv`,
rebuilds the same encoding/split/scaling as Part 2, then trains and
compares a series of tree-based models (Decision Tree, Random Forest,
Gradient Boosting), tunes the best one with `GridSearchCV`, runs a manual
learning curve, and serializes the final tuned pipeline to `best_model.pkl`.

## How to run

```bash
pip install pandas numpy scikit-learn joblib
python part3_ensembles.py
```
Requires `cleaned_data.csv` (from Part 1) in the same folder.

**Outputs produced automatically:**
```
best_model.pkl
outputs/
├── model_comparison_summary.csv
├── cv_results.csv
├── learning_curve.csv
├── top5_feature_importance.csv
└── ablation_study.csv
```

## Decision Tree — Unconstrained vs. Controlled

| Model | Train Accuracy | Test Accuracy | Train-Test Gap |
|---|---|---|---|
| Unconstrained (max_depth=None) | 1.0000 | 0.9181 | **0.0819** |
| Controlled (max_depth=5, min_samples_split=20) | 0.9245 | 0.9198 | **0.0047** |

The unconstrained tree shows clear signs of **overfitting**: it perfectly
memorizes the training data (100% train accuracy) but drops to 91.8% on
unseen test data — an 8.2-point gap. This is the classic signature of a
high-variance model: a decision tree fits the training data greedily at
each split, choosing whatever split best separates the *current* training
sample without ever revisiting or correcting earlier decisions, so it
happily grows branches that memorize noise specific to the training set.

The controlled tree closes this gap almost entirely (0.47 points) while
achieving nearly identical test accuracy (91.98% vs. 91.81%) — proof that
much of the unconstrained tree's depth was pure overfitting, not genuine
signal.

- **`max_depth`** limits how many splits deep the tree can grow, directly
  reducing variance at the cost of a small amount of bias (the tree can no
  longer perfectly separate every training example).
- **`min_samples_split`** prevents a node from splitting further unless it
  has at least this many samples, which stops the tree from creating
  splits that respond to noise in tiny, statistically unreliable subsets.

## Gini vs. Entropy (both max_depth=5)

| Criterion | Test Accuracy |
|---|---|
| Gini | 0.9249 |
| Entropy | 0.9215 |

**Gini impurity**: `1 − Σ pᵢ²`
**Entropy**: `−Σ pᵢ log₂(pᵢ)`

A node with **Gini = 0** means every sample in that node belongs to the
same class — the split has achieved perfect purity at that point in the
tree. The two criteria perform almost identically here (0.3-point
difference), which is typical — Gini and Entropy usually agree on the
"best" split in practice, differing mainly in computational cost (Gini is
slightly cheaper, no logarithm).

## Random Forest

- **Train accuracy: 0.9881**
- **Test accuracy: 0.9454**
- **Test ROC-AUC: 0.9904**

**Top 5 features by importance:**

| Feature | Importance |
|---|---|
| `Gr_Liv_Area` | 0.0852 |
| `Overall_Qual` | 0.0719 |
| `Garage_Yr_Blt` | 0.0518 |
| `Garage_Area` | 0.0493 |
| `Full_Bath` | 0.0488 |

Random Forest computes feature importance as the **average reduction in
Gini impurity** contributed by that feature across every split, in every
tree in the forest, averaged over all trees. This is fundamentally
different from a linear regression coefficient: a coefficient measures
the *linear, additive* effect of a one-unit change in a feature holding
others constant, while Random Forest importance measures how much a
feature *actually helped separate classes* across many different split
points and interaction contexts — it can capture non-linear and
interaction effects a linear coefficient cannot.

**Bagging concept:** each tree in the forest is trained on a bootstrap
sample — a random sample of the training data drawn *with replacement*, so
each tree sees a slightly different subset (some rows repeated, some
omitted). At every split, only a random subset of √(number of features)
features is even considered as a candidate. This double randomization
means individual trees are decorrelated from each other — they make
different mistakes on different parts of the data. Averaging their
predictions cancels out much of that individual noise, which is why the
ensemble's variance is far lower than any single deep, unconstrained tree.

## Gradient Boosting

- **Train accuracy: 0.9744**
- **Test accuracy: 0.9471**
- **Test ROC-AUC: 0.9900**

Included in the cross-validated comparison below.

## Feature Ablation Study

5 lowest-importance features from the Random Forest (all had **importance
= 0.0**): `Sale_Type_Con`, `Sale_Type_ConLw`, `Sale_Type_ConLI`,
`Sale_Type_Oth`, `Sale_Type_VWD`.

| Model | Test ROC-AUC |
|---|---|
| Full (197 features) | 0.9904 |
| Reduced (192 features, 5 removed) | 0.9903 |
| **Change** | **−0.0001** |

These 5 features were **genuinely uninformative** — removing them changed
AUC by essentially nothing (a rounding-level difference). This is a
favorable finding for production deployment: a simpler, 192-feature model
has slightly lower inference cost and one less category of feature to
maintain (these were rare one-hot dummy categories for uncommon sale
types), with zero measurable cost in predictive performance. In general,
this trade-off is only acceptable when the AUC degradation stays below
whatever tolerance the business sets — here, a change of −0.0001 is well
within any reasonable threshold.

## Cross-Validated Comparison (5-fold, `scoring='roc_auc'`)

| Model | CV Mean AUC | CV Std AUC |
|---|---|---|
| Logistic Regression | 0.9710 | 0.0057 |
| Decision Tree (max_depth=5) | 0.9296 | 0.0067 |
| Random Forest | 0.9805 | 0.0040 |
| **Gradient Boosting** | **0.9825** | **0.0028** |

Cross-validation gives a more reliable estimate of generalization than a
single train-test split because it evaluates the model on 5 different
held-out folds rather than one — a single split's test-set score can be
unusually lucky or unlucky depending on which rows happened to land in the
test set; averaging over 5 folds smooths out that sampling variance and
also gives a standard deviation, showing how *consistent* the model's
performance is (Gradient Boosting's low std of 0.0028 indicates it's the
most stable performer here).

## Hyperparameter Tuning — GridSearchCV

**Parameter grid:**
```python
param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [5, 10, None],
    'randomforestclassifier__min_samples_leaf': [1, 5]
}
```
3 × 3 × 2 = **18 parameter combinations × 5 folds = 90 total model fits.**

**Best params:** `max_depth=10, min_samples_leaf=1, n_estimators=200`
**Best CV score (roc_auc): 0.9810**
**Best pipeline test-set AUC: 0.9904**

**Grid Search vs. Randomized Search trade-off:** Grid Search is exhaustive
— it tries every combination in the grid, guaranteeing the best
combination *within that grid* is found, but cost grows multiplicatively
with each added parameter or value (adding one more `n_estimators` option
would mean 30 more fits). Randomized Search instead samples a fixed number
of random combinations from the parameter space, which scales much better
for large grids or continuous ranges, at the cost of no guarantee the true
optimum is ever tried — a reasonable trade when the grid is large or
compute time is limited.

## Manual Learning Curve

| Training Fraction | Training AUC | Test AUC |
|---|---|---|
| 20% | ~1.0000 | 0.9882 |
| 40% | ~1.0000 | 0.9891 |
| 60% | ~1.0000 | 0.9897 |
| 80% | ~1.0000 | 0.9901 |
| 100% | ~1.0000 | 0.9904 |

*(exact values are in `outputs/learning_curve.csv`)*

- **(i)** Training AUC stays essentially at 1.0 across all fractions —
  it does **not** decrease as the training set grows, which is slightly
  atypical for a "textbook" high-variance model (usually training
  performance drops a bit as more, harder-to-fit examples are added). Here
  the tuned Random Forest is powerful enough to still fit almost every
  training fold near-perfectly even at 100% of the data.
- **(ii)** Test AUC **does** increase steadily with more training data
  (0.9882 → 0.9904), suggesting more data would likely help further.
- **(iii) Conclusion:** the model is **mildly capacity-limited, not
  severely data-limited** — test AUC is still climbing slightly at 100%,
  so a bit more data could help, but the training AUC staying pinned near
  1.0 throughout suggests the bigger lever for improvement would actually
  be adding *more* regularization (to close the persistent train/test
  gap) rather than simply collecting more rows.

## Serialization

`best_model.pkl` (the tuned Random Forest pipeline) was saved with
`joblib.dump()`. A reload-and-predict block was run and confirmed working
without errors — loading the file with `joblib.load()` and calling
`.predict()` on two real test rows.

## Final Summary Comparison Table

| Model | CV Mean AUC | CV Std AUC | Test AUC |
|---|---|---|---|
| Logistic Regression | 0.9710 | 0.0057 | 0.9798 |
| Decision Tree (depth=5) | 0.9296 | 0.0067 | 0.9657 |
| Random Forest | 0.9805 | 0.0040 | 0.9904 |
| Gradient Boosting | 0.9825 | 0.0028 | 0.9900 |
| **Tuned RF (GridSearchCV)** | 0.9810 | — | **0.9904** |

## Recommendation

**The tuned Random Forest** (saved as `best_model.pkl`) is the recommended
model to hand to the client. It achieves the highest test-set AUC (0.9904,
tied with the untuned Random Forest) and one of the tightest cross-
validation standard deviations, meaning its performance is both strong and
consistent across different data splits — not a lucky single result.
Gradient Boosting is a very close second (test AUC 0.9900, best CV mean of
0.9825) and would be a reasonable alternative if training/inference time
were more of a constraint, but since Random Forest already delivers
equal-or-better test performance and is the model that was actually tuned
and serialized in this pipeline, it's the practical choice to deploy.

## Files in this folder

```
part3/
├── part_3_ensemble.ipynb   (outputs cleared)
├── cleaned_data.csv
├── best_model.pkl
├── outputs/
│   ├── model_comparison_summary.csv
│   ├── cv_results.csv
│   ├── learning_curve.csv
│   ├── top5_feature_importance.csv
│   └── ablation_study.csv
└── README.md
```
