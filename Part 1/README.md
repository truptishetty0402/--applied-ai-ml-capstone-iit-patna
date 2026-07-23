# Part 1 — Data Acquisition, Cleaning, and Exploratory Analysis

## Dataset

**Ames Housing Dataset** (Iowa residential home sales, 2006–2010)
Source: https://www.kaggle.com/datasets/prevek18/ames-housing-dataset

Chosen because it has 2,930 rows and 82 raw columns — well above the
500-row/5-column minimum — with a natural continuous target (`SalePrice`),
a rich mix of ordinal quality columns (e.g. `Kitchen_Qual`: Po/Fa/TA/Gd/Ex)
and nominal categorical columns (e.g. `Neighborhood`), and realistic
missing-data patterns that make the cleaning tasks meaningful rather than
trivial.

## What the code does

`part1_eda.py` (or the equivalent notebook) performs, in order:
1. Loads the raw CSV, inspects shape/dtypes/head, drops `Order`/`PID`
   (row identifiers, not predictive features)
2. Computes null count/percentage for **every** column, drops columns
   exceeding 20% nulls, and median/mode-fills the rest
3. Detects and removes duplicate rows
4. Corrects `MS_SubClass` (a building-class code wrongly inferred as
   numeric) to `category`, and converts `Neighborhood` to `category` for
   memory efficiency
5. Computes descriptive statistics and skewness for all numeric columns
6. Detects IQR-based outliers on `SalePrice` and `Gr_Liv_Area` (documented,
   not dropped)
7. Produces all 5 required visualizations plus a correlation heat map
8. Compares mean vs. median for the two most-skewed columns, computes
   Spearman vs. Pearson correlation, and a grouped aggregation by
   `Neighborhood`
9. Saves the cleaned dataset to `cleaned_data.csv`

## How to run

```bash
pip install pandas numpy matplotlib seaborn
python part1_eda.py
```
Requires `AmesHousing.csv` in the same folder (download from the Kaggle
link above).

**Outputs produced automatically:**
- `cleaned_data.csv` — feeds Parts 2, 3, and 4
- `eda_plots/*.png` — the 6 required plots

## Key Findings & Design Decisions

### Null handling
27 of 74 columns (after dropping IDs) had missing values. Columns exceeding
20% nulls (`Pool_QC`, `Misc_Feature`, `Alley`, `Fence`, `Mas_Vnr_Type`,
`Fireplace_Qu`) were **dropped entirely** rather than imputed — for most of
these, a missing value means "feature not present" (e.g. no pool), so
imputing a central tendency would be misleading. Remaining numeric columns
below 20% nulls were filled with the **median**, not the mean, because
several of them (e.g. `Lot_Frontage`, `Mas_Vnr_Area`) are skewed; the mean
would be dragged toward outliers, while the median stays representative of
a typical value.

### Duplicates
0 duplicate rows found — no removal needed, so null percentages were
unaffected by this step.

### Data type correction
`MS_SubClass` was stored as `int64` but is actually a categorical building-
class code (20, 60, 120, ...) with no real numeric/ordinal meaning —
treating it as numeric is a dtype error. Converted to `category`.
`Neighborhood` (a repetitive string column) was also converted to
`category`. Memory usage dropped from **6,433.21 KB to 6,261.16 KB**
(≈2.7% reduction).

### Skewness
The most skewed numeric column is **`Misc_Val`** (skew ≈ 22.0) — an
extreme positive skew, meaning most homes have `Misc_Val = 0` but a small
number have very large values (e.g. an elevator or tennis court add-on),
pulling the mean far above the typical value. Mean-imputation on a column
this skewed would badly overstate what a "typical" value looks like — the
median is the appropriate choice here.

### IQR outliers
- **`SalePrice`**: 137 outlier rows (upper-bound driven — a small number
  of very expensive homes)
- **`Gr_Liv_Area`**: 75 outlier rows (also upper-bound driven — a handful
  of unusually large homes)

Outliers were **documented, not dropped**, at this stage. Decision deferred
to Part 2: these are legitimate, real transactions (not data-entry errors),
so rather than deleting them, Part 2's regression models can be evaluated
with and without capping to see how much they influence the fit.

### Visualizations
All 5 required types were produced (line plot of `SalePrice` over
sale-date order, bar chart of mean `SalePrice` by `House_Style`, histogram
of `Misc_Val`, scatter plot of `Gr_Liv_Area` vs. `SalePrice`, box plot of
`SalePrice` by `Kitchen_Qual`), plus the correlation heat map.

- **Scatter plot**: `Gr_Liv_Area` vs. `SalePrice` shows a clear positive,
  moderately strong linear relationship — larger homes command higher
  prices, as expected.
- **Box plot**: median `SalePrice` rises monotonically from `Po` through
  `Ex` kitchen quality, with spread (IQR) also widening at the higher
  quality tiers — better kitchens aren't just pricier on average, they're
  also more variable in price.

### Correlation heat map
The highest absolute correlation pair is **`Garage_Cars` and
`Garage_Area`** (r = 0.890). This is very unlikely to be a direct causal
relationship between the two — instead, both are driven by a third,
unmeasured variable: **the physical size of the garage structure itself**.
A bigger garage naturally holds more cars *and* has more square footage;
the two columns are really two different measurements of the same
underlying "how big is this garage" fact.

### Imputation strategy comparison (Task 8a)
Top-2 highest-|skew| numeric columns: **`Misc_Val`** and **`Pool_Area`**.
For both, the **median** was chosen over the mean for any remaining
imputation — both columns are extremely right-skewed (most values are 0,
a few are very large), so the mean is pulled well above what's typical,
while the median (0, for both) correctly reflects that most homes simply
don't have this feature. `isnull().sum()` confirms 0 nulls remain in both
columns after this step.

### Spearman vs. Pearson (Task 8b)
The top-3 column pairs by `|Spearman − Pearson|` were computed and printed
(see console output / notebook for exact values). Where `|Spearman| >
|Pearson|`, the relationship is monotonic but non-linear — the variables
move together consistently, but not proportionally (e.g. price effects
that plateau at high values). Where `|Pearson| ≥ |Spearman|`, the
relationship is closer to linear. **Pearson** is used to guide Part 2's
linear-model feature selection, since Part 2's regression models
(Linear/Ridge) assume linear relationships; Spearman is noted as a
secondary check for any features whose importance might be
underestimated by a purely linear correlation.

### Grouped aggregation (Task 8c)
Grouped `SalePrice` by `Neighborhood`:
- **Highest mean**: `NoRidge`
- **Highest standard deviation**: `StoneBr`
- **Ratio of highest to lowest group mean**: **3.45×**

A ratio this large (homes in the priciest neighborhood sell for over 3x
the homes in the cheapest, on average) suggests `Neighborhood` carries
strong predictive signal and is worth keeping as a modeling feature. The
high within-group standard deviation in `StoneBr`, however, is a caution:
neighborhood alone isn't sufficient to predict price precisely for a home
in that group — other features (size, quality) still matter a lot within
it.

## Files in this folder

```
part1/
├── PART_1_Exploratory_Analysis.ipynb   (outputs cleared)
├── AmesHousing.csv
├── cleaned_data.csv
├── eda_plots/
│   ├── 01_line_saleprice_over_time.png
│   ├── 02_bar_mean_saleprice_by_house_style.png
│   ├── 03_histogram_most_skewed.png
│   ├── 04_scatter_grlivarea_vs_saleprice.png
│   ├── 05_boxplot_saleprice_by_kitchenqual.png
│   └── 06_correlation_heatmap.png
└── README.md
```
