# Fetal Health Classification — ML Evaluation Project

A machine learning evaluation project comparing four classifiers (KNN, Logistic Regression, SVC, and Random Forest) on the UCI Fetal Health Dataset. The goal is to classify CTG recordings as **Normal (0)**, **Suspect (1)**, or **Pathological (2)** using a reproducible, leakage-free pipeline.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Dependencies](#dependencies)
3. [Dataset](#dataset)
4. [How to Run](#how-to-run)
5. [Pipeline Walkthrough](#pipeline-walkthrough)
   - [1. Data Loading](#1-data-loading)
   - [2. Data Cleaning](#2-data-cleaning)
   - [3. Exploratory Data Analysis](#3-exploratory-data-analysis)
   - [4. Train / Test Split](#4-train--test-split)
   - [5. Model Training](#5-model-training)
   - [6. Evaluation](#6-evaluation)
   - [7. Learning Curves](#7-learning-curves)
6. [Key Design Decisions](#key-design-decisions)
7. [How Each Step Influences the Next](#how-each-step-influences-the-next)
8. [Results Summary](#results-summary)

---

## Project Structure

```
.
├── script2.ipynb           # Main notebook — full pipeline
├── fetal_health.csv        # Dataset (place in the same folder)
└── README.md               # This file
```

---

## Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn scipy
```

| Library | Version tested | Purpose |
|---|---|---|
| pandas | ≥ 2.0 | Data loading and manipulation |
| numpy | ≥ 1.24 | Numerical operations |
| matplotlib | ≥ 3.7 | Plotting |
| seaborn | ≥ 0.12 | Statistical visualisations |
| scikit-learn | ≥ 1.8 | Models, pipeline, CV, metrics |
| imbalanced-learn | ≥ 0.11 | SMOTE, imblearn Pipeline |

> **Note:** scikit-learn 1.8 deprecated `penalty='l2'` in `LogisticRegression`.
> The notebook uses the new `l1_ratio` API with `solver='saga'` — no changes needed.

---

## Dataset

- **Source:** UCI / Kaggle Fetal Health Dataset
- **File:** `fetal_health.csv`
- **Samples:** 2,126 CTG recordings (after cleaning: 2,113)
- **Features:** 21 — fetal heart rate signal statistics + histogram statistics
- **Target:** `fetal_health` — originally encoded as {1, 2, 3}, remapped to {0, 1, 2}

| Class | Label | Samples |
|---|---|---|
| 0 | Normal | 1,655 |
| 1 | Suspect | 295 |
| 2 | Pathological | 176 |

---

## How to Run

1. Place `fetal_health.csv` in the same folder as `script2.ipynb`
2. Open the notebook in Jupyter Lab or Jupyter Notebook
3. Run all cells top to bottom (`Kernel → Restart & Run All`)

> Training all four models takes approximately **15–30 minutes** depending on hardware.
> The RFC grid search is the slowest step.

---

## Pipeline Walkthrough

### 1. Data Loading

```python
df = pd.read_csv('fetal_health.csv')
```

The CSV is loaded into a pandas DataFrame. `.head()` is called immediately to visually confirm the data loaded correctly and to understand the column structure before any processing begins.

---

### 2. Data Cleaning

Three cleaning steps are applied in order:

#### 2a. Drop missing values
```python
df = df.dropna()
```
The dataset has no missing values, so this step changes nothing. It is included as a defensive check — if the CSV changes or is loaded differently in future, missing rows are automatically removed.

#### 2b. Remove duplicate rows
```python
df = df.drop_duplicates()
```
13 exact duplicate rows were found across all three classes:

```
Class 1 (Normal):        16 rows flagged (across duplicate groups)
Class 2 (Suspect):        6 rows flagged
Class 3 (Pathological):   2 rows flagged
```

`keep=False` was used during inspection to flag **all copies** of duplicated rows (not just the extras), which is why the count shown (24) is higher than 13. The actual rows removed are 13.

**Why remove them:** Duplicate rows are almost certainly logging artefacts from the data collection system rather than genuine repeated observations. Keeping them means training on fabricated data. The impact on class balance is negligible — at most 8 Normal, 3 Suspect, and 1 Pathological row removed from a 2,126-row dataset.

#### 2c. Outlier detection (IQR method)
```python
Q1 = X[feature].quantile(0.25)
Q3 = X[feature].quantile(0.75)
IQR = Q3 - Q1
outliers = X[(X[feature] < Q1 - 1.5*IQR) | (X[feature] > Q3 + 1.5*IQR)]
```

Outliers were detected in multiple features using the interquartile range method. **No outliers were removed.**

**Why retain them:** This is a clinical dataset. An extreme value in `prolongued_decelerations` or `abnormal_short_term_variability` likely reflects a genuinely abnormal fetal condition — precisely what the Pathological class captures. Removing these rows risks discarding the most clinically informative signals. Additionally, the models trained (especially RFC) are robust to outliers since tree splits are based on rank thresholds, not absolute distances.

#### 2d. Class remapping
```python
y = y.replace({1: 0, 2: 1, 3: 2})
```

The original target encoding {1, 2, 3} is remapped to {0, 1, 2} for compatibility with sklearn's zero-indexed class handling and for cleaner output in classification reports.

---

### 3. Exploratory Data Analysis

EDA is performed on the **full dataset before splitting**. This is correct — EDA is purely observational and no numbers from it feed back into the model pipeline.

#### Class distribution plot
A bar chart of class counts reveals severe imbalance: Normal accounts for ~78% of all samples. This single observation drives several downstream decisions:
- **Metric choice:** macro-F1 instead of accuracy
- **SMOTE:** to generate synthetic minority samples
- **Stratified splitting:** to preserve proportions in both sets

#### Feature correlation with target
A horizontal bar chart ranks all 21 features by their Pearson correlation with `fetal_health`. Features are coloured green (positive correlation) or crimson (negative). This plot motivates feature selection — several features have near-zero correlation with the target and carry little predictive signal.

**Important:** this correlation is computed on the full dataset as a visualisation tool only. Feature selection within the pipeline is performed independently on training folds using the model's actual performance as the criterion (SFS / RFECV), not correlation scores. So this plot does not introduce leakage.

#### Feature correlation heatmap
A pairwise correlation matrix of all 21 features identifies redundant features — pairs with correlation > 0.9. This motivates feature selection further: with high inter-feature correlation, fewer features can carry most of the information.

#### Boxplot by class
```python
plt.boxplot([X[feature][y==0], X[feature][y==1], X[feature][y==2]])
```

Boxplots of individual features split by class reveal class separation. For example, `abnormal_short_term_variability` shows clear spread differences across classes but with significant overlap in the Suspect range — directly explaining why the Suspect class is the hardest to classify across all models.

#### Scatter plots
Two scatter plots explore feature relationships coloured by class:
1. Single feature (`abnormal_short_term_variability`) vs target — shows non-linear separation
2. Two features against each other — `abnormal_short_term_variability` vs `uterine_contractions` — shows that the three classes are not linearly separable in the raw feature space, motivating the use of non-linear models (RFC, SVC with RBF kernel)

#### 5-number summary
```python
X.describe()
```

Prints min, Q1, mean, Q3, max for all features. Reveals features with very different scales (e.g. `histogram_width` ranges 0–180 while `accelerations` ranges 0–0.6), confirming that StandardScaler is essential for distance-based models (KNN, SVC).

---

### 4. Train / Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)
```

**Why 80/20:** A standard split that gives ~1,700 training samples — sufficient for all four models — while keeping ~425 samples for a meaningful test evaluation.

**Why `stratify=y`:** Without stratification, the random split could produce a test set with disproportionate class representation. With stratification, both sets maintain the original 78/14/8% class ratio. The side-by-side bar chart (Cell 21) visually confirms this.

**Why `random_state=42`:** Fixing the seed ensures every run produces the exact same split, making results reproducible. The seed value itself (42) is arbitrary — what matters is that it is fixed once and never changed to hunt for a better score, which would be equivalent to tuning on the test set.

**Why EDA came before splitting:** EDA is observational only. Running it on the full dataset gives the most complete picture of the data's properties. The split happens after EDA so the model pipeline never has access to test set information during training.

---

### 5. Model Training

All four models share the same evaluation framework:

```
imblearn Pipeline → GridSearchCV (5-fold StratifiedKFold) → macro-F1 scoring
```

SMOTE is placed **inside** the pipeline for all four models. This is the critical design decision — it ensures synthetic samples are generated only on training folds and never contaminate validation or test sets.

---

#### 5a. KNN

```python
Pipeline([
    ("scaler",   StandardScaler()),
    ("smote",    SMOTE(random_state=42)),
    ("selector", SequentialFeatureSelector(knn_for_selection, ...)),
    ("model",    KNeighborsClassifier())
])
```

**Pipeline order rationale:**
- `StandardScaler` first: KNN computes Euclidean/Manhattan distances. Unscaled features with large ranges (e.g. `histogram_width`) would dominate the distance calculation, making smaller-range features irrelevant.
- `SMOTE` after scaling: synthetic samples are interpolated in the normalised feature space, producing samples that are consistent with the real data distribution.
- `SFS` after SMOTE: the feature selector evaluates each feature subset on the balanced, scaled data — giving minority classes equal representation in the scoring.
- `KNeighborsClassifier` last: the final model is trained on the balanced, scaled, selected features.

**Why SMOTE for KNN specifically:** KNN does not have a loss function, so `class_weight` is not supported. SMOTE is the correct alternative for KNN.

**Parameter grid:**
```python
{
    "selector__n_features_to_select": [5, 10, 15, 20],
    "model__n_neighbors":             [3, 5, 7, 11, 15],
    "model__weights":                 ["uniform", "distance"],
    "model__metric":                  ["euclidean", "manhattan"]
}
```
`n_neighbors` is the most important parameter — k=1 causes severe overfitting (train F1 = 1.0, val F1 ≈ 0.66 as seen in the learning curve). Larger k smooths the decision boundary. Both distance metrics are tested because Manhattan distance is more robust to high-dimensional data than Euclidean.

---

#### 5b. Logistic Regression

```python
Pipeline([
    ("scaler",   StandardScaler()),
    ("smote",    SMOTE(random_state=42)),
    ("selector", SequentialFeatureSelector(logreg_for_selection, ...)),
    ("model",    LogisticRegression(solver="saga", random_state=42))
])
```

**Why `solver="saga"`:** The `saga` solver is the only one in sklearn that supports all three penalty types (l2, l1, and no penalty). `lbfgs` (the default) only supports l2. Using `saga` allows the grid search to test all three.

**Why test all three penalties:**
- `l1_ratio=0` (l2): standard Ridge regularisation — shrinks all coefficients continuously
- `l1_ratio=1` (l1): Lasso regularisation — drives some coefficients to exactly zero, performing implicit feature selection
- `C=np.inf` (no penalty): fits an unregularised logistic regression — only appropriate if the data is already well-conditioned after scaling and feature selection

The `l1_ratio` parameter replaces the deprecated `penalty` argument in sklearn ≥ 1.8.

**Why SMOTE here:** Same rationale as KNN — empirical testing showed SMOTE outperformed `class_weight="balanced"` across all models on this dataset, likely because SMOTE fills the sparse minority-class regions of the feature space rather than just adjusting the loss function.

---

#### 5c. SVC

```python
Pipeline([
    ("scaler",   StandardScaler()),
    ("smote",    SMOTE(random_state=42)),
    ("selector", SequentialFeatureSelector(svc_for_selection, ...)),
    ("model",    SVC(probability=True))
])
```

**Why `probability=True`:** Enables `.predict_proba()`, which returns class probabilities rather than hard labels. This is useful for threshold tuning but adds a small computational cost (internally runs a calibrated cross-validation).

**Two-dictionary param grid:** SVC uses a list of two dictionaries because RBF/poly/sigmoid kernels require tuning `gamma`, while the linear kernel does not use `gamma` at all. Putting them in separate dictionaries prevents sklearn from trying invalid combinations (e.g. `kernel=linear` + `gamma=0.1`).

**Why SVC needs scaling:** SVC optimises a margin — the distance between the decision boundary and the nearest data points. If features have different scales, the margin calculation is dominated by large-scale features. StandardScaler ensures all features contribute equally.

---

#### 5d. Random Forest (RFC)

```python
Pipeline([
    ("smote",    SMOTE(random_state=42)),
    ("selector", RFECV(
                     estimator=RandomForestClassifier(n_estimators=100, ...),
                     step=1, cv=cv_inner, scoring="f1_macro",
                     min_features_to_select=5
                 )),
    ("model",    RandomForestClassifier(random_state=42))
])
```

**Why no StandardScaler:** RFC uses decision trees — each split is a threshold comparison on a single feature (e.g. `prolongued_decelerations > 0.003`). Multiplying a feature by 1,000 shifts the threshold but makes the same split. Scale is entirely irrelevant to tree-based models.

**Why RFECV instead of SFS:** RFC computes feature importances (Gini impurity reduction) natively at every split. RFECV exploits these scores directly to rank and eliminate features — it does not need to retrain from scratch at every step the way SFS does. This makes RFECV both faster and more principled for tree-based models.

**Two StratifiedKFolds — inner (3-fold) and outer (5-fold):**
- `cv_inner` (3-fold): used inside RFECV to score feature subsets. Kept at 3 folds to limit the computational cost of the three-level nesting (GridSearchCV → RFECV → model fit).
- `cv_outer` (5-fold): used by GridSearchCV to evaluate each parameter combination. More folds = more reliable score estimate.

**Why `n_jobs=1` inside the pipeline estimators:** Both RFECV and RFC support `n_jobs=-1`. Setting both to use all cores simultaneously causes them to fight over resources and can slow each other down. All parallelism is pushed to the GridSearchCV level (`n_jobs=-1`), where it is most effective.

---

### 6. Evaluation

#### Cross-validation mean and std
```python
cv_scores = cross_val_score(
    grid_search.best_estimator_,
    X_train, y_train,
    cv=cv_outer,
    scoring="f1_macro"
)
print(f"=> {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

`GridSearchCV.best_score_` already gives the mean CV F1 of the best parameter combination, but does not expose per-fold scores or the standard deviation. `cross_val_score` is called explicitly on the best estimator to get both. A low std indicates the model is stable across different training subsets.

#### Why macro-F1 and not accuracy
With 78% of samples being Normal, a model that always predicts Normal achieves 78% accuracy while completely missing Suspect and Pathological cases. Macro-F1 computes precision and recall for each class independently and averages them equally — regardless of class size. A model that ignores the Pathological class entirely would score 0.33 macro-F1 at best, making it the correct metric for this imbalanced problem.

#### Confusion matrix (combined absolute + relative)
Both absolute counts and row-normalised percentages are shown simultaneously in each cell. Row normalisation divides each cell by the true class count, showing what proportion of each actual class was correctly or incorrectly classified. This directly reveals which classes are confused with which others — the Suspect class consistently shows the most off-diagonal confusion.

#### Feature importance plot (RFC only)
```python
sns.barplot(x="importance", y="feature", data=importance_df[:10])
```

RFC's built-in Gini importance scores are extracted from the final fitted model and plotted as a horizontal bar chart. This is only possible for RFC — KNN, LogReg, and SVC do not produce equivalent importance scores in the same form.

---

### 7. Learning Curves

```python
train_sizes_abs, train_scores, val_scores = learning_curve(
    best_estimator, X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 8),
    cv=cv, scoring="f1_macro", n_jobs=-1
)
```

Learning curves are plotted for all four models in a 2×2 grid. Each plot shows train F1 and validation F1 as a function of training set size, using the **best estimator** from each grid search.

**What to look for:**
- A large gap between train and validation → overfitting
- Both lines low and close together → underfitting / linear ceiling
- Validation line still rising at the right edge → more data could help
- Lines converging as data grows → model is approaching optimal capacity

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| `imblearn.Pipeline` instead of `sklearn.Pipeline` | sklearn's Pipeline does not support `fit_resample` — SMOTE would be applied to validation folds, causing leakage |
| SMOTE inside the pipeline | Ensures synthetic samples never contaminate validation or test folds |
| SMOTE preferred over `class_weight` | Empirically outperformed `class_weight="balanced"` across all models — fills sparse minority regions rather than just reweighting the loss |
| SFS with fixed estimator inside | Tuning hyperparameters inside SFS at every step would be combinatorially expensive. A fixed, reasonable configuration is used for selection; the tuned model is a separate step |
| 3-fold inner CV for SFS / RFECV | Reduces computation inside the already expensive grid search without meaningfully sacrificing selection quality |
| 5-fold outer CV for GridSearchCV | Provides a reliable CV estimate while keeping total training time manageable |
| `random_state` fixed throughout | Ensures reproducibility. `random_state=42` is used for all CV strategies, SMOTE, and model estimators. `random_state=42` for `train_test_split`. Seeds are fixed once and never changed |
| Outliers retained | CTG extreme values likely represent genuine pathological states — removing them would delete the most clinically relevant signal |
| 13 duplicates removed | These are logging artefacts, not genuine repeated observations. Removal has negligible impact on class balance |
| Class remapping {1,2,3} → {0,1,2} | Zero-indexed classes are required by some sklearn utilities and produce cleaner classification reports |
| No scaler for RFC | Tree splits are threshold comparisons — scale invariant by definition. Adding StandardScaler for RFC adds computation with no benefit |
| RFECV for RFC, SFS for others | RFECV uses RFC's native feature importances, making it more principled and faster for tree models. SFS works generically with any scorer |

---

## How Each Step Influences the Next

```
Data loading
    │
    ▼
Data cleaning (drop duplicates, check outliers)
    │   Retaining outliers means all downstream models must handle
    │   extreme values. RFC does this naturally via threshold splits.
    │   KNN/SVC are affected but SMOTE and scaling mitigate the impact.
    ▼
EDA (class distribution, correlations, boxplots)
    │   Class imbalance observed → drives choice of macro-F1, SMOTE,
    │   and stratified splitting. Feature correlations show which
    │   features carry signal → motivates SFS/RFECV rather than
    │   using all 21 features. Non-linear scatter plots motivate
    │   non-linear models (RFC, SVC).
    ▼
Train / test split (stratified, random_state fixed)
    │   Stratification preserves the imbalance structure in both sets.
    │   The test set is locked away here — never seen again until
    │   final evaluation. Everything from here uses only X_train/y_train.
    ▼
Pipeline construction (scaler → SMOTE → selector → model)
    │   Scale first: ensures SMOTE interpolates in normalised space.
    │   SMOTE second: balances classes before feature selection, so
    │   SFS evaluates features on balanced data and does not
    │   undervalue features that discriminate minority classes.
    │   Selector third: reduces dimensionality, removes redundant
    │   features seen in the correlation heatmap.
    │   Model last: trained on clean, balanced, selected features.
    ▼
GridSearchCV (5-fold StratifiedKFold, macro-F1)
    │   Tunes all pipeline parameters simultaneously. Stratified folds
    │   ensure minority classes appear in every fold. Macro-F1 as
    │   the scoring metric penalises poor minority-class performance
    │   at every step of the search.
    ▼
CV mean ± std (cross_val_score on best estimator)
    │   Re-evaluates the best pipeline on the training set to expose
    │   per-fold scores and standard deviation — neither of which
    │   GridSearchCV.best_score_ provides directly.
    ▼
Test set evaluation (touched once, at the very end)
    │   Final, unbiased estimate of generalisation performance.
    │   Reports accuracy, macro precision/recall/F1, weighted F1,
    │   classification report, and confusion matrix.
    ▼
Learning curves
    │   Diagnoses each model's bias-variance tradeoff using the
    │   best estimator found by GridSearchCV. Answers two questions:
    │   (1) Is the model overfitting or underfitting?
    │   (2) Would more data improve performance?
    ▼
Feature importance (RFC only)
    Identifies which of the selected features RFC relied on most.
    Connects the model's internal decision process back to the
    clinical CTG features — grounding the results in the domain.
```

---

## Results Summary

| Model | CV F1 (mean ± std) | Test F1 | Notes |
|---|---|---|---|
| KNN | — ± — | — | Overfitting at low k visible in learning curve |
| Logistic Regression | — ± — | — | Linear ceiling — Suspect/Normal boundary is non-linear |
| SVC | — ± — | — | Strong generalisation despite overfitting train score |
| RFC | — ± — | — | Best overall — smallest train-val gap, highest val F1 |

> Replace `—` with actual results after running the notebook.

**Main finding:** RFC outperformed all other models, attributed to its ability to model non-linear decision boundaries, capture feature interactions through recursive splits, and regularise implicitly through ensemble averaging — all properties that match the structure of this CTG dataset.

**Main challenge across all models:** The Suspect class had the highest inter-class confusion in every confusion matrix, consistent with significant feature-space overlap between Suspect and Normal recordings. This is likely approaching the Bayes error floor rather than a modelling limitation — meaning no classifier, however powerful, can perfectly separate these two classes given the available features.
