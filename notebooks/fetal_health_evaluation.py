import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
    precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector, RFECV

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


# class names used throughout (0=Normal, 1=Suspect, 2=Pathological)
CLASS_NAMES = ["Normal", "Suspect", "Pathological"]


# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & PREPARATION
# ═════════════════════════════════════════════════════════════════════════════

def load_data(csv_path):
    """Load the dataset from a CSV file and drop rows with missing values."""
    df = pd.read_csv(csv_path)

    print("Number of rows before dropping missing values:", len(df))
    df = df.dropna()
    print("Number of rows after dropping missing values:", len(df))

    print("\nMissing values per column:")
    print(df.isnull().sum())

    return df


def split_features_target(df, target_col="fetal_health", remap_classes=True):
    """
    Split a dataframe into features (X) and target (y).

    If remap_classes=True, the classes {1, 2, 3} are mapped to {0, 1, 2}
    for better readability (0=Normal, 1=Suspect, 2=Pathological).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    if remap_classes:
        y = y.replace({1: 0, 2: 1, 3: 2})

    print("Feature matrix shape:", X.shape)
    print("Target vector shape :", y.shape)
    print("\nClass distribution:")
    print(y.value_counts())

    return X, y


def split_train_test(X, y, test_size=0.2, random_state=20):
    """
    Split into train and test sets.
    Stratify by target to maintain the class distribution.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print("Train samples:", len(X_train), " Test samples:", len(X_test))
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE to the training data to handle class imbalance.
    NOTE: only ever apply this to the training set, never the test set.
    """
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print("Original training set class distribution:")
    print(y_train.value_counts())
    print("\nResampled training set class distribution:")
    print(y_res.value_counts())

    return X_res, y_res


# ═════════════════════════════════════════════════════════════════════════════
# 2. DATA DISTRIBUTION / VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════
# Each plot is its own function so you can call only the ones you need.

def plot_target_distribution(y, class_names=None):
    """Bar chart showing how many samples are in each target class."""
    plt.figure(figsize=(8, 6))
    counts = y.value_counts().sort_index()
    plt.bar(counts.index, counts.values)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Distribution of Target Classes")
    if class_names is not None:
        plt.xticks(ticks=list(counts.index), labels=class_names)
    plt.show()


def plot_feature_histograms(X, bins=30):
    """
    Plot a histogram for every feature so you can see the distribution
    (shape, skew, outliers) of each one at a glance.
    """
    n_features = X.shape[1]
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))

    plt.figure(figsize=(n_cols * 4, n_rows * 3))
    for i, col in enumerate(X.columns):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.hist(X[col], bins=bins, color="steelblue", edgecolor="white")
        plt.title(col, fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_feature_boxplots(X):
    """
    Boxplot of every feature (helps spot outliers and compare spreads).
    Features are standardised first so they share a common scale.
    """
    X_scaled = pd.DataFrame(
        StandardScaler().fit_transform(X),
        columns=X.columns
    )

    plt.figure(figsize=(14, 6))
    sns.boxplot(data=X_scaled, orient="h")
    plt.title("Feature Distributions (standardised)")
    plt.xlabel("Standardised value")
    plt.show()


def plot_feature_by_class(df, feature, target_col="fetal_health"):
    """Boxplot of a single feature split by target class."""
    plt.figure(figsize=(10, 8))
    sns.boxplot(data=df, x=target_col, y=feature)
    plt.title(f"Boxplot of {feature} by {target_col}")
    plt.show()


def plot_scatter(X, y, feature_x, feature_y=None):
    """
    Scatter plot.
    - If feature_y is given: feature_x vs feature_y, coloured by class.
    - If feature_y is None : feature_x vs the target class.
    """
    plt.figure(figsize=(8, 6))
    if feature_y is None:
        plt.scatter(X[feature_x], y, c=y, cmap="plasma", alpha=0.5)
        plt.xlabel(feature_x)
        plt.ylabel("Class")
        plt.title(f"{feature_x} vs Class")
    else:
        plt.scatter(X[feature_x], X[feature_y], c=y, cmap="plasma", alpha=0.5)
        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
        plt.title(f"{feature_x} vs {feature_y}")
    plt.show()


def plot_correlation_with_target(X, y):
    """Heatmap of how each feature correlates with the target."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(X.corrwith(y).to_frame(), annot=True, cmap="Blues")
    plt.title("Correlation of Features with Target Variable")
    plt.show()


def plot_correlation_matrix(X):
    """Heatmap of how the features correlate with each other."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(X.corr(), annot=True, cmap="Blues")
    plt.title("Correlation Heatmap of Features with Each Other")
    plt.show()


def plot_train_test_distribution(y_train, y_test, class_names=None):
    """Side-by-side bar charts of the class distribution in train vs test."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    tr = y_train.value_counts().sort_index()
    te = y_test.value_counts().sort_index()

    axes[0].bar(tr.index, tr.values)
    axes[0].set_title("Train Set Distribution")
    axes[1].bar(te.index, te.values)
    axes[1].set_title("Test Set Distribution")

    if class_names is not None:
        for ax, counts in zip(axes, [tr, te]):
            ax.set_xticks(list(counts.index))
            ax.set_xticklabels(class_names)

    plt.tight_layout()
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# 3. MODEL TRAINING
# ═════════════════════════════════════════════════════════════════════════════


def _build_grid_search(selector_estimator, model_estimator, param_grid,
                       use_scaler=True, sfs_n_features=10, sfs_cv=3,
                       outer_cv_splits=5, random_state=42):
    """
    Internal helper: builds a (scaler -> SFS selector -> model) pipeline
    and wraps it in a GridSearchCV. Returns the (unfitted) GridSearchCV.
    """
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))

    steps.append((
        "selector",
        SequentialFeatureSelector(
            selector_estimator,
            n_features_to_select=sfs_n_features,
            direction="forward",
            scoring="f1_macro",
            cv=sfs_cv            # inner CV for SFS (keep low for speed)
        )
    ))
    steps.append(("model", model_estimator))

    pipeline = Pipeline(steps)

    cv_outer = StratifiedKFold(
        n_splits=outer_cv_splits, shuffle=True, random_state=random_state
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_outer,
        scoring="f1_macro",
        n_jobs=-1,           # use all CPU cores
        verbose=2,           # print progress
        error_score="raise"  # surface errors immediately
    )
    return grid_search


def _report_grid_search(grid_search, X_test, y_test, feature_names):
    """Internal helper: prints best params, scores and selected features."""
    print(f"Best CV F1:  {grid_search.best_score_:.4f}")
    print(f"Best Params: {grid_search.best_params_}")

    y_pred = grid_search.predict(X_test)
    print(f"Test F1:     {f1_score(y_test, y_pred, average='macro'):.4f}")

    best_selector = grid_search.best_estimator_.named_steps["selector"]
    selected = feature_names[best_selector.get_support()]
    print(f"\nSelected features ({len(selected)}):")
    print(selected.tolist())

    return grid_search


# ── KNN ──────────────────────────────────────────────────────────────────────
def train_knn(X_train, y_train, X_test=None, y_test=None):
    """Train a K-Nearest-Neighbors classifier with feature selection + grid search."""
    knn_for_selection = KNeighborsClassifier(n_neighbors=5)

    param_grid = {
        "selector__n_features_to_select": [5, 10, 15, 20],
        "model__n_neighbors":             [3, 5, 7, 11, 15],
        "model__weights":                 ["uniform", "distance"],
        "model__metric":                  ["euclidean", "manhattan"]
    }

    grid_search = _build_grid_search(
        selector_estimator=knn_for_selection,
        model_estimator=KNeighborsClassifier(),
        param_grid=param_grid,
        use_scaler=True
    )

    grid_search.fit(X_train, y_train)
    if X_test is not None:
        _report_grid_search(grid_search, X_test, y_test, X_train.columns)
    return grid_search


# ── GaussianNB ───────────────────────────────────────────────────────────────
def train_gaussian_nb(X_train, y_train, X_test=None, y_test=None):
    """Train a Gaussian Naive Bayes classifier with feature selection + grid search."""
    nb_for_selection = GaussianNB()

    # GaussianNB has very few hyperparameters; var_smoothing is the useful one.
    param_grid = {
        "selector__n_features_to_select": [5, 10, 15, 20],
        "model__var_smoothing":           [1e-9, 1e-8, 1e-7, 1e-6]
    }

    grid_search = _build_grid_search(
        selector_estimator=nb_for_selection,
        model_estimator=GaussianNB(),
        param_grid=param_grid,
        use_scaler=True
    )

    grid_search.fit(X_train, y_train)
    if X_test is not None:
        _report_grid_search(grid_search, X_test, y_test, X_train.columns)
    return grid_search


# ── LDA ──────────────────────────────────────────────────────────────────────
def train_lda(X_train, y_train, X_test=None, y_test=None):
    """Train a Linear Discriminant Analysis classifier with feature selection + grid search."""
    lda_for_selection = LinearDiscriminantAnalysis()

    # 'shrinkage' only works with the 'lsqr' or 'eigen' solver (not 'svd').
    param_grid = [
        {
            "selector__n_features_to_select": [5, 10, 15, 20],
            "model__solver":    ["svd"]
        },
        {
            "selector__n_features_to_select": [5, 10, 15, 20],
            "model__solver":    ["lsqr", "eigen"],
            "model__shrinkage": [None, "auto", 0.1, 0.5]
        }
    ]

    grid_search = _build_grid_search(
        selector_estimator=lda_for_selection,
        model_estimator=LinearDiscriminantAnalysis(),
        param_grid=param_grid,
        use_scaler=True
    )

    grid_search.fit(X_train, y_train)
    if X_test is not None:
        _report_grid_search(grid_search, X_test, y_test, X_train.columns)
    return grid_search


# ── Logistic Regression ──────────────────────────────────────────────────────
def train_logreg(X_train, y_train, X_test=None, y_test=None):
    """Train a Logistic Regression classifier with feature selection + grid search."""
    logreg_for_selection = LogisticRegression(max_iter=1000)

    param_grid = {
        "selector__n_features_to_select": [5, 10, 15, 20],
        "model__C":            [0.01, 0.1, 1, 10, 100],
        "model__penalty":      ["l2"],
        "model__solver":       ["lbfgs"],
        "model__class_weight": [None, "balanced"]
    }

    grid_search = _build_grid_search(
        selector_estimator=logreg_for_selection,
        model_estimator=LogisticRegression(max_iter=1000),
        param_grid=param_grid,
        use_scaler=True
    )

    grid_search.fit(X_train, y_train)
    if X_test is not None:
        _report_grid_search(grid_search, X_test, y_test, X_train.columns)
    return grid_search


# ── SVC ──────────────────────────────────────────────────────────────────────
def train_svc(X_train, y_train, X_test=None, y_test=None):
    """Train a Support Vector Classifier with feature selection + grid search."""
    svc_for_selection = SVC(kernel="rbf", C=1.0, gamma="scale")

    param_grid = [
        {
            "selector__n_features_to_select": [5, 10, 15, 20],
            "model__kernel": ["rbf", "poly", "sigmoid"],
            "model__C":      [0.1, 1, 10, 100],
            "model__gamma":  ["scale", "auto", 0.01, 0.1]
        },
        {
            "selector__n_features_to_select": [5, 10, 15, 20],
            "model__kernel": ["linear"],
            "model__C":      [0.1, 1, 10, 100]
        }
    ]

    grid_search = _build_grid_search(
        selector_estimator=svc_for_selection,
        model_estimator=SVC(probability=True),
        param_grid=param_grid,
        use_scaler=True
    )

    grid_search.fit(X_train, y_train)
    if X_test is not None:
        _report_grid_search(grid_search, X_test, y_test, X_train.columns)
    return grid_search


# ── Random Forest (uses RFECV native selector, no scaler needed) ─────────────
def train_random_forest(X_train, y_train, X_test=None, y_test=None,
                        random_state=42):
    """
    Train a Random Forest classifier using RFECV (the tree's own feature
    importances) for feature selection + grid search.
    RFC does not need scaling, so no scaler is used here.
    """
    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    pipeline = Pipeline([
        ("selector", RFECV(
            estimator=RandomForestClassifier(
                n_estimators=100, random_state=random_state, n_jobs=1
            ),
            step=1,
            cv=cv_inner,
            scoring="f1_macro",
            min_features_to_select=5,
            n_jobs=-1
        )),
        ("model", RandomForestClassifier(random_state=random_state, n_jobs=1))
    ])

    param_grid = {
        "selector__min_features_to_select": [5, 10, 15],
        "selector__step":                   [1, 2, 3],
        "model__n_estimators":              [100, 200, 300],
        "model__max_depth":                 [None, 5, 10, 20],
        "model__min_samples_split":         [2, 5, 10],
        "model__max_features":              ["sqrt", "log2"],
        "model__class_weight":              ["balanced", None]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_outer,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=2,
        error_score="raise"
    )

    grid_search.fit(X_train, y_train)
    if X_test is not None:
        _report_grid_search(grid_search, X_test, y_test, X_train.columns)
    return grid_search


# ── Registry: makes it easy to loop over models or add/remove them ───────────
MODEL_TRAINERS = {
    "KNN":                train_knn,
    "GaussianNB":         train_gaussian_nb,
    "LDA":                train_lda,
    "LogisticRegression": train_logreg,
    "SVC":                train_svc,
    "RandomForest":       train_random_forest,
}


def train_all(X_train, y_train, X_test, y_test, models=None):
    """
    Train several models in one go and return a dict of fitted GridSearchCVs.
    Pass `models=["KNN", "GaussianNB"]` to run only a subset.
    """
    if models is None:
        models = list(MODEL_TRAINERS.keys())

    results = {}
    for name in models:
        print("\n" + "=" * 78)
        print(f"  Training: {name}")
        print("=" * 78)
        results[name] = MODEL_TRAINERS[name](X_train, y_train, X_test, y_test)
    return results


# ═════════════════════════════════════════════════════════════════════════════
# 4. CROSS-VALIDATION SCORE (mean ± std of the F1)
# ═════════════════════════════════════════════════════════════════════════════

def show_cv_f1(estimator, X_train, y_train, cv_splits=5, random_state=42):
    """
    Print the mean and standard deviation of the macro-F1 across CV folds.

    `estimator` can be a plain pipeline OR a fitted GridSearchCV
    (in which case its best_estimator_ is used).
    """
    if hasattr(estimator, "best_estimator_"):
        estimator = estimator.best_estimator_

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring="f1_macro")

    print(f"CV macro-F1 per fold: {np.round(scores, 4)}")
    print(f"CV macro-F1 mean:     {scores.mean():.4f}")
    print(f"CV macro-F1 std:      {scores.std():.4f}")
    print(f"=> {scores.mean():.4f} ± {scores.std():.4f}")

    return scores


# ═════════════════════════════════════════════════════════════════════════════
# 5. EVALUATION ON THE TEST SET
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test, class_names=CLASS_NAMES):
    """
    Full evaluation on the held-out test set:
    accuracy, macro precision/recall/F1, weighted F1, classification
    report, and a confusion matrix plot.
    `model` can be a fitted estimator or a fitted GridSearchCV.
    """
    y_pred = model.predict(X_test)

    accuracy        = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro")
    recall_macro    = recall_score(y_test, y_pred, average="macro")
    f1_macro        = f1_score(y_test, y_pred, average="macro")
    f1_weighted     = f1_score(y_test, y_pred, average="weighted")

    print("Accuracy        :", round(accuracy, 4))
    print("Macro Precision :", round(precision_macro, 4))
    print("Macro Recall    :", round(recall_macro, 4))
    print("Macro F1        :", round(f1_macro, 4))
    print("Weighted F1     :", round(f1_weighted, 4))

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=list(class_names)))

    plot_confusion(y_test, y_pred, class_names)

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def plot_confusion(y_test, y_pred, class_names=CLASS_NAMES):
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=list(class_names),
        yticklabels=list(class_names)
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def compare_models(results, X_test, y_test):
    """
    Given the dict returned by train_all(), print a small table comparing
    the test macro-F1 of every model so you can see which won.
    """
    rows = []
    for name, gs in results.items():
        y_pred = gs.predict(X_test)
        rows.append({
            "model":     name,
            "cv_f1":     round(gs.best_score_, 4),
            "test_f1":   round(f1_score(y_test, y_pred, average="macro"), 4),
            "test_acc":  round(accuracy_score(y_test, y_pred), 4),
        })

    table = pd.DataFrame(rows).sort_values("test_f1", ascending=False)
    print(table.to_string(index=False))
    return table


# ═════════════════════════════════════════════════════════════════════════════
# 6. MAIN WORKFLOW
# ═════════════════════════════════════════════════════════════════════════════
# This runs the full pipeline when you execute `python fetal_health_evaluation.py`.
# Comment out the plots / models you don't want.

def main():
    # ── Update this path to point to your CSV ─────────────────────────────────
    CSV_PATH = r'C:\Users\Bakwowi Junior\Documents\school-documents\4th-semester-SS-26\Machine Learning\ml-evaluation-project\datasets\fetal_health.csv'

    # ── 1. Load and prepare ───────────────────────────────────────────────────
    df = load_data(CSV_PATH)
    X, y = split_features_target(df, target_col="fetal_health", remap_classes=True)

    # ── 2. View the distribution of the data ──────────────────────────────────
    plot_target_distribution(y, class_names=CLASS_NAMES)
    plot_feature_histograms(X)
    plot_feature_boxplots(X)
    plot_feature_by_class(df, feature="uterine_contractions", target_col="fetal_health")
    plot_scatter(X, y, feature_x="baseline value")
    plot_scatter(X, y, feature_x="baseline value", feature_y="uterine_contractions")
    plot_correlation_with_target(X, y)
    plot_correlation_matrix(X)

    # ── 3. Train / test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2, random_state=20)
    plot_train_test_distribution(y_train, y_test, class_names=CLASS_NAMES)

    # Optional: SMOTE on the training set only
    # X_train, y_train = apply_smote(X_train, y_train)

    # ── 4. Train models ───────────────────────────────────────────────────────
    knn_gs    = train_knn(X_train, y_train, X_test, y_test)
    nb_gs     = train_gaussian_nb(X_train, y_train, X_test, y_test)
    lda_gs    = train_lda(X_train, y_train, X_test, y_test)
    logreg_gs = train_logreg(X_train, y_train, X_test, y_test)
    # svc_gs  = train_svc(X_train, y_train, X_test, y_test)            # slower
    # rf_gs   = train_random_forest(X_train, y_train, X_test, y_test)  # slower

    # ── 5. Cross-validation F1 (mean ± std) ───────────────────────────────────
    print("\n--- KNN CV F1 ---");    show_cv_f1(knn_gs,    X_train, y_train)
    print("\n--- NB  CV F1 ---");    show_cv_f1(nb_gs,     X_train, y_train)
    print("\n--- LDA CV F1 ---");    show_cv_f1(lda_gs,    X_train, y_train)
    print("\n--- LogReg CV F1 ---"); show_cv_f1(logreg_gs, X_train, y_train)

    # ── 6. Detailed evaluation of one model ───────────────────────────────────
    print("\n--- Detailed evaluation (KNN) ---")
    evaluate_model(knn_gs, X_test, y_test, class_names=CLASS_NAMES)

    # ── 7. Compare all trained models ─────────────────────────────────────────
    print("\n--- Model comparison ---")
    trained = {
        "KNN":                knn_gs,
        "GaussianNB":         nb_gs,
        "LDA":                lda_gs,
        "LogisticRegression": logreg_gs,
        # "SVC":              svc_gs,
        # "RandomForest":     rf_gs,
    }
    compare_models(trained, X_test, y_test)


if __name__ == "__main__":
    main()
