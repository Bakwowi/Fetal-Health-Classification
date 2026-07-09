# Speaker Notes — Fetal Health Classification (15 min, 20 slides)

General pacing: ~45 seconds/slide on average. Slides 13, 14, 18 need a bit more time
(results); slides 2, 4, 8, 11 can go faster (they're mostly visual/self-explanatory).

---

**Slide 1 — Title**
Good [morning/afternoon], we're Group 16. Today we'll present our project on classifying
fetal health from CTG recordings using four different machine learning classifiers.

**Slide 2 — Motivation**
CTG is the standard tool doctors use to monitor a baby's heart rate during pregnancy, but
it's still read manually, and different doctors often disagree on what they see. Our goal
was to build a classifier that automatically labels a recording as Normal, Suspect, or
Pathological using 21 extracted features. The key challenge: mistakes matter — missing a
real problem is dangerous, but false alarms cause unnecessary interventions like C-sections.

**Slide 3 — Data Overview**
We used the public Kaggle Fetal Health dataset: 2,126 recordings, 21 features, 3 classes,
no missing values, only 13 duplicate rows. Features fall into a few groups: heart rate and
movement signals, deceleration counts, variability measures, and histogram statistics.

**Slide 4 — Data Sample**
Just a quick look at what the raw data looks like — each row is one CTG recording, and the
last column is our target, fetal_health.

**Slide 5 — Class Distribution**
This is the core challenge of the project: the data is heavily imbalanced — 1,646 Normal
vs. only 175 Pathological cases. This is why we chose macro-F1 instead of accuracy, why we
used a stratified train/test split, and why we tested oversampling methods.

**Slide 6 — Feature Exploration**
Here you can see Pathological cases (yellow) are fairly well separated from the rest, but
Suspect cases (red) overlap heavily with Normal — this is the main source of difficulty in
the whole project, and it shows up again later in our confusion matrix.

**Slide 7 — Feature-Target Correlations**
Some features correlate strongly with the target — like prolonged decelerations and
abnormal short-term variability — while others correlate very weakly. This motivated doing
feature selection rather than using all 21 features for every model.

**Slide 8 — Pipeline Architecture**
This is our overall pipeline: clean the data, split into train/test, scale the features,
then do model-specific feature selection, then hyperparameter tuning with GridSearchCV. We
removed 13 duplicates but kept all outliers, since extreme values can be real clinical
signals rather than errors.

**Slide 9 — Stratified Train/Test Split**
We split 80/20, stratified by class so both sets keep the same class proportions. Important:
the test set was only touched once, at the very end, after all tuning was done — to avoid
any bias in our final performance numbers.

**Slide 10 — Classifiers Selected**
We compared two base classifiers — KNN and Logistic Regression — with two more advanced
ones — SVC and Random Forest. For each, we tuned the hyperparameters with the biggest known
effect on performance, while keeping the search space manageable computationally.

**Slide 11 — Feature Selection**
For KNN, Logistic Regression, and SVC we used forward Sequential Feature Selection, testing
5, 10, 15, or 20 features. For Random Forest we used RFECV instead, since it can use the
model's own built-in feature importance scores directly, which is faster.

**Slide 12 — Cross-Validation & Hyperparameter Tuning**
We used 5-fold stratified cross-validation with GridSearchCV, macro-F1 as the scoring
metric so all three classes count equally. We also compared two ways of handling class
imbalance — SMOTE and class-weighting — and SMOTE consistently performed better across all
four models.

**Slide 13 — Model Comparison (Results)**
[Take your time here] This table summarizes all four models. Random Forest had the best
cross-validation score (0.896) and the second-best test score (0.887), just slightly below
SVC's test score (0.883). Logistic Regression performed clearly worst, since it can only
model linear boundaries. KNN and SVC landed in between.

**Slide 14 — Best Model (Confusion Matrix)**
This is the confusion matrix for our best model, Random Forest, on the test set. Normal and
Pathological are both classified very accurately — 97.6% and 94.3%. But Suspect is the weak
point: only 70.7% correctly classified, confused with both other classes. This matches what
we saw earlier in the feature overlap plot.

**Slide 15 — Why RFC Outperformed the Others**
A few reasons: Random Forest can model non-linear boundaries naturally, it captures feature
interactions automatically, averaging across many trees reduces error, it doesn't assume any
particular data distribution, and it's not distorted by outliers the way distance-based
models like KNN and SVC can be.

**Slide 16 — Feature Importances**
These are the top features Random Forest relied on most — abnormal short-term variability
and percentage of abnormal long-term variability were the two strongest predictors, which
makes clinical sense since these directly reflect fetal heart rate stability.

**Slide 17 — Discussion**
To summarize what we learned: the advanced classifiers beat the base ones, confirming the
decision boundary is non-linear. Suspect was consistently the hardest class across all four
models. We also saw KNN overfit badly at k=1, which is a clear, concrete example of why
hyperparameter tuning matters. And placing SMOTE inside the cross-validation loop — not
before splitting — was essential to avoid leaking synthetic samples into validation folds.

**Slide 18 — Learning Curves**
These curves show validation performance still rising for all four models — none have fully
plateaued. This suggests more real data, especially more Pathological cases, would likely
improve performance further, particularly for the minority classes.

**Slide 19 — Summary**
To wrap up: 2,126 CTG recordings, 21 features, 3 imbalanced classes, macro-F1 as our metric.
We cleaned duplicates, selected features per model, and tuned hyperparameters with 5-fold CV.
Random Forest was our best model with a test F1 of 0.887, and the Suspect class remains the
main open challenge.

**Slide 20 — Thank You**
Thank you — happy to take any questions.

---

## Optional split between two presenters
- **Presenter A:** Slides 1–9 (motivation, data, pipeline) — roughly 6–7 minutes
- **Presenter B:** Slides 10–20 (methods, results, discussion, summary) — roughly 8–9 minutes

## Timing checkpoint
If you're running long, the easiest slides to compress are 4, 6, 7, and 11 — each can be
cut to one sentence without losing the argument. Don't compress 13 or 14 — those carry the
core results.
