# Machine Learning — Repetition Questions: Answers

> Prepared as a study guide for project presentations. Where a question depends on exact
> phrasing used in your specific lecture slides, a note is added — double-check terminology
> against Prof. Mayer's slides, since different courses phrase some of these definitions slightly
> differently (especially "two foundations of ML" and "3 classifier categorizations").

---

## 2. ML Overview

**What are the two foundations of machine learning?**
Statistics/mathematics (probability theory, statistical inference) and computer science
(algorithms, optimization, computational efficiency). ML sits at the intersection: it uses
statistical theory to make claims about data and generalization, and computational methods to
actually fit models to data at scale.

**Name the 3 main ML problem classes.**
1. Supervised learning (learn a mapping from inputs to known outputs/labels)
2. Unsupervised learning (find structure in unlabeled data, e.g. clustering)
3. Reinforcement learning (an agent learns by interacting with an environment via rewards)

**Formally define supervised learning.**
Given a training set of pairs D = {(x₁, y₁), …, (xₙ, yₙ)} drawn i.i.d. from an unknown joint
distribution P(X, Y), the goal is to learn a function f: X → Y (a hypothesis) from a hypothesis
space H that minimizes the expected loss (risk) R(f) = E₍X,Y₎[L(Y, f(X))] for a chosen loss
function L, so that f generalizes well to new, unseen data drawn from the same distribution.

**Formal difference between classification and regression.**
The difference is the type of the output/target variable Y:
- Classification: Y takes values in a finite, discrete, unordered (categorical) set.
- Regression: Y takes values in a continuous space (typically ℝ or a subset of it).

**Most used error measure for classification (formal definition).**
Misclassification error rate (0–1 loss):
L(y, ŷ) = 1[y ≠ ŷ], and the empirical error is
Err = (1/n) Σᵢ 1[yᵢ ≠ ŷᵢ]

**Most used error measure for regression (formal definition).**
Mean Squared Error (MSE):
MSE = (1/n) Σᵢ (yᵢ − ŷᵢ)²

---

## 3. First Classifiers and ML Experiments

### First Classifiers

**What are the resources in an ML project?**
Data, computational hardware (CPU/GPU/memory), time, domain expertise/human labor, and
software/algorithms/tools.

**Name the (broadly generalized) steps of an ML project.**
1. Problem definition
2. Data collection
3. Data exploration & preprocessing (cleaning, feature engineering)
4. Model selection
5. Training
6. Evaluation
7. Deployment / monitoring

**What do you usually try to optimize during an ML evaluation?**
The generalization performance — i.e., the expected performance (according to a chosen error
measure) on unseen data, not on the training data.

**What are the steps of a single ML experiment?**
1. Split data into training and test set
2. Fix a set of hyperparameters
3. Train the model on the training set
4. Evaluate the trained model on the test set
5. Record the resulting performance metric

**What is the relation between an ML evaluation and an ML experiment?**
An ML evaluation typically consists of many individual ML experiments (e.g. across different
hyperparameter settings, different folds, or different random seeds), whose results are
aggregated (e.g. averaged) to get a robust performance estimate. A single experiment is one
train/test run; the evaluation is the overall procedure built from many such experiments.

**What are general properties of good data?**
Representative of the true underlying distribution, sufficiently large, accurate/low-noise
labels, relevant features, free of leakage between train and test, reasonably balanced across
classes (or the imbalance is explicitly handled), consistent formatting, and up to date.

**Name 3 possible classifier categorizations.**
1. Parametric vs. non-parametric
2. Generative vs. discriminative
3. Linear vs. non-linear (alternatively: eager learners vs. lazy learners)

**Formally explain the Bayes classifier and write down the formula used.**
The Bayes classifier assigns an observation x to the class k that has the highest posterior
probability given x:

ŷ(x) = argmax_k P(Y = k | X = x) = argmax_k [ P(X = x | Y = k) · P(Y = k) ] / P(X = x)

Since P(X = x) doesn't depend on k, this simplifies to argmax_k P(X = x | Y = k) · P(Y = k). It
is the classifier that minimizes the expected 0-1 loss (misclassification error) and is
therefore theoretically optimal.

**Why do we need other classifiers if the Bayes classifier is guaranteed to be optimal?**
Because the Bayes classifier requires knowledge of the true class-conditional distributions
P(X|Y) and class priors P(Y), which are unknown in practice. We only have a finite sample and
must estimate these quantities (or approximate the decision rule directly), which introduces
estimation error — this is exactly what other classifiers (kNN, logistic regression, trees,
etc.) attempt to do well.

**Give a definition of the term "decision boundary".**
The decision boundary is the surface in feature space that separates regions assigned to
different classes by a classifier — formally, for a two-class problem, the set of points x
where the classifier is indifferent between the classes, e.g. {x : P(Y=1|X=x) = P(Y=2|X=x)}.

**How does the kNN classifier work (informal description)?**
To classify a new point, find the k training points that are closest to it (by some distance
metric, e.g. Euclidean distance), and assign the class that is the majority among those k
neighbors.

**How does the training procedure of a kNN classifier work?**
kNN is a "lazy learner" — there is no explicit training/fitting phase beyond storing the
training data. All the computation (finding neighbors, voting) happens at prediction time.

**What are the hyperparameters of the kNN classifier?**
- k (the number of neighbors)
- The distance metric (Euclidean, Manhattan, Minkowski, etc.)
- The weighting scheme (uniform vote vs. distance-weighted vote)

### ML Experiments

**Why do we need a separation between train and test data?**
To get an unbiased estimate of how well the model generalizes to unseen data. If you evaluate
on the same data used for training, the estimate is overly optimistic because the model may
have simply memorized the training data.

**How is the term "overfitting" defined?**
A model overfits when it fits the training data too closely — including its noise and
idiosyncrasies — such that it achieves low training error but high error on unseen (test) data;
i.e., it fails to generalize.

**What do we have to consider when we separate train and test data?**
- The split should be random and representative of the overall distribution
- Stratification (preserving class proportions) if classes are imbalanced
- No data leakage (e.g. preprocessing/scaling fit only on training data, no shared duplicates)
- Both sets need to be large enough to be statistically meaningful
- For time series, respect temporal order (no future data in training)

**How can we check that the training data is large enough?**
By plotting a learning curve: model performance (train/validation) as a function of training
set size. If performance plateaus as more data is added, the training set is likely sufficient;
if it's still improving, more data would help.

**How can we check if the test data is large enough?**
By checking the variance/confidence interval of the test performance estimate — e.g. via
repeated resampling or cross-validation — and seeing whether the estimate is stable. If the
confidence interval is too wide, the test set is too small to draw reliable conclusions.

**What is the difference between test and validation data? What do we use these datasets for?**
- Validation data is used during model development — for hyperparameter tuning and model
  selection — and can be used repeatedly.
- Test data is held out and used only once, at the very end, to get a final, unbiased estimate
  of the chosen model's generalization performance.

**Give our definition of "parameters" and "hyperparameters".**
- Parameters are the internal values a model learns automatically from the training data during
  fitting (e.g. the weights/coefficients of a logistic regression model).
- Hyperparameters are settings that are fixed before training and control the learning process
  or model complexity (e.g. k in kNN, C in SVM); they are typically chosen via model selection
  on a validation set rather than learned directly from the training objective.

**What are features?**
Features are the measurable, individual input variables (predictors/attributes) used to
describe each observation and that the model uses to make its prediction.

**What is a common trap in feature normalization?**
Fitting the normalizer/scaler (e.g. computing mean and standard deviation for standardization)
on the entire dataset — including the test set — rather than fitting it only on the training
data. This causes data leakage and gives an overly optimistic performance estimate.

**How is accuracy defined and in which situations can you use it?**
Accuracy = (number of correct predictions) / (total number of predictions) =
(TP + TN) / (TP + TN + FP + FN). It's appropriate for roughly balanced class distributions; it
is misleading under class imbalance, since a trivial classifier that always predicts the
majority class can achieve high accuracy.

**Define Specificity, Sensitivity, Recall, Precision.**
For a 2-class confusion matrix with TP, TN, FP, FN:
- Sensitivity (= Recall = TPR) = TP / (TP + FN) — proportion of actual positives correctly identified
- Specificity (= TNR) = TN / (TN + FP) — proportion of actual negatives correctly identified
- Precision (= PPV) = TP / (TP + FP) — proportion of predicted positives that are correct

**What is an easy-to-interpret single number quality measure? What is the most meaningful single number quality measure?**
- Easy to interpret: accuracy (simple percentage correct).
- Most meaningful (especially under imbalance / when both error types matter): F1-score
  (harmonic mean of precision and recall) or AUC-ROC (threshold-independent measure of ranking
  quality). AUC-ROC is often argued to be more meaningful because it doesn't depend on choosing
  a single decision threshold.

**Name at least 2 criticisms of the F1 score.**
1. It ignores true negatives entirely, so it can be uninformative (or misleading) for problems
   where the negative class matters.
2. It weights precision and recall equally, which may not reflect the actual costs/priorities
   of a specific application (a weighted Fβ score would be needed instead).
   (Additional criticism: it is threshold-dependent and doesn't summarize performance across
   the whole range of decision thresholds the way AUC-ROC does.)

**What is the best/worst value of the auROC?**
Best = 1.0 (perfect ranking/separation of classes). Worst (meaningful worst) = 0.5, which
corresponds to random guessing. An AUC below 0.5 indicates a systematically inverted
classifier (worse than random), and 0.0 would be perfectly wrong.

---

## 4. Regression

**Write down the simple linear regression model in a formal way.**
Y = β₀ + β₁X + ε, where ε ~ N(0, σ²) is an i.i.d. error term independent of X.

**Name the steps to compute the linear regression parameter estimation formulas.**
1. Define the Residual Sum of Squares: RSS(β) = Σᵢ (yᵢ − ŷᵢ)² = Σᵢ (yᵢ − β₀ − β₁xᵢ)²
2. Take the partial derivatives of RSS with respect to each parameter (β₀, β₁, or the full
   vector β for multiple regression)
3. Set the derivatives to zero (first-order optimality conditions)
4. Solve the resulting system of equations ("normal equations")
5. This yields the closed-form OLS estimator: β̂ = (XᵀX)⁻¹Xᵀy (matrix form for ordinary
   multiple regression)

**How is the RSE computed?**
Residual Standard Error: RSE = √(RSS / (n − p − 1)), where p is the number of predictors
(p = 1 for simple linear regression), and RSS = Σᵢ(yᵢ − ŷᵢ)².

**How is the R² measure computed?**
R² = 1 − RSS/TSS, where TSS = Σᵢ(yᵢ − ȳ)² is the total sum of squares.

**What advantages does RSE have compared to R² and vice versa?**
- RSE is measured in the units of the response variable Y, so it's directly interpretable as a
  "typical" prediction error in real, meaningful units — but that also makes it hard to compare
  across different datasets/response scales.
- R² is unit-free, always between 0 and 1 (for the training fit), and therefore easy to compare
  across different models/datasets — but it doesn't tell you whether the absolute error is
  acceptable for the application, and it can be inflated by simply adding more predictors.

**What is the relation between a kNN classifier and a kNN regressor?**
Both use the same core idea — find the k nearest neighbors of a query point. The classifier
predicts the majority class among the neighbors; the regressor predicts the (typically
unweighted) average of the neighbors' target values.

**Give at least 2 general rules when to prefer a parametric or a non-parametric model.**
- Prefer a parametric model when you have strong prior knowledge/assumptions about the
  functional form, a small amount of data, and/or need an interpretable model.
- Prefer a non-parametric model when the true relationship is complex/unknown, you have a
  large amount of data (non-parametric methods need more data to work well), and flexibility
  matters more than interpretability.

---

## 5. Basic Classifiers

### Logistic Regression

**Give two reasons why we can't use linear regression for classification.**
1. Linear regression's output is unbounded (−∞, ∞), so it can't be directly interpreted as a
   probability that must lie in [0, 1].
2. For more than 2 classes, encoding classes as numbers (e.g. 1, 2, 3) imposes an artificial
   ordering and equal-spacing assumption between classes that generally doesn't correspond to
   any real relationship between them.

**State the logistic function. How does it look like? Make a sketch.**
σ(z) = 1 / (1 + e^(−z))

It is an S-shaped ("sigmoid") curve: it approaches 0 as z → −∞, approaches 1 as z → +∞, and
passes through 0.5 at z = 0, monotonically increasing throughout.

```
1 |                    _____
  |                 __/
  |               _/
0.5|            _/
  |          _/
  |      ___/
0 |_____/
  +---------------------------
      -6  -3   0   3   6   (z)
```

**There are two ways to extend logistic regression to multiple classes. Which?**
1. One-vs-Rest (OvR / one-vs-all): train K separate binary logistic regression models, each
   distinguishing one class from all others, and pick the class with the highest predicted
   probability.
2. Multinomial (softmax) logistic regression: directly model all K class probabilities jointly
   using the softmax function, trained with a single multinomial objective.

**Why does logistic regression have so many function parameters in your favorite ML library?**
Because library implementations (e.g. scikit-learn) expose many configuration options beyond
just the learned coefficients — e.g. the regularization type and strength (penalty, C), the
solver algorithm, convergence tolerance, maximum iterations, class weighting for imbalance, and
the multiclass strategy (OvR vs. multinomial) — all of which affect how the underlying
coefficients are fit.

### Bayesian Classifiers

**Write down the Bayes classifier. For which error measure is it optimal?**
ŷ(x) = argmax_k P(Y = k | X = x) = argmax_k P(X = x | Y = k) · P(Y = k)

It is optimal for the 0-1 loss (misclassification error) — it minimizes the expected number of
misclassifications.

**Name at least 3 Bayesian type classifiers and the assumptions each of those make.**
1. Linear Discriminant Analysis (LDA): assumes each class's features follow a multivariate
   Gaussian distribution, and all classes share the same covariance matrix (leading to linear
   decision boundaries).
2. Quadratic Discriminant Analysis (QDA): also assumes Gaussian class-conditional
   distributions, but allows each class its own covariance matrix (leading to quadratic
   decision boundaries).
3. (Gaussian) Naive Bayes: assumes conditional independence of the features given the class
   (i.e. the joint class-conditional density factorizes as a product over individual features).

**How can we easily compute ROC curves for Bayesian type classifiers?**
Bayesian classifiers naturally output posterior class probabilities P(Y=1|X=x). To get a ROC
curve, vary the decision threshold applied to this posterior probability from 0 to 1, and at
each threshold compute the resulting True Positive Rate and False Positive Rate; plotting TPR
against FPR across all thresholds gives the ROC curve.

---

## 6. Feature Engineering

**Name the two most common methods to transform qualitative data into features.**
1. One-hot encoding (dummy variables) — creates a binary indicator column per category
2. Label/ordinal encoding — maps each category to an integer (appropriate when categories have
   a natural order)

**What is meant by "feature space lifting"?**
Explicitly transforming the original input features into a higher-dimensional feature space
(e.g. via polynomial terms, basis functions) so that relationships which are non-linear in the
original space become linear (or more easily separable) in the new, higher-dimensional space.

**What is an "interaction term"?**
A new feature constructed by combining (typically multiplying) two or more existing predictors,
used to capture the effect that the predictors have jointly on the target beyond what each
predictor contributes individually (a non-additive/combined effect).

**What is the (in terms of result) best feature selection method?**
Best subset selection — exhaustively evaluating every possible subset of features and picking
the one that performs best. It is optimal in terms of result quality but computationally
infeasible for even a moderate number of features (2^p subsets).

**What statistics can we use for feature selection? Name a typical statistical feature selection method.**
Statistics such as correlation coefficients, t-statistics/p-values, the chi-squared statistic,
ANOVA F-statistic, and mutual information. A typical method: univariate statistical filtering
(e.g. selecting the k features with the highest F-test score or lowest p-value, such as
scikit-learn's `SelectKBest`).

**Name and explain at least 2 feature selection methods that are based on classifiers/regressors.**
1. Recursive Feature Elimination (RFE): repeatedly fit the model, rank features by importance
   (e.g. coefficient magnitude), remove the least important feature(s), and repeat until the
   desired number of features remains.
2. Sequential Feature Selection (SFS, forward/backward stepwise selection): greedily add (forward)
   or remove (backward) one feature at a time, at each step choosing the feature that most
   improves (or least hurts) the model's cross-validated performance.
3. (Additional example: embedded methods like LASSO, where L1 regularization shrinks some
   coefficients exactly to zero, effectively performing feature selection during model fitting.)

---

## 7. Outliers, Cross Validation, Resampling

**What should we do when we detect an outlier in our data? Should we remove it?**
Don't remove it automatically. First investigate the cause: if it's a measurement/data-entry
error, correct or remove it; if it's a genuine, rare-but-valid observation, it may carry real
information and should typically be kept (or handled with robust methods) rather than deleted,
since removing genuine outliers can bias the model and hide real-world variability.

**What are the advantages and disadvantages of "leave-one-out cross-validation" (LOOCV)?**
- Advantages: very low bias (each training set uses n−1 of n points, almost the full dataset),
  and it's deterministic (no randomness in how folds are formed, unlike k-fold with random
  splits).
- Disadvantages: computationally expensive (requires n model fits), and the resulting error
  estimate can have high variance because the n training sets are nearly identical to each
  other (highly correlated), so the individual fold errors are highly correlated too.

**Describe k-fold cross-validation. Why do we use it? What are the dangers of it?**
The data is randomly split into k roughly equal-sized folds. For each of the k iterations, one
fold is held out as the test/validation set while the model is trained on the remaining k−1
folds; the held-out fold's performance is recorded. Results are averaged over all k iterations
to get the final performance estimate. It is used because it gives a good bias/variance
trade-off and is much cheaper than LOOCV while still using all data for both training and
testing. Dangers: choosing k too small increases bias (less training data per fold) and too
large increases variance and computation cost (approaching LOOCV); data leakage if
preprocessing (e.g. scaling, feature selection) is performed before splitting into folds rather
than within each fold; and misleading results if the data isn't properly shuffled/stratified
(e.g. ordered or imbalanced data).

**What is bootstrapping?**
A resampling technique where new samples ("bootstrap samples") of size n are drawn from the
original dataset of size n with replacement. Each bootstrap sample is used to fit a model or
compute a statistic, and repeating this many times allows estimation of the variability
(standard error, confidence intervals) of that statistic or of model performance. It is also
the basis of bagging, used e.g. in Random Forests.

---

## 8. Optimization

**Explain the iteration scheme of Gradient Descent in one or two sentences.**
Starting from an initial parameter value, gradient descent repeatedly updates the parameters by
taking a step in the direction opposite to the gradient of the objective function (the
direction of steepest decrease):
θ_(t+1) = θ_t − η · ∇L(θ_t), where η is the learning rate — this is repeated until the
parameters converge (change becomes negligible) or a stopping criterion is reached.

**What is a black-box optimization problem?**
An optimization problem in which the objective function has no known closed-form expression and
no accessible derivatives — it can only be evaluated ("queried") at specific input points
(often at high computational cost), so the optimizer must decide where to query based only on
past evaluation results.

**What is the best optimization method for a small number of possible parameter combinations and a cheap-to-compute objective function?**
Grid search (exhaustive search over all combinations).

**What are the advantages of grid search?**
It's simple to implement and understand, trivially parallelizable (each combination can be
evaluated independently), deterministic/reproducible, and guaranteed to find the best
combination within the specified grid.

**Explain Monte Carlo search and when we use it.**
Monte Carlo (random) search evaluates the objective function at randomly sampled points from
the parameter search space, rather than exhaustively evaluating a fixed grid. It's used when
the search space is too large or continuous for grid search to be practical, and empirically
often finds good solutions faster than grid search because it doesn't waste evaluations on
unimportant dimensions.

**Name at least 3 advanced optimization methods.**
Bayesian Optimization, Genetic Algorithms / Evolutionary Strategies, Simulated Annealing (also
acceptable: Hyperband/successive halving, Particle Swarm Optimization).

**When do we usually use Bayesian Optimization?**
When the objective function is expensive to evaluate (e.g. training a large model can take
hours), is treated as a black box (no gradient information), and we want to find a good
solution using as few evaluations as possible. Bayesian Optimization builds a probabilistic
surrogate model (typically a Gaussian Process) of the objective based on past evaluations, and
uses it to intelligently choose the next most promising point to evaluate.

---

## 9. Advanced Classifiers

**Explain the concept of the SVM classifier in a few sentences.**
The Support Vector Machine finds the hyperplane that separates the two classes while
maximizing the margin — the distance between the hyperplane and the closest data points from
each class (the "support vectors"). For data that isn't perfectly linearly separable, it allows
some points to violate the margin or even be misclassified, controlled by a penalty parameter
(C). For non-linear problems, the kernel trick is used to implicitly map data into a
higher-dimensional space where a separating hyperplane exists.

**What does the parameter C control in the SVM?**
C controls the trade-off between maximizing the margin and minimizing training misclassification
(margin violations) — it acts as a regularization parameter. A small C allows a wider margin
with more tolerance for misclassified/margin-violating points (stronger regularization, simpler
model); a large C penalizes margin violations heavily, resulting in a narrower margin that fits
the training data more closely (weaker regularization, higher risk of overfitting).

**What is the idea of the kernel trick and how does it differ from feature lifting?**
The kernel trick computes the inner product between two points as if they had been mapped into
a (possibly very high- or infinite-dimensional) feature space, without ever explicitly
constructing that mapping — via a kernel function K(x, x') = φ(x)·φ(x'). This differs from
explicit feature space lifting, which constructs the transformed features φ(x) directly; that
approach becomes computationally infeasible as the target dimensionality grows large, whereas
the kernel trick sidesteps this by only ever needing pairwise similarities, not the explicit
coordinates.

**What is tree pruning?**
The process of removing branches or nodes from a fully grown decision tree (post-pruning) to
reduce its complexity and prevent overfitting — typically done via cost-complexity pruning,
which balances tree size against training error using a complexity penalty parameter, chosen
e.g. via cross-validation.

**What is the main advantage of the basic decision tree classifier?**
High interpretability — it's a "white-box" model whose decision rules can be easily visualized
and explained. Additional advantages: handles both numerical and categorical data naturally,
requires no feature scaling/normalization, and can capture non-linear relationships and
interactions automatically.

**Name at least 4 parameters that influence random forest computation.**
1. Number of trees (n_estimators)
2. Number of features considered at each split (max_features, "mtry")
3. Maximum tree depth (max_depth)
4. Minimum number of samples required to split a node / to be at a leaf (min_samples_split /
   min_samples_leaf)
5. (Additional: bootstrap sample size, criterion used for splitting e.g. Gini vs. entropy)

---

*Tip for the presentation: for the "formal" questions (Bayes classifier, RSE/R², logistic
function), be ready to write the formula on a whiteboard from memory, not just recite it —
that's usually what's actually being tested.*
