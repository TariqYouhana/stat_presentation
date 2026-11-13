# Code Walkthrough: Probability Distributions in Machine Learning

This document provides a plain-English, line-by-line explanation of every cell in `notebook.ipynb` (and its counterpart script `demo.py`). It is specifically written for an undergraduate student who needs to explain and defend every line of code during an oral examination. The final section provides **10 likely examination questions** with model answers.

---

## 1. Cell-by-Cell Notebook Walkthrough

### Cell 1 (Markdown) & Cell 2 (Markdown): Project Header and Section 0 Overview
- **Purpose:** Introduces the project scope, audience, and learning objectives.
- **Key Concepts:** Distinguishes between theoretical distributions, statistical estimation via MLE, and machine learning classification via Naive Bayes.

### Cell 3 (Code): Section 0 Setup & Imports
- **What it does:** Imports the required libraries (`numpy`, `pandas`, `matplotlib.pyplot`, `seaborn`, `sklearn`, `scipy.stats`), sets plotting themes, creates output folders (`figures/` and `results/`), and initializes the random number generator: `rng = np.random.default_rng(42)`.
- **Why it matters:** Using `np.random.default_rng(42)` ensures that anyone running this code on any computer will produce the exact same numbers, tables, and figures.

### Cell 4 (Markdown): Section 1 Mathematical Derivation of Bernoulli MLE
- **What it does:** Derives the log-likelihood function $\ln L(p) = k \ln(p) + (n-k) \ln(1-p)$ and proves analytically that taking the first derivative and setting it to zero yields $\hat{p} = k/n$ (the sample mean).

### Cell 5 (Code): Section 1 Simulation & Table 1 Generation
- **What it does:**
  - Calls `rng.binomial(n=1, p=0.7, size=1000)` to simulate 1,000 coin-toss trials.
  - Counts successes using `k_successes = int(np.sum(bernoulli_trials))`.
  - Calculates the sample mean `np.mean(bernoulli_trials)` ($0.6960$) and sample variances using `ddof=0` (MLE divisor $n$, $0.2116$) and `ddof=1` (unbiased divisor $n-1$, $0.2118$).
  - Assembles and saves `results/table1_bernoulli_stats.csv`.

### Cell 6 (Markdown): "What this shows" - Table 1
- **What it does:** Explains the observed values: 696 successes, an MLE estimate of $0.6960$, and an absolute error of only $0.0040$ ($0.57\%$) compared to the true probability $0.7000$.

### Cell 7 (Code): Section 1 Log-Likelihood Function (Figure 1)
- **What it does:**
  - Creates an evaluation grid of 500 candidate $p$ values between $0.01$ and $0.99$ (`np.linspace(0.01, 0.99, 500)`).
  - Evaluates $\ln L(p)$ vectorized across the grid using NumPy.
  - Plots the curve, drawing a red vertical dashed line at $\hat{p}_{\text{MLE}} = 0.6960$ and a green dotted line at true $p = 0.7000$.
  - Saves the plot to `figures/fig1_bernoulli_log_likelihood.png`.

### Cell 8 (Markdown): "What this shows" - Figure 1
- **What it does:** Points out that the log-likelihood reaches its global peak at $\hat{p} = 0.6960$ with a value of $-614.22$. Explains that any candidate parameter far from $0.70$ suffers severe log-likelihood penalties.

### Cell 9 (Code): Observed vs Theoretical Bar Chart (Figure 2)
- **What it does:** Computes relative frequencies of 0s and 1s using `pd.Series(bernoulli_trials).value_counts(normalize=True)`. Draws side-by-side bars for observed ($0.3040$ vs $0.6960$) and theoretical ($0.3000$ vs $0.7000$) values. Saves to `figures/fig2_bernoulli_observed_vs_theory.png`.

### Cell 10 (Markdown): "What this shows" - Figure 2
- **What it does:** Explains that the observed frequencies converge within $0.0040$ of theory, confirming the Law of Large Numbers for discrete random variables.

### Cell 11 (Code): Bernoulli Parameter Effect (Figure 3)
- **What it does:** Loops over $p \in [0.2, 0.5, 0.8]$ and draws subplots showing the distribution of probability mass between failure (0) and success (1). Saves to `figures/fig3_bernoulli_parameter_effect.png`.

### Cell 12 (Markdown): "What this shows" - Figure 3
- **What it does:** Explains how varying $p$ linearly transfers probability mass, controlling the expectation $\mathbb{E}[X] = p$ and the distributional skewness.

### Cell 13 (Markdown): Section 2 Normal Distribution and MLE Theory
- **What it does:** Provides the Gaussian PDF formula and mathematically derives the MLE for the mean ($\hat{\mu} = \bar{x}$) and variance ($\hat{\sigma}^2 = \frac{1}{n} \sum (x_i - \bar{x})^2$).

### Cell 14 (Code): Section 2 Simulation & Table 2 Generation
- **What it does:**
  - Simulates 1,000 samples for Class A ($\mu=60, \sigma=8$) and Class B ($\mu=75, \sigma=8$) using `rng.normal()`.
  - Calculates sample means ($59.3769$ and $75.1931$), MLE variances with `ddof=0` ($65.8003$ and $65.1695$), and unbiased variances with `ddof=1` ($65.8662$ and $65.2347$).
  - Assembles and saves `results/table2_normal_mle_stats.csv`.

### Cell 15 (Markdown): "What this shows" - Table 2
- **What it does:** Notes that sample means and variances are very close to the true population parameters ($60.0$, $75.0$, and $64.0$). Explains that for $n=1,000$, the $n$ vs $n-1$ variance difference is only $0.066$ ($0.10\%$).

### Cell 16 (Code): Histograms with True Normal PDF Overlay (Figure 4)
- **What it does:**
  - Uses `sns.histplot(stat="density")` to display normalized sample histograms.
  - Overlays exact continuous density curves using `scipy.stats.norm.pdf()`.
  - Saves to `figures/fig4_normal_hist_pdf_overlay.png`.

### Cell 17 (Markdown): "What this shows" - Figure 4
- **What it does:** Explains how the discrete histograms follow the continuous bell curves, and notes that the overlap between $x = 60$ and $x = 75$ represents classification uncertainty.

### Cell 18 (Code): Normal PDF Parameter Effect (Figure 5)
- **What it does:** Plots Gaussian curves for narrow ($\sigma=4$), standard ($\sigma=8$), wide ($\sigma=12$), and shifted ($\mu=75$) configurations. Saves to `figures/fig5_normal_parameter_effect.png`.

### Cell 19 (Markdown): "What this shows" - Figure 5
- **What it does:** Explains that $\sigma$ governs peak height (density ranges from $0.033$ to $0.100$) and spread, while $\mu$ shifts the curve horizontally without altering its shape.

### Cell 20 (Code): Normal CDF Evaluation (Figure 6)
- **What it does:** Plots $F(x) = P(X \le x)$ for $\mathcal{N}(60, 8^2)$ using `scipy.stats.norm.cdf()`. Marks evaluation at $x = 68.0$ ($\mu + 1\sigma$), drawing dashed projection lines to axes. Saves to `figures/fig6_normal_cdf_reading.png`.

### Cell 21 (Markdown): "What this shows" - Figure 6
- **What it does:** Confirms that $P(X \le 68.0) = 0.8413$, directly validating the 68-95-99.7 empirical rule ($50\% + 34.13\% = 84.13\%$).

### Cell 22 (Code): Sample-Size Error Experiment (Table 3 & Figure 7)
- **What it does:**
  - Iterates over sample sizes $n \in [20, 100, 1000, 10000]$.
  - Runs 200 Monte Carlo trials per sample size, recording $|\bar{x} - 60.0|$.
  - Computes mean absolute errors ($1.4178$, $0.7154$, $0.1826$, $0.0602$).
  - Plots simulation vs theoretical expected error $\sqrt{2/\pi}\cdot\sigma/\sqrt{n}$ on a log-log scale. Saves to `figures/fig7_sample_size_error.png` and `results/table3_sample_size_errors.csv`.

### Cell 23 (Markdown): "What this shows" - Table 3 & Figure 7
- **What it does:** Highlights that the error decreases linearly with slope $-0.5$ on a log-log plot, proving that the sample mean converges to the true mean at the rate $1/\sqrt{n}$.

### Cell 24 (Markdown): Section 3 Gaussian Naive Bayes Theory
- **What it does:** Formulates Bayes' theorem for continuous features, explains how the conditional independence assumption allows the joint likelihood to become a product of 1D Gaussian densities, and shows how working in log space transforms products into sums.

### Cell 25 (Code): Section 3 Reference Implementation (scikit-learn)
- **What it does:**
  - Concatenates Class A and Class B data into a single feature matrix $X$ and label vector $y$.
  - Performs a stratified 70/30 train/test split (`train_test_split(..., test_size=0.3, random_state=42, stratify=y)`).
  - Fits `GaussianNB(var_smoothing=1e-9)` and predicts labels on the 600 test instances.
  - Prints accuracy ($85.00\%$), learned means (`theta_`), and learned variances (`var_`).

### Cell 26 (Markdown): "Backup: From-Scratch Implementation" Architecture
- **What it does:** Outlines the design of our NumPy implementation: estimating class priors, means, and variances; adding scikit-learn's smoothing $\epsilon$; calculating Gaussian log-densities; and using the log-sum-exp trick for softmax probabilities.

### Cell 27 (Code): From-Scratch Functions Implementation
- **What it does:**
  - `fit_gaussian_nb`: Computes priors, means, and variances using vector operations, adding $\epsilon = 10^{-9} \times \max(\text{var}(X))$.
  - `gaussian_log_pdf`: Implements $-0.5\ln(2\pi\sigma^2) - 0.5(x-\mu)^2/\sigma^2$.
  - `predict_gaussian_nb`: Computes $\ln P(c) + \sum \ln f(x_j \mid c)$ and applies log-sum-exp softmax to obtain calibrated probabilities.

### Cell 28 (Code): Section 3 Automated Validation Checks (Assertions)
- **What it does:** Compares from-scratch predictions and parameters against scikit-learn using `assert`:
  1. `assert np.array_equal(pred_scratch, pred_sklearn)` (0 mismatches)
  2. `assert np.allclose(proba_scratch, proba_sklearn, atol=1e-6)` (difference $= 5.55 \times 10^{-16}$)
  3. `assert np.allclose(means_2c, gnb_reference.theta_, atol=1e-6)` (difference $= 0.00$)
  4. `assert np.allclose(vars_2c, gnb_reference.var_, atol=1e-6)` (difference $= 0.00$)

### Cell 29 (Markdown): "What this shows" - Section 3 Assertions
- **What it does:** Confirms that all four assertions pass with machine-level precision, proving mathematical and numerical equivalence with scikit-learn.

### Cell 30 (Code): GaussianNB Confusion Matrix & Heatmap (Figure 8 & Table 4)
- **What it does:** Computes the test confusion matrix, saves `results/table4_gaussian_nb_2class_metrics.csv`, and renders `figures/fig8_gaussian_nb_confusion_matrix.png` with annotations.

### Cell 31 (Markdown): "What this shows" - Table 4 & Figure 8
- **What it does:** Explains that the model correctly classifies 510 out of 600 samples ($85.00\%$). Shows that misclassifications are evenly split (46 for Class A, 44 for Class B) because priors and variances are balanced.

### Cell 32 (Code): GaussianNB Decision Boundary Analysis (Figure 9)
- **What it does:**
  - Generates a dense grid of 2,000 $x$ values between 40 and 95.
  - Finds the exact threshold where posterior probability crosses $0.5$ ($x = 67.22$).
  - Computes the sample midpoint $(59.45 + 75.00)/2 = 67.23$ and population midpoint $(60.0 + 75.0)/2 = 67.50$.
  - Overlays all three thresholds on the fitted Gaussian bell curves. Saves to `figures/fig9_gaussian_nb_decision_boundary.png`.

### Cell 33 (Markdown): "What this shows" - Figure 9
- **What it does:** Explains why the empirical decision boundary ($67.22$) matches the sample midpoint ($67.23$) within $0.01$ units, verifying the theoretical theorem for equal-variance Gaussian classifiers.

### Cell 34 (Markdown): Section 4 Real-World Iris Benchmark Introduction
- **What it does:** Explains the Iris dataset (150 observations, 4 features, 3 flower species) and outlines the multiclass experimental setup.

### Cell 35 (Code): Iris Model Fitting and Assertions
- **What it does:**
  - Loads Iris and creates a 70/30 stratified train/test split.
  - Fits both `GaussianNB` and `fit_gaussian_nb`.
  - Runs all 4 assertions comparing predictions, probabilities, means, and variances. All pass with zero errors.
  - Prints accuracy ($91.11\%$, 41/45 correct).

### Cell 36 (Markdown): "What this shows" - Iris Assertions
- **What it does:** Verifies that the from-scratch implementation handles multiclass, multi-feature real-world data with the same machine precision as scikit-learn.

### Cell 37 (Code): Iris Learned Parameters (Table 5) & Confusion Matrix (Table 6)
- **What it does:** Extracts learned Gaussian parameters for all 12 combinations of species and features, saving `results/table5_iris_learned_parameters.csv`. Saves confusion matrix to `results/table6_iris_metrics.csv`.

### Cell 38 (Markdown): "What this shows" - Tables 5 & 6
- **What it does:** Details the $91.11\%$ accuracy, noting that *setosa* has $100\%$ accuracy (15/15) while *versicolor* and *virginica* have 4 mutual misclassifications.

### Cell 39 (Code): Iris Confusion Matrix Heatmap (Figure 10)
- **What it does:** Renders a 3x3 green confusion matrix heatmap with large font numbers. Saves to `figures/fig10_iris_confusion_matrix.png`.

### Cell 40 (Markdown): "What this shows" - Figure 10
- **What it does:** Explains that off-diagonal misclassifications are strictly confined to the versicolor/virginica sub-matrix.

### Cell 41 (Code): Iris Petal Length Distribution Plot (Figure 11)
- **What it does:** Uses `sns.kdeplot(fill=True)` to plot the continuous distribution of petal length for each species. Saves to `figures/fig11_iris_feature_distributions.png`.

### Cell 42 (Markdown): "What this shows" - Figure 11
- **What it does:** Points out that *setosa* is completely isolated around mean $1.49$ cm, while *versicolor* (mean $4.24$) and *virginica* (mean $5.63$) overlap between $4.5$ and $5.2$ cm, explaining why all errors occur between them.

### Cell 43 (Markdown): Section 5 Text Classification Theory
- **What it does:** Explains the difference between the Bernoulli model (binary presence/absence, document-level evidence) and Multinomial model (term frequency counts, document-length normalization), introducing Laplace smoothing ($\alpha = 1$).

### Cell 44 (Code): Text Vocabulary and True Parameters
- **What it does:** Defines the 8-word vocabulary (`['offer', 'free', 'money', 'click', 'meeting', 'project', 'report', 'schedule']`), specifies true Bernoulli probabilities, and specifies true Multinomial probabilities summing to $1.0$.

### Cell 45 (Code): Text Simulation (1,000 Documents)
- **What it does:**
  - Simulates 500 Ham and 500 Spam documents.
  - Generates binary indicators for Bernoulli via `rng.binomial()`.
  - Generates word count vectors (length 20 words) for Multinomial via `rng.multinomial()`.
  - Splits both into 70% train and 30% test sets using identical stratified index splits.

### Cell 46 (Code): Reference Text Fits and Accuracies
- **What it does:** Fits `BernoulliNB(alpha=1.0)` and `MultinomialNB(alpha=1.0)`. Exponentiates `feature_log_prob_` to recover probability parameters. Computes test accuracies ($97.67\%$ for Bernoulli, $100.00\%$ for Multinomial).

### Cell 47 (Markdown): "Backup: From-Scratch Text Classifiers" Theory
- **What it does:** Provides mathematical definitions of Laplace smoothing for both models and explains log-joint score evaluation.

### Cell 48 (Code): Text Classifiers from Scratch & Assertions
- **What it does:**
  - Implements `fit_multinomial_nb`, `predict_multinomial_nb`, `fit_bernoulli_nb`, and `predict_bernoulli_nb`.
  - Runs 6 automated assertions against scikit-learn. All pass with differences $< 10^{-14}$.

### Cell 49 (Markdown): "What this shows" - Section 5 Assertions
- **What it does:** Confirms that from-scratch text implementations match scikit-learn identically in predictions, probabilities, and learned smoothed parameters.

### Cell 50 (Code): Parameter Recovery Table (Table 7) & Bar Charts (Figures 12 & 13)
- **What it does:** Assembles Table 7 comparing true vs recovered parameters, saving to `results/table7_text_parameter_recovery.csv`. Plots side-by-side bar charts for Bernoulli (`figures/fig12_bernoulli_text_parameters.png`) and Multinomial (`figures/fig13_multinomial_text_parameters.png`).

### Cell 51 (Markdown): "What this shows" - Figures 12 & 13
- **What it does:** Details specific parameter recoveries (e.g., Ham *meeting* recovered at $0.2476$ vs true $0.2500$), explaining that Laplace smoothing causes a minor, harmless shrinkage toward uniform probabilities.

### Cell 52 (Code): Text Performance Table (Table 8) & Heatmaps (Figure 14)
- **What it does:** Saves performance metrics to `results/table8_text_nb_metrics.csv` and renders side-by-side confusion matrix heatmaps to `figures/fig14_text_nb_confusion_matrices.png`.

### Cell 53 (Markdown): "What this shows" - Table 8 & Figure 14
- **What it does:** Explains why Multinomial achieves $100\%$ accuracy (word counts compound log-likelihood evidence additively) while Bernoulli achieves $97.67\%$ (clipping counts to 1 loses frequency intensity).

### Cell 54 (Markdown): Zero-Frequency Problem Mathematical Background
- **What it does:** Shows why an unseen word causes $\ln(0) = -\infty$, leading to $0 \times (-\infty)$ or $(-\infty) - (-\infty)$ arithmetic which produces `NaN`.

### Cell 55 (Code): Zero-Frequency Demonstration on 20 Documents (Table 9)
- **What it does:**
  - Takes the first 20 documents of the training set.
  - Fits `BernoulliNB(alpha=0.0)` and `BernoulliNB(alpha=1.0)`.
  - Evaluates an incoming test email containing *free* (which had count 0 in Ham training).
  - Demonstrates that $\alpha=0$ produces `NaN` probabilities, whereas $\alpha=1$ yields $P(\text{Ham}) = 0.9990$.
  - Saves `results/table9_zero_frequency_experiment.csv`.

### Cell 56 (Markdown): "What this shows" - Table 9
- **What it does:** Summarizes the empirical breakdown of the unsmoothed model and demonstrates how Laplace smoothing rescues the classification without distorting predictions.

### Cell 57 (Markdown): Section 6 Presentation Summary & Core Takeaways
- **What it does:** Summarizes the entire presentation into 7 clear, defensible takeaways.

---

## 2. Ten Likely Examination Questions & Model Answers

### Question 1: Why do we work in log-space when implementing Naive Bayes instead of multiplying probabilities directly?
**Answer:** Multiplying many small probabilities (e.g., $10^{-3} \times 10^{-4} \times \dots$) causes floating-point **arithmetic underflow**, where the product rounds down to absolute zero in standard 64-bit IEEE floating-point arithmetic. Working in log space transforms products into sums:
$$\ln \left( P(c) \prod_{j=1}^d P(x_j \mid c) \right) = \ln P(c) + \sum_{j=1}^d \ln P(x_j \mid c)$$
Logarithms preserve the exact argmax ordering because $\ln(z)$ is a strictly monotonically increasing function.

---

### Question 2: What is the difference between the MLE variance ($1/n$) and the unbiased sample variance ($1/(n-1)$)? Which does scikit-learn use?
**Answer:** The Maximum Likelihood Estimator of Gaussian variance is $\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n} \sum (x_i - \bar{x})^2$. It is slightly **biased downwards** because deviations are measured from the sample mean $\bar{x}$ rather than the true population mean $\mu$, consuming one degree of freedom. The unbiased estimator divides by $n-1$. Scikit-learn's `GaussianNB` implements the pure MLE formulation with divisor $n$ (`ddof=0`). For large sample sizes (such as $n = 1,000$ in Section 2), the difference between the two is negligible ($65.8003$ vs $65.8662$).

---

### Question 3: Why does scikit-learn add a `var_smoothing` term to feature variances in GaussianNB?
**Answer:** If a continuous feature has zero or near-zero variance within a training class (e.g., all samples have identical values), the Gaussian denominator $\sqrt{2\pi\sigma^2}$ becomes zero, and division by variance causes division-by-zero crashes. To ensure numerical stability, scikit-learn computes $\epsilon = \text{var\_smoothing} \times \max_j(\text{variance}(X_j))$ and adds $\epsilon$ to every class feature variance. By default, $\text{var\_smoothing} = 10^{-9}$.

---

### Question 4: What is the "log-sum-exp trick" and why is it necessary for the softmax function?
**Answer:** The softmax formula converts unnormalized log scores $\mathbf{z}$ into posterior probabilities:
$$P(c) = \frac{\exp(z_c)}{\sum_k \exp(z_k)}$$
If $z_c$ is a large negative number (e.g., $-850$), computing $\exp(-850)$ underflows to $0.0$, producing $0 / 0 = \text{NaN}$. If $z_c$ is large and positive (e.g., $+850$), $\exp(850)$ overflows to $+\infty$. The log-sum-exp trick subtracts $M = \max_k(z_k)$ before exponentiating:
$$\frac{\exp(z_c - M)}{\sum_k \exp(z_k - M)}$$
Because the constant $e^{-M}$ cancels in the numerator and denominator, the probabilities remain mathematically identical while ensuring the largest exponent evaluated is $\exp(0) = 1$, completely eliminating underflow and overflow.

---

### Question 5: What is the "naive" assumption in Naive Bayes, and how does the model perform when the assumption is violated?
**Answer:** The naive assumption is that all features are **conditionally independent given the class label**:
$$P(\mathbf{x} \mid c) = \prod_{j=1}^d P(x_j \mid c)$$
In real-world data (such as Iris or text documents), features are often correlated (e.g., petal length and petal width are strongly correlated). Although this assumption is technically false, Naive Bayes often performs remarkably well in practice. This is because classification only requires picking the correct class with the largest posterior score ($\arg\max$); the probability values themselves may be overconfident, but the rank-ordering of classes frequently remains optimal.

---

### Question 6: For two Gaussian classes, under what mathematical conditions is the optimal decision boundary exactly at the midpoint between their means?
**Answer:** The log-posterior ratio between Class A and Class B is:
$$\ln \frac{P(A \mid x)}{P(B \mid x)} = \ln \frac{P(A)}{P(B)} + \frac{1}{2}\ln \frac{\sigma_B^2}{\sigma_A^2} - \frac{(x - \mu_A)^2}{2\sigma_A^2} + \frac{(x - \mu_B)^2}{2\sigma_B^2}$$
When the **priors are equal** ($P(A) = P(B)$) and the **variances are equal** ($\sigma_A^2 = \sigma_B^2 = \sigma^2$), the first two terms vanish and the quadratic terms simplify to:
$$-\frac{1}{2\sigma^2} \left[ (x - \mu_A)^2 - (x - \mu_B)^2 \right] = \frac{\mu_A - \mu_B}{\sigma^2} \left( x - \frac{\mu_A + \mu_B}{2} \right)$$
Setting the ratio to $0$ (which corresponds to equal posterior probabilities $P(A \mid x) = P(B \mid x) = 0.5$) yields $x = \frac{\mu_A + \mu_B}{2}$. Thus, the optimal boundary is strictly the midpoint between the two class means.

---

### Question 7: How do Bernoulli Naive Bayes and Multinomial Naive Bayes differ in how they represent text data?
**Answer:** 
- **Bernoulli Naive Bayes:** Represents documents as binary vectors $x_j \in \{0, 1\}$ (word present or absent). Its likelihood accounts for both word presence ($p_{cj}$) and word absence ($1 - p_{cj}$).
- **Multinomial Naive Bayes:** Represents documents as integer word-count vectors $x_j \in \{0, 1, 2, \dots\}$. Its likelihood models the frequency of word occurrences given a document length. It does not penalize for words that are absent, but strongly rewards repetitive occurrences of distinctive keywords.

---

### Question 8: What is the "zero-frequency problem" in Naive Bayes, and how does Laplace smoothing resolve it?
**Answer:** If word $w_j$ never appeared in training documents of class $c$, the empirical Maximum Likelihood estimate is $\hat{p}_{cj} = 0$. When a test document containing $w_j$ arrives, the model computes $\ln(0) = -\infty$, driving the joint class likelihood to zero ($-\infty$ in log-space) and completely vetoing the class regardless of how many other strong keywords are present. **Laplace smoothing** adds a pseudo-count of $\alpha = 1$ to the numerator and adjusts the denominator:
- Multinomial: $\hat{p}_{cj} = \frac{\text{count}_{cj} + 1}{\sum_k \text{count}_{ck} + V}$
- Bernoulli: $\hat{p}_{cj} = \frac{\text{doc\_count}_{cj} + 1}{N_c + 2}$
This guarantees that all probabilities are strictly positive ($\hat{p}_{cj} > 0$), preventing $-\infty$ log-likelihoods and ensuring smooth, robust predictions.

---

### Question 9: Why did Multinomial Naive Bayes achieve 100% accuracy while Bernoulli Naive Bayes achieved 97.67% on our simulated dataset?
**Answer:** In our simulation, documents contained 20 words. Spam emails frequently contained multiple occurrences of words like *offer* and *free*, while Ham emails contained multiple occurrences of *meeting* and *schedule*. Multinomial Naive Bayes accumulates log-likelihood additively with each occurrence: $3 \times \ln(0.25)$ provides three times the discriminative evidence of a single occurrence. Bernoulli Naive Bayes truncates any positive count to $1$, throwing away frequency intensity. This lost evidence caused Bernoulli NB to misclassify 7 borderline documents.

---

### Question 10: In the sample-size experiment (Section 2), why does the log-log plot of error vs. sample size have a slope of -0.5?
**Answer:** The Central Limit Theorem establishes that the sample mean $\bar{X}_n$ of $n$ independent random variables with variance $\sigma^2$ has standard error:
$$\text{SE}(\bar{X}_n) = \frac{\sigma}{\sqrt{n}} = \sigma \cdot n^{-0.5}$$
Taking the natural logarithm of both sides:
$$\ln(\text{Error}) \approx \ln(\text{constant}) - 0.5 \ln(n)$$
When plotted on log-log axes ($\ln(\text{Error})$ versus $\ln(n)$), the relationship is a straight line with a constant theoretical slope of $-0.5$. This confirms that reducing estimation error by a factor of 10 requires a 100-fold ($10^2$) increase in sample size.
