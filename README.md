# Probability Distributions in Machine Learning
## University Data Science Presentation & Code Demonstration

This repository contains a self-contained, heavily commented demonstration for a university Data Science presentation on **"Probability Distributions in Machine Learning"**, focusing on Gaussian, Bernoulli, and Multinomial Naive Bayes, likelihood functions, and Maximum Likelihood Estimation (MLE).

---

## Environment & Package Versions

Tested and verified under the following software stack:
- **Python:** `3.14.7`
- **NumPy:** `2.5.4`
- **pandas:** `3.1.0`
- **matplotlib:** `3.11.2`
- **seaborn:** `0.13.2`
- **scikit-learn:** `1.9.1`
- **SciPy:** `1.18.1`
- **nbformat:** `5.11.1`
- **nbclient:** `0.11.0`

---

## Deliverables & Directory Structure

```
stat_presentation/
├── notebook.ipynb                  # Main Jupyter Notebook (runs top to bottom)
├── demo.py                         # Complete standalone Python demonstration script
├── README.md                       # Execution guide and environment specifications
├── code_walkthrough.md             # Cell-by-cell plain-English walkthrough + 10 oral exam Q&As
├── results_summary.md              # Empirical analysis, table/figure catalog, and check logs
├── figures/                        # High-resolution (dpi=200) figures
│   ├── fig1_bernoulli_log_likelihood.png
│   ├── fig2_bernoulli_observed_vs_theory.png
│   ├── fig3_bernoulli_parameter_effect.png
│   ├── fig4_normal_hist_pdf_overlay.png
│   ├── fig5_normal_parameter_effect.png
│   ├── fig6_normal_cdf_reading.png
│   ├── fig7_sample_size_error.png
│   ├── fig8_gaussian_nb_confusion_matrix.png
│   ├── fig9_gaussian_nb_decision_boundary.png
│   ├── fig10_iris_confusion_matrix.png
│   ├── fig11_iris_feature_distributions.png
│   ├── fig12_bernoulli_text_parameters.png
│   ├── fig13_multinomial_text_parameters.png
│   └── fig14_text_nb_confusion_matrices.png
└── results/                        # Exported CSV tables of every metric and experiment
    ├── table1_bernoulli_stats.csv
    ├── table2_normal_mle_stats.csv
    ├── table3_sample_size_errors.csv
    ├── table4_gaussian_nb_2class_metrics.csv
    ├── table5_iris_learned_parameters.csv
    ├── table6_iris_metrics.csv
    ├── table7_text_parameter_recovery.csv
    ├── table8_text_nb_metrics.csv
    └── table9_zero_frequency_experiment.csv
```

---

## How to Run

### 1. Running the Standalone Python Script
Execute the complete demonstration script from the terminal:
```bash
python demo.py
```
This runs all simulations, executes all validation assertions, saves all 14 figures into `figures/`, exports all 9 CSV tables into `results/`, and prints descriptive "What this shows" summaries with actual empirical numbers directly to standard output.

### 2. Running the Jupyter Notebook
Open the notebook in Jupyter Lab, Jupyter Notebook, or VS Code:
```bash
jupyter lab notebook.ipynb
# or
jupyter notebook notebook.ipynb
```
Select **Kernel -> Restart & Run All** to execute all cells top to bottom on a clean kernel. Every cell executes with zero errors.

To run headless from the command line:
```bash
jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb
```

---

## Notebook Structure & Highlights

1. **Section 0. Setup:** Imports, fixed random seed (`np.random.default_rng(42)`), and clean visual styling.
2. **Section 1. Bernoulli and MLE:**
   - 1,000 trials simulated with true $p = 0.7$.
   - Sample mean ($\hat{p} = 0.6960$) vs theory ($0.7000$).
   - Log-likelihood curve plotted over candidate $p \in [0.01, 0.99]$, peaking at $-614.22$.
   - Observed vs theoretical frequencies and parameter effects ($p \in \{0.2, 0.5, 0.8\}$).
3. **Section 2. Normal Distribution and MLE:**
   - Class A ~ $\mathcal{N}(60, 8^2)$ and Class B ~ $\mathcal{N}(75, 8^2)$ with $n = 1,000$ each.
   - Histograms on density scale with overlaid continuous true Gaussian curves.
   - Variance divisor comparison: MLE ($n$) vs unbiased ($n-1$).
   - Parameter sensitivity curves ($\mu, \sigma$) and CDF evaluation ($P(X \le 68) = 0.8413$).
   - Monte Carlo sample size experiment ($n \in \{20, 100, 1000, 10000\}$, 200 repeats each) validating the $1/\sqrt{n}$ convergence rate.
4. **Section 3. Gaussian Naive Bayes (Reference vs From-Scratch):**
   - Reference classifier: `sklearn.naive_bayes.GaussianNB(var_smoothing=1e-9)` ($85.00\%$ accuracy).
   - Backup from-scratch NumPy implementation matching scikit-learn's variance smoothing and using log-sum-exp softmax.
   - **Automatic Assertions:** Verified 100% identical test labels, probabilities matching within $5.55 \times 10^{-16}$, and parameters matching within $0.00$.
   - Decision boundary analysis: empirical threshold $67.22$ matches sample midpoint $67.23$ and theoretical midpoint $67.50$.
5. **Section 4. Real-World Benchmark: Iris with GaussianNB:**
   - Stratified 70/30 split, $91.11\%$ test accuracy ($41/45$ correct).
   - Setosa isolated with $100\%$ precision/recall.
   - KDE distribution plots of petal length explaining the 4 versicolor/virginica overlap errors.
   - All 4 automated assertions pass against scikit-learn.
6. **Section 5. Text Classification (Bernoulli vs Multinomial NB):**
   - 8-word vocabulary, 1,000 documents (500 Ham, 500 Spam).
   - Parameter recovery bar charts comparing ground truth against Laplace estimates ($\alpha = 1$).
   - Classification accuracy: Multinomial ($100.00\%$) vs Bernoulli ($97.67\%$).
   - From-scratch implementations of both models verified with assertions matching scikit-learn.
   - **Zero-frequency demonstration:** On a tiny 20-doc training set, unsmoothed models ($\alpha = 0$) crash with `NaN` due to $\ln(0) = -\infty$, whereas Laplace smoothing ($\alpha = 1$) enables correct classification with $99.90\%$ confidence.

---

## Educational Documentation

- **`code_walkthrough.md`:** Comprehensive, plain-English explanation of every cell in the notebook, written for undergraduate students explaining their code in an exam, along with **10 typical examination questions and model answers**.
- **`results_summary.md`:** Detailed report cataloging every figure and CSV table, a summary of key findings with actual numbers, a Theory vs Simulation verification matrix, and a checks log.
