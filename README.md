# Loan Approval EDA & Statistical Analysis

## Key Finding

**Credit history is the dominant factor in loan approval** — statistically confirmed (χ² = 176.1, p < 0.0001) and confirmed again by a joint logistic regression with real p-values, where it carries an odds ratio of **39.5x** versus every other feature. Applicants with a positive credit history are approved 79.1% of the time vs. 7.9% without one.

An earlier pass of this project reported this odds ratio as ≈3.4x — that number came from fitting the model with `Credit_History` standardized (`StandardScaler`) alongside continuous features, which re-expresses its coefficient in "per 1 standard deviation" units instead of "0 vs 1." For a binary variable with SD ≈ 0.35, those are very different sizes — the fix (leaving binary/dummy columns unscaled) didn't change the conclusion, it corrected the number by almost 10x.

Just as important: **raw income and loan amount show no statistically significant relationship with approval**, either alone (all p > 0.4) or jointly in the model (all p > 0.17). A repeated cross-validated baseline model beats the 69.1% majority-class baseline modestly — accuracy 72.6% (95% CI [71.5%, 73.7%]), ROC-AUC 0.714 (95% CI [0.699, 0.730]) — reported with a real confidence interval rather than the better-looking single-split number (79.7% accuracy, 0.864 AUC), which was an optimistic draw.

---

## Problem Statement

Financial institutions must evaluate loan applications to minimize default risk while avoiding overly strict criteria that cost approved business. This project uses hypothesis testing — not just visualization — to identify which applicant factors are actually associated with approval, and which only *look* associated on a chart.

---

## Dataset Overview

* **Source:** Kaggle (Loan Prediction Dataset)
* **Records:** 614 | **Features:** 13
* **Target:** `Loan_Status` (Y/N) — **68.7% approved / 31.3% rejected.** This imbalance matters: any classifier built on this data needs to beat a 68.7% "approve everyone" baseline to demonstrate real value.

---

## Data Cleaning & Preprocessing

Two separate cleaning passes exist in this project, deliberately:

* **EDA-level cleaning** (mode for categorical, median for numeric) uses statistics from the full 614-row dataset — standard for descriptive EDA, since there's no train/test split at that stage.
* **Modeling-level cleaning** fits imputation values (mode/median) on the training set only, *after* splitting — the earlier version of this project computed imputation on the full dataset before splitting, which leaks test-set information into training rows. Fixed by reordering the pipeline: split first, then impute.

Also: `Dependents` `"3+"` → `3`; `Loan_ID` dropped (identifier, non-predictive); `Total_Income` and `Loan_to_Income` engineered post-imputation, separately for train and test.

Before imputing `Credit_History` (8.1% missing), checked whether missingness itself correlates with approval — it doesn't behave like an automatic-reject group (74.0% vs. 68.3% approval), so mode imputation is a safe simplification, not a blind guess.

---

## Method

* **Categorical variables vs. `Loan_Status`:** chi-square test of independence
* **Numeric variables vs. `Loan_Status`:** point-biserial correlation
* **Outliers:** quantified via IQR rule, not just eyeballed on a histogram
* **Multiple comparisons:** 13 hypothesis tests were run in total — corrected via Bonferroni (α = 0.05/13 ≈ 0.0038) rather than reading each p-value against the uncorrected 0.05 threshold

---

## Statistical Findings

| Variable | Test | Statistic | p-value | Sig. (p<0.05) | Sig. (Bonferroni) |
|---|---|---|---|---|---|
| Credit_History | χ² | 176.11 | <0.0001 | ✅ | ✅ |
| Property_Area | χ² | 12.30 | 0.0021 | ✅ | ✅ |
| Married | χ² | 4.73 | 0.0296 | ✅ | ❌ |
| Loan_to_Income | point-biserial r | -0.086 | 0.0328 | ✅ | ❌ |
| Education | χ² | 4.09 | 0.0431 | ✅ | ❌ |
| CoapplicantIncome | point-biserial r | -0.059 | 0.1429 | ❌ | ❌ |
| Dependents | χ² | 3.15 | 0.3689 | ❌ | ❌ |
| LoanAmount | point-biserial r | -0.033 | 0.4113 | ❌ | ❌ |
| Total_Income | point-biserial r | -0.031 | 0.4392 | ❌ | ❌ |
| Loan_Amount_Term | point-biserial r | — | 0.5771 | ❌ | ❌ |
| Gender | χ² | 0.11 | 0.7391 | ❌ | ❌ |
| ApplicantIncome | point-biserial r | -0.005 | 0.9073 | ❌ | ❌ |
| Self_Employed | χ² | 0.00 | 1.0000 | ❌ | ❌ |

**Only `Credit_History` and `Property_Area` survive Bonferroni correction.** With 13 tests run at uncorrected α=0.05, roughly 0.65 false positives are expected from chance alone — `Married` and `Education` sit exactly where that predicts. They're not described as "borderline significant, secondary drivers" here (which implies a real but weak effect) — the honest framing is that this dataset cannot distinguish them from noise, and they'd need re-testing on a larger or independent sample to say more.

**Approval rate by Credit_History:** 79.1% (has history) vs. 7.9% (no history)
**Approval rate by Property_Area:** Semiurban 76.8% > Urban 65.8% > Rural 61.5%

### Correction from an earlier pass
An earlier version claimed "applicants with no dependents have higher approval rates." Under a chi-square test this doesn't hold (p = 0.37) — approval rates across 0/1/2/3 dependents (68.6% / 64.7% / 75.3% / 64.7%) show no clean pattern. Dropped rather than kept.

### Outliers (IQR method)
* `ApplicantIncome`: 50 outliers (8.1%), max 81,000 vs. an IQR upper bound of ~10,171
* `LoanAmount`: 41 outliers (6.7%), max 700 vs. an IQR upper bound of ~262

---

## Baseline Model — Validating the EDA Jointly

A logistic regression checks whether `Credit_History` still dominates once every variable is controlled for simultaneously, and whether the model's own significance tests agree with the corrected EDA findings.

**Pipeline (leakage-free):** categorical labels encoded before splitting (deterministic, not a statistical fit); train/test split; imputation (mode/median) fit on training data only and applied to both sets; feature engineering and log-transforms applied post-imputation to each set independently; `StandardScaler` fit on training data, applied **only to continuous columns** — `Credit_History`, `Gender`, `Married`, `Education`, `Self_Employed`, `Dependents`, and the `Property_Area` dummies stay on their natural 0/1 scale so coefficients are directly interpretable.

**Statistical inference:** fit with `statsmodels.Logit` (not just sklearn) to get real standard errors and p-values per coefficient — necessary to say the model "confirms" anything, rather than just "points the same direction."

### Coefficients (statsmodels, real p-values)

| Feature | Odds Ratio | p-value | Significant? |
|---|---|---|---|
| **Credit_History** | **39.51** | <0.0001 | ✅ |
| Property_Area_Semiurban | 2.23 | 0.0062 | ✅ |
| Married | 1.55 | 0.124 | ❌ |
| Education | 1.43 | 0.202 | ❌ |
| CoapplicantIncome | 1.28 | 0.171 | ❌ |
| Gender | 0.79 | 0.468 | ❌ |
| Self_Employed | 0.82 | 0.570 | ❌ |
| LoanAmount | 0.83 | 0.379 | ❌ |
| ApplicantIncome | 1.18 | 0.471 | ❌ |
| Property_Area_Urban | 1.18 | 0.565 | ❌ |
| Dependents | 1.12 | 0.381 | ❌ |
| Loan_to_Income | 1.04 | 0.824 | ❌ |
| Loan_Amount_Term | 1.02 | 0.881 | ❌ |

**Only `Credit_History` and `Property_Area` are significant in the joint model — the exact same two variables that survive Bonferroni correction in the univariate EDA.** That agreement across two independent methods (corrected hypothesis tests, and a joint model with its own significance tests) is real evidence, not a coincidence of framing. `Married` and `Education` are non-significant here too, consistent with them failing correction above.

### Model Performance (repeated cross-validation)

| Metric | Majority Baseline | Single 80/20 Split | Repeated CV (5-fold × 10) — report this |
|---|---|---|---|
| Accuracy | 69.1% | 79.7% | **72.6%, 95% CI [71.5%, 73.7%]** |
| ROC-AUC | — | 0.864 | **0.714, 95% CI [0.699, 0.730]** |

The single split is optimistic — repeated CV (50 estimates instead of 5) gives an actual confidence interval instead of one lucky or unlucky draw, and it's substantially tighter than a single 5-fold run would produce.

Test-set classification report (single split, for reference):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Rejected | 0.68 | 0.66 | 0.67 |
| Approved | 0.85 | 0.86 | 0.85 |

Recall on `Rejected` matters most for a lender — missing a bad loan is the expensive error — and it's the weaker of the two classes here, worth flagging rather than burying under overall accuracy.

---

## Scope of the Income Finding

Income doesn't predict approval *in this dataset* — that's a defensible, tested claim. It is not, on its own, evidence about how real underwriting should be designed: this is a 614-row Kaggle convenience sample with no documented sampling process (unknown institution, market, year, or what stage of underwriting these applications reflect). Whether "credit history matters more than income" generalizes beyond this specific sample is a hypothesis worth testing against real, provenance-known data — not a policy recommendation this project can support.

---

## Business Recommendations

*(Scoped to this dataset, not stated as generalizable policy — see above.)*

* Within this dataset, `Credit_History` is the only large, robust signal — everything else is either non-significant or fails correction for multiple testing
* `Property_Area` (Semiurban) is the only other variable that holds up under both corrected testing and joint modeling — worth investigating *why* before treating it as causal
* Income thresholds carry no independent signal here once credit history is known — but this is a within-dataset finding, not a transferable underwriting rule
* Before deploying any model on this data: address the 69/31 class imbalance and the confirmed income/loan-amount outliers

---

## Limitations

* n=614 is a small, well-worn benchmark dataset with unknown sampling provenance — findings here demonstrate methodology, not production-ready policy
* `Credit_History` missingness (n=50) was checked but the sample is too small to draw a strong conclusion from
* `Married` and `Education` are flagged as failing multiple-comparisons correction, not dropped entirely — a larger or independent sample could still find a real (if smaller) effect
* This baseline model is linear (logistic regression) — it won't capture interaction effects a tree-based model might find

---

## Future Scope

* Compare against tree-based models (Random Forest, Gradient Boosting) to check for interaction effects a linear model misses
* Hyperparameter tuning (regularization strength, class weighting) via grid search
* Threshold tuning — the default 0.5 cutoff isn't necessarily optimal given the asymmetric cost of a missed bad loan
* SHAP/permutation importance if a non-linear model is adopted
* Deploy behind a simple decision-support interface for manual underwriting review

---

## Tools & Technologies
Python · Pandas · NumPy · Matplotlib · Seaborn · SciPy (`chi2_contingency`, `pointbiserialr`) · Statsmodels (`Logit`) · Scikit-learn (`LogisticRegression`, `RepeatedStratifiedKFold`)

---

## Project Structure
```bash
loan-risk-eda-analysis/
│
├── data/
│   └── train.csv
├── notebooks/
│   └── loan_eda.ipynb
├── images/
│   └── (visualizations)
└── README.md
```

---

## Author
**Mohammad Adil**
Data Analyst | Risk & Business Analytics
