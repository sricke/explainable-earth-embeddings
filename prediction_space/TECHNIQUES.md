## Techniques for Post-Hoc Explainability in Prediction Space
 
### 1 Linear Probing (Baseline)
 
**Question:** *How much task-relevant information is linearly accessible in each embedding?*
 
| Technique | Regression | Classification | Details |
|-----------|------------|----------------|---------|
| **Ridge Regression** | ✓ | — | L2-regularized linear model; α selected via cross-validation |
| **Logistic Regression (L2)** | — | ✓ | Multinomial logistic; regularization strength via CV |
| **k-NN (k=1, k=3)** | ✓ | ✓ | Non-parametric baseline; tests metric-space quality |
 
Linear probes are the standard "how much is in there" test. Ridge/logistic regression reveals linearly decodable information; k-NN reveals whether the embedding's metric geometry preserves task-relevant distances (which is important because AEF uses a von Mises–Fisher bottleneck that constrains points to a hypersphere).
 
### 2 Feature Selection & Sparsity
 
**Question:** *Which embedding dimensions carry the most task-relevant signal? Is the information concentrated or distributed?*
 
| Technique | Type | Details |
|-----------|------|---------|
| **LASSO (L1 regression / L1 logistic)** | Filter → Embedded | Drives coefficients to zero; produces a sparse selection of informative dimensions |
| **Elastic Net** | Embedded | L1 + L2 hybrid; handles correlated dimensions better than pure LASSO |
| **Stability Selection** | Meta-wrapper | Runs subsampled LASSO many times; reports selection probability per dimension. Produces robust feature rankings even when dimensions are correlated |
| **Mutual Information** | Filter | MI(dimension, target); non-parametric, catches nonlinear univariate associations |
| **mRMR (min-Redundancy Max-Relevance)** | Filter | Selects dimensions that are individually informative and mutually non-redundant |
| **Progressive Ablation** | Perturbation | Following Benavides-Martinez: rank dimensions by importance, then remove top-k and retrain. Measures how concentrated the signal is |
 
**Key output:** Per-task "importance profiles" for each embedding — heatmaps of (dimension × task) showing which dimensions are specialists vs. generalists. This extends Benavides-Martinez's analysis to GeoCLIP and SatCLIP.