# SHGR: A Generalized Maximal Correlation Coefficient

This repository provides the code associated with the paper  
**“SHGR: A Generalized Maximal Correlation Coefficient”**: https://neurips.cc/virtual/2025/loc/san-diego/poster/117128

---

**Abstract:**
*Traditional correlation measures, such as Pearson’s and Spearman’s coefficients, are limited in their ability to capture complex relationships, particularly nonlinear and multivariate dependencies. The Hirschfeld–Gebelein–Rényi (HGR) maximal correlation offers a powerful alternative by measuring the highest Pearson correlation achievable through nonlinear transformations of two random variables. However, estimating the HGR coefficient remains challenging due to the complexity of optimizing arbitrary nonlinear functions. We introduce a new coefficient, satisfying Rényi's axioms, based on the extension of HGR with Spearman's rank correlation: the Spearman HGR (\texttt{SHGR}). We propose a neural network-based estimator tailored to estimate (i) the bivariate correlation matrix, (ii) the multivariate correlations between a set of variables and another one, and (iii) the full correlation between two sets of variables. This estimate effectively detects nonlinear dependencies and demonstrates robustness to noise, outliers, and spurious correlations (\textit{hallucinations}). Additionally, it achieves competitive computational efficiency through designed neural architectures. Comprehensive numerical experiments and feature selection tasks confirm that \texttt{SHGR} outperforms existing state-of-the-art methods.*


---

The project is organized as follows:
- src/: Core Python functions and the \texttt{SHGR} estimator implementation.
- SHGR_Illustration.ipynb: A quick tutorial on how to use the \texttt{SHGR} 
- SHGR_Paper.ipynb: A notebook containing the exact experiments and figures presented in the paper.

---

**Reference**  
If you wish to cite this work, please refer to the corresponding paper or this repo.
