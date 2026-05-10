# Extended Analysis: Between-Peaks Cross-Period Comparison

This folder contains the cross-period extension of the main thesis analysis,
implemented in response to supervisor feedback (April 27, 2026) on whether
the Delta-only findings generalize across pandemic phases.

## Periods analyzed

| Period | Window | N tweets |
|---|---|---|
| Pre-Delta inter-peak | April 1 – June 30, 2021 | 78,410 |
| Delta peak | July 1 – September 30, 2021 | 72,847 |
| Post-Delta inter-peak | October 1 – November 30, 2021 | 30,796 |
| **Total** | | **182,053** |

## Pipeline

The same multi-method pipeline as the main Delta-only analysis was applied
across the three windows:

1. Data filtering and period definition
2. Period-specific descriptive statistics
3. Sentiment analysis (VADER + Twitter-RoBERTa)
4. Topic modeling (BERTopic)
5. Negative Binomial GLM regression per period
6. Cross-period Wald tests
7. Master visualization

## Methods note

The main thesis used Zero-Inflated Negative Binomial (ZINB) regression. In the
extended analysis, ZINB failed to converge for the Pre-Delta and Delta windows
due to Hessian inversion issues. For cross-window comparability, all three
windows were re-fit with Negative Binomial GLM. NB-GLM results are reported
as the primary cross-period comparison. See thesis Section 3.5.1 and Appendix D
for details.

## Key findings

1. The negative dialogic-engagement effect generalizes across all three
   windows (β = −0.55, −0.24, −0.14).
2. The sentiment-engagement coefficient varies systematically with crisis
   intensity (β = 0.63 at Delta peak, β = 1.11 in Post-Delta inter-peak).
3. The media-richness coefficient reverses sign between Delta peak (β = +1.26)
   and Post-Delta inter-peak (β = −0.37), suggesting phase-dependent dynamics.

## Files

- `R&D_extended_analysis.ipynb` — full Colab notebook with all 7 steps
- `results/` — CSV outputs from each step (sentiment, topics, regressions, Wald tests)
- `figures/` — key visualizations (sentiment comparison, topic evolution, regression coefficients, master figure)

## Reproducibility

The notebook expects the Kaggle dataset `gpreda/all-covid19-vaccines-tweets`
to be available. Run cells sequentially. Intermediate checkpoints are saved
to Google Drive (paths in the notebook); raw checkpoints are not committed
to git due to size.
