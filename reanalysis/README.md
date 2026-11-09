# Corrected re-analysis (v2)

See the top-level [README](../README.md) for run instructions and [CORRECTIONS.md](../CORRECTIONS.md) for the changes from v1.

| Output | Content |
|---|---|
| `out/T1_descriptives.csv` | Descriptives by window, all tweets and excluding automated accounts (Table 1) |
| `out/M0`–`M3` | Delta models: naive, + account controls, − automated accounts, corrected (Table 2) |
| `out/R1`–`R3` | Robustness: accounts with ≤50 tweets, one tweet per account, zero-inflated NB (Table 3) |
| `out/P_pre`, `P_delta`, `P_post` | Corrected model for each window (Table 4) |
| `out/P_pooled_interactions.csv` | Pooled model with feature × window interactions (tests of change, Table 4) |
| `out/paper_stats.json` | Automation counts, validation, sentiment classes and other descriptive figures cited in the text |
| `figs/` | Figures 1–3 |
