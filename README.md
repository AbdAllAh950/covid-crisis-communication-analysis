# Automated Accounts, Account Reach, and Message Features in COVID-19 Vaccine Tweets

Code and outputs for:

> Essa, A. S. A., & Kovalchuk, S. V. (2026). *Who Speaks Matters More Than How: Automated Accounts, Account Reach, and Message Features in COVID-19 Vaccine Tweets Around the Delta Wave* (Version 2) [Preprint]. SocArXiv.

**Version 2 corrects version 1 (29 October 2026).** See [CORRECTIONS.md](CORRECTIONS.md) for what changed and why. The original notebooks are kept for transparency, but their results are superseded.

## Summary

We analyse 180,278 deduplicated English-language COVID-19 vaccine tweets from the public Kaggle corpus by G. Preda. They cover three windows: before the Delta wave (Apr–Jun 2021), during it (Jul–Sep 2021) and after it (Oct–Nov 2021). Engagement is likes plus retweets. The models are negative binomial (NB2) with standard errors clustered by account.

- **Automated vaccine-slot accounts produced 34.0% of Delta-window tweets** and almost no engagement.
- A naive model of all tweets suggests a link or attached media multiplies engagement **4.71×**. After removing automated accounts and controlling for account reach, the estimate is **1.40× [1.19, 1.64]**.
- Positive sentiment: **1.48×**. Questions: **0.85×**. Hashtags: **0.91× per tag**. @Mentions: **no association**.
- **Followers are the strongest predictor**: about +36% engagement per doubling.
- Across windows, the link and sentiment associations are stable. The dialogic cues shift. There is no reversal of the link effect after the peak.

## Repository structure

```
reanalysis/            corrected analysis (v2) — use this
  scripts/             prep.py, models.py, robust.py, periods.py, figures.py, stats_for_paper.py
  out/                 all model outputs (CSV) and descriptive statistics (JSON)
  figs/                manuscript figures
paper/                 manuscript PDF (v2); older drafts are superseded
CORRECTIONS.md         what changed between v1 and v2
R&D26.ipynb            original Delta-window notebook (superseded; kept for transparency)
extended_analysis/     original cross-period notebook (superseded; kept for transparency)
```

## Reproduce

1. Download `vaccination_all_tweets.csv` from https://www.kaggle.com/datasets/gpreda/all-covid19-vaccines-tweets (CC0).
2. Install dependencies with `pip install pandas numpy statsmodels vaderSentiment matplotlib`.
3. Run:

```bash
cd reanalysis
mkdir -p out figs
python3 scripts/prep.py vaccination_all_tweets.csv out/features.pkl
python3 scripts/models.py out/features.pkl out            # Delta models M0–M3
python3 scripts/robust.py out/features.pkl out scripts    # robustness R1–R3
python3 scripts/periods.py out/features.pkl out scripts   # per-window + pooled interaction models
python3 scripts/stats_for_paper.py out/features.pkl scripts out/paper_stats.json
python3 scripts/figures.py out/features.pkl scripts figs out
```

`scripts/repro.py` checks that the naive specification reproduces the original notebook (AIC 259,082.23).

**Automation rule** (in `models.py`): a tweet is classed as automated when its source label matches `cowin|vaxblr|vaccinealert|vaccination availability|slothelp|alertbot|alertsbot`, or contains "bot" (the Tweetbot client is excluded). All 24,404 Delta tweets from the five vaccine-slot accounts contain booking terms, compared with 1.7% of unflagged tweets. Accounts are identified by display name because the corpus has no user IDs.

**Optional:** `scripts/colab_roberta_bertopic.py` computes Twitter-RoBERTa sentiment and re-runs BERTopic without automated accounts. It needs a GPU (Colab).

## Data and ethics

The corpus is public and CC0-licensed. Results are reported only in aggregate. Automated accounts are identified by posting application, not by account name.

## Contact

Abdallah Sayed Ali Essa — abdallahsayedali60@gmail.com — ORCID 0009-0001-8378-6832

## License

MIT (code). The data are licensed separately by their provider (CC0).
