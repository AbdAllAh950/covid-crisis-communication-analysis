# Corrections (v2, 6 November 2026)

This repository supports the paper by Abdallah Sayed Ali Essa and Sergey V. Kovalchuk. Version 1 was posted on SocArXiv on 29 October 2026 as *The Dialogic Paradox Across Platforms* (doi:10.31235/osf.io/fb7xy_v1). A re-analysis of the raw data and the original notebooks found errors that change its main conclusions. Version 2 is titled *Who Speaks Matters More Than How: Automated Accounts, Account Reach, and Message Features in COVID-19 Vaccine Tweets Around the Delta Wave*, and it replaces v1.

The original notebooks (`R&D26.ipynb`, `extended_analysis/`) are kept unchanged for transparency. Their outputs should not be cited. The corrected analysis is in `reanalysis/`.

## What was wrong in v1

1. **Automated accounts were not removed.** Automated vaccine-slot notification accounts (source labels such as CowinAlertsBot, Cowin Vaccination Availability, VaxBlr and cowin_vaccine_app) produced 34.0% of Delta-window tweets (24,415 of 71,845) and almost no engagement. Including them inflated the link/media effect and distorted the dialogic estimates.
2. **Account reach was not controlled, and v1 wrongly said it could not be.** v1 stated that the corpus lacks follower counts. It contains `user_followers`, `user_friends`, `user_verified` and `user_created`.
3. **Table 2 in v1 did not correspond to a single fitted model.** The question, hashtag and mention coefficients came from a 7-predictor ZINB that had followers in its inflation component. The link coefficient came from an 11-predictor ZINB whose standard errors failed (p-values were NaN). Several control coefficients and p-values (side effects, vaccine count, weekend, exclamation marks, tweet length, intercept) do not appear in any code output. The AIC reported for the "final ZINB" (254,733.93) belongs to the 7-predictor model. The claimed 4,348-point AIC improvement was wrong; the code reports 570.88 for the model actually described.
4. **Table 1 in v1 did not match the data.** Examples: v1 gave mean engagement of 5.2 (pre-Delta) and 4.9 (post-Delta); the data give 16.06 and 15.78. v1 gave link share of 69.8% and question share of 14.7% for Delta; the data give 82.5% and 5.1%. v1 labelled the pre/post sample sizes as cleaned, but 78,410 and 30,796 are raw counts.
5. **The "phase-dependent reversal" of the link effect is withdrawn.** It came from per-window GLMs with dispersion fixed at 1 and no account controls (Wald z = 61.9). In the corrected pooled interaction model, the link association is positive before and during Delta and imprecise afterwards, and the differences are not significant (p = .16 and .24).
6. **The claim that the "dialogic paradox held across all three windows" is withdrawn.** After correction, questions and hashtags are modestly negative, @mentions are unrelated (Delta), and the cue effects shift across windows.
7. **Literature anchor.** v1 cited findings it attributed to "Zhang et al. (2022)" (Hebei rebound, 3,489 posts, dialogic loop IRR = 0.23). These came from an unpublished manuscript. The published Zhang et al. (2022, *Frontiers in Public Health*, doi:10.3389/fpubh.2022.807459) studies Hubei government accounts from December 2019 to March 2020 and reports **positive** dialogic-loop effects. v2 cites only published work. v1 also misdescribed Bonsón et al. (2019).
8. **Removed in v2.** Twitter-RoBERTa results, the medical-entity features and the USA–UK comparison. RoBERTa scores could not be recomputed under the corrected specification for v2. `reanalysis/scripts/colab_roberta_bertopic.py` produces them.

## Corrected main results (Delta window, automated accounts removed, account controls, SEs clustered by account)

| Predictor | v1 reported | v2 corrected IRR [95% CI] |
|---|---|---|
| Link or attached media | 4.56 | 1.40 [1.19, 1.64] |
| Sentiment (VADER) | 1.37 / 1.91 (RoBERTa) | 1.48 [1.28, 1.70] |
| Question | 0.76 | 0.85 [0.75, 0.97] |
| Hashtags (per tag) | 0.87 | 0.91 [0.87, 0.95] |
| @Mentions (per mention) | 0.89 | 1.03 [0.98, 1.08], n.s. |
| Log followers | not modelled | 1.55 [1.49, 1.63] (≈ +36% per doubling) |

The naive specification (all tweets, no account controls, conventional SEs) reproduces the original notebook exactly (AIC 259,082.23), so the differences above come from the corrections, not from the data or code base.
