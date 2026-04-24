# Crisis Communication Effectiveness on Social Media During COVID-19 Pandemic Rebound Periods

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-research%20project-green.svg)]()

**Master's Thesis Research Project — ITMO University, Saint Petersburg**

---

## Table of Contents

- [Project Information](#project-information)
- [Quick Start](#quick-start)
- [Research Overview](#research-overview)
- [Key Findings](#key-findings)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results Summary](#results-summary)
- [Technologies](#technologies)
- [Repository Structure](#repository-structure)
- [How to Reproduce](#how-to-reproduce)
- [Citation](#citation)
- [License](#license)

---

## Project Information

| Field | Details |
|-------|---------|
| **Author** | Abdallah Sayed Ali Essa |
| **Student ID** | 474299 |
| **Group** | J4233 |
| **Supervisor** | Sergey V. Kovalchuk, PhD, Associate Professor |
| **Faculty** | Faculty of Artificial Intelligence Technologies |
| **Program** | Big Data and Machine Learning (M.Sc.) |
| **University** | ITMO University, Saint Petersburg, Russia |
| **Period** | 2024–2026 |
| **Status** | Semester 4 — Research Project Defense |

---

## Quick Start

This repository contains the full implementation of a quantitative computational analysis of crisis communication on Twitter during the COVID-19 Delta variant rebound (July–September 2021). The research applies a multi-method pipeline combining sentiment analysis, neural topic modeling, and count-data regression to 71,845 vaccine-related tweets.

```bash
# Clone the repository
git clone https://github.com/AbdAllAh950/covid-crisis-communication-analysis.git
cd covid-crisis-communication-analysis

# Install dependencies
pip install -r requirements.txt

# Run the main analysis
jupyter notebook R_D26.ipynb
```

---

## Research Overview

### Research Question

**Which characteristics of COVID-19 vaccine-related tweets predicted audience engagement during the Delta variant rebound period of July–September 2021?**

### Scientific Gap

Three gaps in the existing literature motivate this study:

1. **Temporal coverage**: Most COVID-19 Twitter studies cover the early pandemic (2020–early 2021) and end before the Delta rebound. The Delta window is substantively distinct (vaccines available, fatigue, breakthrough infections, cardiac safety debates).

2. **Methodological monoculture**: Existing work relies on a single sentiment tool (typically VADER) without robustness checks against transformer-based alternatives.

3. **Theoretical isolation**: Studies typically test one theoretical framework at a time. This research simultaneously tests Media Richness Theory, Dialogic Communication Theory, and Situational Crisis Communication Theory on the same corpus.

### Hypotheses

| # | Hypothesis | Prediction | Result |
|---|------------|------------|--------|
| H1 | Media richness (URL presence) increases engagement | +  | ✅ Supported (IRR = 4.56) |
| H2 | Dialogic markers (questions, hashtags, mentions) increase engagement | +  | ❌ Rejected (all effects negative) |
| H3 | Positive sentiment increases engagement | +  | ✅ Supported (IRR = 1.31–1.91) |
| H4 | Findings are robust across sentiment operationalizations | —  | ✅ Confirmed |

---

## Key Findings

### 1. Media Richness Dominates Engagement

Tweets containing embedded URLs receive **4.6× more engagement** than text-only tweets, controlling for sentiment, dialogic markers, topic, and temporal features. This supports Media Richness Theory (Daft & Lengel, 1986) in a high-uncertainty crisis context.

### 2. Dialogic Communication Theory Fails Under Crisis

All three behavioral markers of dialogic communication are **negatively** associated with engagement:
- Questions: 32% reduction in engagement
- Each hashtag: 6–13% reduction
- Each mention: 8–12% reduction

This contradicts the direct prediction of Dialogic Communication Theory (Kent & Taylor, 1998), suggesting the theory requires refinement for the algorithmically mediated, crisis-accelerated social media environment.

### 3. Sentiment Effect Is Tool-Dependent

Positive sentiment increases engagement consistently, but the magnitude varies by operationalization:
- VADER sentiment: IRR = 1.31
- Twitter-RoBERTa sentiment: IRR = 1.91

This argues for routinely reporting sentiment results under multiple tools rather than committing to a single specification.

---

## Dataset

| Attribute | Value |
|-----------|-------|
| **Source** | [All COVID-19 Vaccines Tweets](https://www.kaggle.com/datasets/gpreda/all-covid19-vaccines-tweets) by Gabriel Preda (Kaggle, CC0) |
| **Raw size** | 72,847 tweets (Dec 2020 – Nov 2021) |
| **Analytic window** | July 1 – September 30, 2021 (Delta rebound) |
| **Final corpus** | **71,845 tweets** (after deduplication) |
| **USA subset** | 3,838 tweets |
| **UK subset** | 1,606 tweets |
| **Language** | English |
| **License** | CC0 (Public Domain) |

---

## Methodology

The research follows a **multi-method analytical pipeline** with four complementary components:

### 1. Sentiment Analysis (Dual-Model)

- **Primary:** VADER compound score (lexicon-based, [-1, +1])
- **Robustness:** `cardiffnlp/twitter-roberta-base-sentiment-latest` (transformer)
- **Cross-model correlation:** r = 0.456 (p < 0.001)

### 2. Topic Modeling (BERTopic)

- **Embeddings:** `all-MiniLM-L6-v2` (384-dim)
- **Dimensionality reduction:** UMAP (5D)
- **Clustering:** HDBSCAN (min_cluster_size = 150)
- **Topic representation:** c-TF-IDF with n-grams (1–2)
- **Output:** 19 substantive topics + residual outlier class

### 3. Feature Engineering

Five feature families:
1. **Theoretical variables:** Sentiment, media richness (URL), dialogic markers (questions, hashtags, mentions)
2. **Medical NER:** Vaccine brands (17 surface forms → 8 canonical) + side effects (25 terms)
3. **Temporal:** Hour, day of week, weekend indicator
4. **Stylistic intensity:** Emoji count, exclamation count, URL count
5. **Outcome:** Engagement = favorites + retweets

### 4. Statistical Modeling

Three specifications fit sequentially:

| Model | Predictors | Sentiment | AIC |
|-------|-----------|-----------|-----|
| Baseline NB | 5 | VADER | 259,082.23 |
| Upgraded NB | 7 | + RoBERTa | 258,520.34 |
| **Final ZINB** | **11** | **RoBERTa** | **254,733.93** |

**Total AIC improvement:** 4,348 points (strong support for the upgraded specification).

### 5. Robustness Checks

- VADER vs RoBERTa sentiment operationalization
- Three-model comparison (VADER, DistilBERT-SST-2, DistilRoBERTa) on random 1,000-tweet sample
- All substantive findings preserved across specifications

---

## Results Summary

### Final ZINB Regression (Count Component)

| Predictor | Coefficient | IRR | p-value |
|-----------|-------------|-----|---------|
| Media richness (URL) | **+1.517** | **4.56** | < 0.001 |
| RoBERTa sentiment | +0.668 | 1.91 | < 0.001 |
| Side-effect mention | +0.184 | 1.20 | < 0.05 |
| Vaccine count | +0.093 | 1.10 | < 0.001 |
| Weekend indicator | +0.064 | 1.07 | < 0.05 |
| Mention count | −0.116 | 0.89 | < 0.001 |
| Hashtag count | −0.139 | 0.87 | < 0.001 |
| Question presence | **−0.271** | **0.76** | < 0.001 |

### Topic Distribution (BERTopic)

Top topics by volume:
1. Indian vaccination booking notifications (14,874 tweets)
2. Moderna-related content (11,943 tweets)
3. Covaxin / OCGN discussion (7,205 tweets)
4. Booking availability (Bengaluru, etc.) (4,303 tweets)
5. Sinopharm / Sinovac (5,093 tweets)

### USA vs UK Comparison

| Country | N | Mean Engagement | Mean Sentiment |
|---------|---|-----------------|----------------|
| United Kingdom | 1,606 | 6.88 | +0.106 |
| United States | 3,838 | 5.39 | +0.090 |

UK tweets show **28% higher** average engagement than USA tweets.

---

## Technologies

### Core Stack
- **Python 3.8+**
- **Jupyter Notebook** (Google Colab)

### Libraries
- **Data:** `pandas`, `numpy`
- **NLP:** `nltk`, `vaderSentiment`, `transformers` (Hugging Face)
- **Topic modeling:** `bertopic`, `sentence-transformers`, `umap-learn`, `hdbscan`
- **Statistical modeling:** `statsmodels` (NB, ZINB)
- **Visualization:** `matplotlib`, `seaborn`, `plotly`

### Models Used
- `all-MiniLM-L6-v2` (sentence embeddings)
- `cardiffnlp/twitter-roberta-base-sentiment-latest` (Twitter-specific sentiment)
- `distilbert-base-uncased-finetuned-sst-2-english` (robustness check)

---

## Repository Structure

```
covid-crisis-communication-analysis/
│
├── R_D26.ipynb                      # Main analysis notebook (full pipeline)
├── analysis_results.py              # Helper script for summary statistics
├── Abdallah_Essa.pptx               # Research project presentation
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── LICENSE                          # MIT License
```

---

## How to Reproduce

### Prerequisites

```bash
Python 3.8 or higher
pip (package manager)
Git
A GPU (recommended for transformer inference)
```

### Installation

```bash
# Clone
git clone https://github.com/AbdAllAh950/covid-crisis-communication-analysis.git
cd covid-crisis-communication-analysis

# Install dependencies
pip install -r requirements.txt
```

### Data Download

Download the Kaggle dataset from:
https://www.kaggle.com/datasets/gpreda/all-covid19-vaccines-tweets

Place `vaccination_all_tweets.csv` in the project root.

### Run the Analysis

```bash
jupyter notebook R_D26.ipynb
```

Or open directly in Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbdAllAh950/covid-crisis-communication-analysis/blob/main/R_D26.ipynb)

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{essa2026crisis,
  title   = {Crisis Communication Effectiveness on Social Media 
             During COVID-19 Pandemic Rebound Periods},
  author  = {Essa, Abdallah Sayed Ali},
  school  = {ITMO University},
  year    = {2026},
  type    = {Master's Thesis},
  address = {Saint Petersburg, Russia}
}
```

---

## Theoretical Framework

This research engages three established communication theories:

- **Media Richness Theory** (Daft & Lengel, 1986) — Predicts richer messages outperform leaner ones under uncertainty. ✅ **Supported**.
- **Dialogic Communication Theory** (Kent & Taylor, 1998) — Predicts dialogic markers boost engagement. ❌ **Rejected** (markers reduce engagement).
- **Situational Crisis Communication Theory** (Coombs, 2007) — Provides the broader crisis-management backdrop. ✅ **Consistent with findings**.

---

## Practical Implications

For public health communicators during pandemic rebounds:

1. **Invest in rich, link-bearing content** — URL presence multiplies engagement by 4.6×
2. **Reconsider reflexive use of dialogic markers** — questions and hashtags reduce reach in crisis contexts
3. **Calibrate tone toward mild positivity** — small positive effect of sentiment is consistent

---

## Limitations

1. No follower counts available — limits ability to control for account reach
2. Corpus heavily weighted toward Indian vaccination booking notifications
3. English-language tweets only
4. Self-reported user locations (noisy geographic classification)

---

## Future Work

This Master's thesis establishes the methodological infrastructure for a doctoral research program at ITMO University that will extend the framework to:

- Other pandemic phases (Omicron, booster rollout, endemic phase)
- Multilingual corpora (Arabic, Russian, Spanish)
- Other crisis types (climate, security, economic)
- Additional platforms (Reddit, Telegram, Weibo)

---

## Contact

For questions or collaboration inquiries:

**Abdallah Sayed Ali Essa**
- GitHub: [@AbdAllAh950](https://github.com/AbdAllAh950)
- Institution: ITMO University, Saint Petersburg

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Dr. Sergey V. Kovalchuk** for research supervision
- **Gabriel Preda** for maintaining the CC0-licensed Kaggle dataset
- **Cardiff NLP** for the Twitter-RoBERTa sentiment model
- **Maarten Grootendorst** for the BERTopic framework
- **ITMO University** Big Data and Machine Learning program

---

*Last updated: April 24, 2026*
