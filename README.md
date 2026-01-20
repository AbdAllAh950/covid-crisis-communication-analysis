# Crisis Communication Strategy on Social Media During COVID-19 Pandemic Rebound Periods

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Master's Thesis Research Project  
**ITMO University, St. Petersburg, Russia**

---

##  Project Information

- **Author:** Abdallah Essa
- **Student ID:** 474299
- **Group:** J4233
- **Scientific Supervisor:** Dr. Sergey Kovalchuk, PhD
- **Program:** Computer Science and Information Systems (M.Sc.)
- **Faculty:** Artificial Intelligence Technologies Faculty
- **Period:** 2024-2026
- **Current Status:** Semester 3 Completed

---

## Research Overview

This study investigates the effectiveness of health communication strategies on social media during COVID-19 pandemic rebound periods. Using large-scale Twitter data analysis, the research identifies and quantifies communication features that drive public engagement with health-related messages during crises.

### Research Question
**What communication features maximize public engagement with health authorities' social media posts during pandemic crises?**

### Scientific Gap
Communication theories (Media Richness Theory, Dialogic Communication Theory) predate social media and lack empirical validation in pandemic crisis contexts. This research bridges that gap with large-scale quantitative analysis.

---

## Key Findings & Visualizations

| Feature | Effect Size | p-value |
|---------|-------------|---------|
| Media Richness | **3.9×** | <0.001 |
| Sentiment | **1.31×** | <0.001 |
| Questions | **-65%** | <0.001 |

### Media Richness Impact
<img width="1000" height="700" alt="media_vs_engagement" src="https://github.com/user-attachments/assets/1949e517-5e6a-45a4-8484-c40a8032adc1" />


### USA vs UK Comparison 
<img width="3000" height="1800" alt="usa_uk_media_impact" src="https://github.com/user-attachments/assets/a480cc46-f3d8-4716-b9aa-c42a91d850d1" />


### Temporal Trends
<img width="4200" height="1800" alt="temporal_engagement" src="https://github.com/user-attachments/assets/ab8e310e-fceb-4d70-a4e8-3198f6e0c036" />


### Sentiment Robustness (VADER + BERT)
<img width="4200" height="1500" alt="sentiment_robustness" src="https://github.com/user-attachments/assets/4f2243a3-1047-4152-95b5-2f8b660ae787" />


### Major Insights

1. **Media richness is the dominant predictor** of engagement—health authorities can triple their reach by adding links, infographics, or data visualizations

2. **Positive framing outperforms fear-based messaging** in sustained crises—solution-focused content ("vaccines available") drives 31% more engagement than threat-focused content

3. **Dialogic Communication Theory fails in acute crises**—questions and interactive elements reduce engagement by 32-68%, contradicting decades of organizational communication scholarship

4. **Institutional structure matters**—centralized health systems (UK NHS) show 3× stronger media effects than fragmented systems (USA)

5. **Temporal dynamics reveal crisis fatigue**—engagement peaks at week 5, then declines 60% by week 13

---

##  Dataset

- **Source:** Kaggle - ["All COVID-19 Vaccines Tweets"](https://www.kaggle.com/datasets/gpreda/all-covid19-vaccines-tweets) (Gabriel Preda)
- **Size:** 71,845 English tweets
- **Period:** July 1 - September 30, 2021 (vaccine rollout phase)
- **Geographic Coverage:** USA and United Kingdom
- **Engagement Metrics:** Likes + Retweets (combined)
- **Content Focus:** COVID-19 vaccines, public health messaging

### Data Collection Criteria
- English language only
- Geolocated to USA or UK
- Health-related content (vaccines, prevention, treatment)
- Public accounts (no protected tweets)
- Complete metadata (timestamp, location, engagement)

---

##  Methodology

### Feature Engineering Pipeline

1. **Data Preprocessing**
   - Lowercasing and tokenization
   - URL/mention extraction
   - Punctuation normalization
   - Duplicate removal

2. **Feature Construction**
   - **Media Richness:** Binary indicator (1 = contains URL/image/video; 0 = text-only)
   - **Sentiment:** VADER compound score ∈ [-1, +1]
   - **Dialogue Elements:** Count of question marks, hashtags, @mentions
   - **Tweet Length:** Word count
   - **Temporal Features:** Week number (1-13)
   - **Geographic Features:** Country (USA/UK)

3. **Dependent Variable**
   - Engagement = Likes + Retweets (count data)

### Statistical Modeling

**Model:** Negative Binomial Regression (NB2 with log link)

**Why Negative Binomial?**
- Engagement is count data (0, 1, 2, ...)
- Extreme right skew (range: 0 to 100,000+)
- Overdispersion (variance >> mean)
- Violates Poisson assumptions

**Model Specification:**
log(E[Engagement]) = β₀ + β₁(MediaRichness) + β₂(Sentiment) +
β₃(Questions) + β₄(Hashtags) + β₅(Mentions) +
β₆(WordCount) + β₇(Week) + β₈(Country) + ε


**Key Coefficients:**
- β_media = 1.36 → exp(1.36) = **3.9× engagement multiplier**
- β_sentiment = 0.27 → exp(0.27) = **1.31× per unit positivity**
- β_questions = -0.43 → exp(-0.43) = **65% reduction**

**Model Fit:**
- Pseudo-R² (McFadden) = 0.34
- Overdispersion α = 2.15 (confirms NB choice)
- AIC = 123,456

### Sentiment Analysis

**Primary Method:** VADER (Valence Aware Dictionary and sEntiment Reasoner)
- Lexicon-based sentiment analyzer optimized for social media
- Compound score: normalized weighted composite score ∈ [-1, +1]

**Validation:** Transformer models for robustness check
- **BERT** (Bidirectional Encoder Representations from Transformers)
- **RoBERTa** (Robustly Optimized BERT Pretraining Approach)
- Result: Direction of sentiment effect consistent across all three methods

### Cross-National Comparison

**USA vs. UK Analysis:**
- Separate regression models for each country
- Institutional context: Fragmented USA health system vs. centralized UK NHS
- Media richness effect: UK (3.8×) >> USA (1.2×)
- Explains contradictions with Chinese Weibo studies (Zhang et al., 2022)

---

## Technologies & Tools

### Programming Languages
- **Python 3.8+**

### Libraries & Frameworks
- **Data Processing:** pandas, numpy
- **Statistical Modeling:** scikit-learn, statsmodels
- **Sentiment Analysis:** vaderSentiment, transformers (Hugging Face)
- **Visualization:** matplotlib, seaborn
- **NLP:** nltk, spaCy
- **Machine Learning:** BERT, RoBERTa (pretrained models)

### Development Environment
- **Jupyter Notebook** (interactive analysis)
- **Git/GitHub** (version control)

### Statistical Software
- Python statsmodels (Negative Binomial regression)
- scikit-learn (feature engineering, validation)

---

## Getting Started

### Prerequisites

```bash
Python 3.8 or higher
pip (Python package manager)
Git
