"""Rebuild the paper's dataset exactly as R&D26.ipynb did, plus account-level fields."""
import sys, re
import numpy as np, pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

src, out = sys.argv[1], sys.argv[2]
raw = pd.read_csv(src)
raw['date'] = pd.to_datetime(raw['date'])
print("raw rows:", len(raw), "| columns:", list(raw.columns))
print("date range:", raw['date'].min(), "->", raw['date'].max())

an = SentimentIntensityAnalyzer()
def build(df):
    df = df.copy()
    df['vader'] = df['text'].apply(lambda t: an.polarity_scores(str(t))['compound'])
    df['hashtag_count'] = df['text'].apply(lambda t: len(re.findall(r"#\w+", str(t))))
    df['mention_count'] = df['text'].apply(lambda t: len(re.findall(r"@\w+", str(t))))
    df['has_question'] = df['text'].apply(lambda t: int("?" in str(t)))
    df['has_url'] = df['text'].apply(lambda t: int("http" in str(t)))
    df['engagement'] = df['favorites'] + df['retweets']
    df['tweet_length'] = df['text'].str.len()
    df['log_followers'] = np.log1p(df['user_followers'].fillna(0))
    df['log_friends'] = np.log1p(df['user_friends'].fillna(0))
    df['verified'] = df['user_verified'].astype(str).str.lower().eq('true').astype(int)
    df['user_created'] = pd.to_datetime(df['user_created'], errors='coerce')
    df['account_age_yrs'] = ((df['date'].dt.tz_localize(None) - df['user_created'].dt.tz_localize(None)).dt.days / 365.25).clip(lower=0)
    df['hour'] = df['date'].dt.hour
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
    df['exclamation_count'] = df['text'].astype(str).str.count('!')
    return df

windows = {
    'pre':   ('2021-04-01', '2021-06-30'),
    'delta': ('2021-07-01', '2021-09-30'),
    'post':  ('2021-10-01', '2021-11-30'),
}
frames = []
for name, (a, b) in windows.items():
    # notebook used string comparison on timestamps: <= 'YYYY-MM-DD' means midnight of that day
    m = (raw['date'] >= a) & (raw['date'] <= b)
    w = raw[m].copy()
    n_raw = len(w)
    w = w.drop_duplicates(subset=['text']).dropna(subset=['text'])
    print(f"{name}: raw={n_raw} after dedupe/dropna={len(w)}")
    w['window'] = name
    frames.append(build(w))
df = pd.concat(frames, ignore_index=True)
df.to_pickle(out)
print("saved", out, df.shape)
