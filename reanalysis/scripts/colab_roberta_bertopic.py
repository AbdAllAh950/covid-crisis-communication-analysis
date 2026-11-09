# ============================================================
# Colab: Runtime > Change runtime type > T4 GPU, then Run all.
# Takes ~20 minutes. Downloads 3 files at the end:
#   roberta_scores.csv, bertopic_assignments.csv, bertopic_topic_info.csv
# Upload all three back to Claude.
# ============================================================
!pip -q install kagglehub transformers bertopic sentence-transformers umap-learn hdbscan
import re, kagglehub, pandas as pd, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# ---------- 1. Same windows / cleaning as the paper ----------
path = kagglehub.dataset_download("gpreda/all-covid19-vaccines-tweets")
raw = pd.read_csv(path + "/vaccination_all_tweets.csv"); raw['date'] = pd.to_datetime(raw['date'])
windows = {'pre': ('2021-04-01','2021-06-30'), 'delta': ('2021-07-01','2021-09-30'), 'post': ('2021-10-01','2021-11-30')}
frames = []
for name,(a,b) in windows.items():
    w = raw[(raw['date'] >= a) & (raw['date'] <= b)].drop_duplicates(subset=['text']).dropna(subset=['text']).copy()
    w['window'] = name; frames.append(w[['id','window','text','source']])
df = pd.concat(frames, ignore_index=True)
print(df.groupby('window').size())

# ---------- 2. Twitter-RoBERTa sentiment (all tweets, 3 windows) ----------
def prep(t):
    return " ".join('@user' if (w.startswith('@') and len(w) > 1) else ('http' if w.startswith('http') else w) for w in str(t).split(" "))
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).cuda().eval().half()
texts = df['text'].map(prep).tolist(); out = []
for i in range(0, len(texts), 256):
    enc = tok(texts[i:i+256], padding=True, truncation=True, max_length=128, return_tensors='pt').to('cuda')
    with torch.no_grad(): out.append(softmax(model(**enc).logits.float().cpu().numpy(), axis=1))
    if i % 25600 == 0: print('RoBERTa', i, '/', len(texts))
p = np.vstack(out)
df['roberta_neg'], df['roberta_neu'], df['roberta_pos'] = p[:,0], p[:,1], p[:,2]
df['roberta_sentiment'] = p[:,2] - p[:,0]
df[['id','window','roberta_neg','roberta_neu','roberta_pos','roberta_sentiment']].to_csv('roberta_scores.csv', index=False)
del model; torch.cuda.empty_cache()

# ---------- 3. BERTopic on NON-BOT Delta tweets (same settings as the paper) ----------
BOT_RE = r"cowin|vaxblr|vaccinealert|vaccination availability|slothelp|alertbot|alertsbot"
src = df['source'].fillna('')
df['bot'] = src.str.contains(BOT_RE, case=False, regex=True) | (src.str.contains('bot', case=False) & ~src.str.contains('tweetbot', case=False))
def clean_for_topics(t):
    t = re.sub(r"http\S+|www\S+", "", str(t)); t = re.sub(r"@\w+", "", t); t = re.sub(r"^RT\s+", "", t); t = re.sub(r"#", "", t)
    return " ".join(t.split()).strip()
d = df[(df.window=='delta') & (~df.bot)].copy()
d['clean_text'] = d['text'].map(clean_for_topics)
fit_docs = d.drop_duplicates('clean_text'); fit_docs = fit_docs[fit_docs['clean_text'].str.split().str.len() >= 5]
docs = fit_docs['clean_text'].tolist(); print('BERTopic docs:', len(docs))

from sentence_transformers import SentenceTransformer
from umap import UMAP; from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
emb_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
emb = emb_model.encode(docs, batch_size=256, show_progress_bar=True)
tm = BERTopic(embedding_model=emb_model,
              umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42),
              hdbscan_model=HDBSCAN(min_cluster_size=150, min_samples=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True),
              vectorizer_model=CountVectorizer(stop_words='english', min_df=5, ngram_range=(1,2)), verbose=True)
topics, _ = tm.fit_transform(docs, emb)
n_initial = len(set(topics)) - (1 if -1 in topics else 0); n_out_initial = int(np.sum(np.array(topics) == -1))
topics = tm.reduce_outliers(docs, topics, strategy='embeddings', embeddings=emb, threshold=0.5)
tm.update_topics(docs, topics=topics, vectorizer_model=CountVectorizer(stop_words='english', min_df=5, ngram_range=(1,2)))
tm.reduce_topics(docs, nr_topics=20)
topics = tm.topics_
print('initial topics:', n_initial, 'initial outliers:', n_out_initial, '| final outliers:', int(np.sum(np.array(topics) == -1)))
m = dict(zip(fit_docs['clean_text'], topics))
d['topic'] = d['clean_text'].map(m).fillna(-2).astype(int)   # -2 = too short / not modeled
d[['id','topic']].to_csv('bertopic_assignments.csv', index=False)
info = tm.get_topic_info(); info['initial_topics'] = n_initial; info['initial_outliers'] = n_out_initial; info['n_docs_modeled'] = len(docs)
info.to_csv('bertopic_topic_info.csv', index=False); print(info[['Topic','Count','Name']].to_string())

from google.colab import files
for f in ['roberta_scores.csv','bertopic_assignments.csv','bertopic_topic_info.csv']: files.download(f)
