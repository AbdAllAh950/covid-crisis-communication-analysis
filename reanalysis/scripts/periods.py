import sys, warnings, numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, sys.argv[3]); from models import *
warnings.filterwarnings('ignore')
df = pd.read_pickle(sys.argv[1]); outdir = sys.argv[2]
df['bot'] = is_bot(df['source'])
cols = CORE + CTRL
order = ['pre','delta','post']; label = {'pre':'Pre-Delta (Apr–Jun)','delta':'Delta (Jul–Sep)','post':'Post-Delta (Oct–Nov 23)'}

# Descriptives (Table 1) — all tweets and non-bot
rows = []
for w in order:
    for tag, sub in [('all', df[df.window==w]), ('non-bot', df[(df.window==w)&(~df.bot)])]:
        e = sub.engagement
        rows.append({'window': label[w], 'sample': tag, 'N': len(sub), 'users': sub.user_name.nunique(), 'bot_share_%': round(df[df.window==w].bot.mean()*100,1) if tag=='all' else None,
                     'mean_eng': round(e.mean(),2), 'median_eng': e.median(), 'zero_%': round((e==0).mean()*100,1),
                     'URL_%': round(sub.has_url.mean()*100,1), 'question_%': round(sub.has_question.mean()*100,1),
                     'hashtags_mean': round(sub.hashtag_count.mean(),2), 'mentions_mean': round(sub.mention_count.mean(),2),
                     'vader_mean': round(sub.vader.mean(),3), 'verified_%': round(sub.verified.mean()*100,1)})
t1 = pd.DataFrame(rows); t1.to_csv(f"{outdir}/T1_descriptives.csv", index=False); print(t1.to_string(index=False))

# Per-window models (non-bot, controls, cluster)
per = {}
for w in order:
    d = prep(df[(df.window==w)&(~df.bot)]); r = fit_nb(d, cols); t = table(r, cols); per[w] = t
    t.to_csv(f"{outdir}/P_{w}.csv")
    print(f"\n{label[w]}: N={len(d)} AIC={r.aic:.1f} conv={r.mle_retvals['converged']}")
    print(t.loc[CORE][['coef','IRR','CI_lo','CI_hi','p']].round(4).to_string())

# Pooled interaction model: CORE x window (ref = delta), controls x window too, cluster by user
d = prep(df[~df.bot]).copy()
for w in ['pre','post']:
    d[f'w_{w}'] = (d.window==w).astype(int)
    for c in cols: d[f'{c}_x_{w}'] = d[c]*d[f'w_{w}']
icols = cols + ['w_pre','w_post'] + [f'{c}_x_{w}' for w in ['pre','post'] for c in cols]
r = fit_nb(d, icols); t = table(r, icols); t.to_csv(f"{outdir}/P_pooled_interactions.csv")
print(f"\nPooled interaction model (ref=Delta): N={len(d)} conv={r.mle_retvals['converged']}")
print("Differences vs Delta (coef diff, p from cluster-robust SE):")
print(t.loc[[f'{c}_x_{w}' for w in ['pre','post'] for c in CORE]][['coef','SE','p']].round(4).to_string())
