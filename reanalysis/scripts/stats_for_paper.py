import sys, json, re, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[2]); from models import is_bot
df = pd.read_pickle(sys.argv[1]); df['bot'] = is_bot(df['source'])
out = {}
for w in ['pre','delta','post']:
    s = df[df.window==w]
    out[w] = {'date_min': str(s.date.min()), 'date_max': str(s.date.max()), 'N': len(s), 'bot_N': int(s.bot.sum()),
              'bot_pct': round(s.bot.mean()*100,1), 'bot_mean_eng': round(s[s.bot].engagement.mean(),2),
              'bot_zero_pct': round((s[s.bot].engagement==0).mean()*100,1), 'nonbot_mean_eng': round(s[~s.bot].engagement.mean(),2),
              'bot_accounts': int(s[s.bot].user_name.nunique()), 'nonbot_N': int((~s.bot).sum()), 'nonbot_users': int(s[~s.bot].user_name.nunique())}
d = df[df.window=='delta']
top = d[d.bot].user_name.value_counts().head(5)
out['delta_top_bot_accounts'] = {k:int(v) for k,v in top.items()}
out['delta_top5_bot_share_of_bot'] = round(top.sum()/d.bot.sum()*100,1)
kw = r"slot|pincode|pin code|18\+|45\+|dose ?1|dose ?2|cowin"
out['delta_slot_accounts_tweets'] = int(top.sum())
out['delta_kw_in_slot_accounts_n'] = int(d[d.user_name.isin(top.index)].text.str.contains(kw, case=False, regex=True).sum())
out['delta_kw_in_bot_pct'] = round(d[d.bot].text.str.contains(kw, case=False, regex=True).mean()*100,1)
out['delta_kw_in_nonbot_pct'] = round(d[~d.bot].text.str.contains(kw, case=False, regex=True).mean()*100,1)
nb = d[~d.bot]
v = nb.vader
out['delta_nonbot_vader_classes'] = {'neg_pct': round((v< -0.05).mean()*100,1), 'neu_pct': round(((v>=-0.05)&(v<=0.05)).mean()*100,1), 'pos_pct': round((v>0.05).mean()*100,1)}
out['delta_nonbot_followers_median'] = float(nb.user_followers.median())
out['delta_nonbot_tweets_per_user_median'] = float(nb.user_name.value_counts().median())
out['delta_nonbot_users_gt50'] = int((nb.user_name.value_counts()>50).sum())
out['corr_logfollowers_engagement_nonbot'] = round(np.corrcoef(nb.log_followers, np.log1p(nb.engagement))[0,1],3)
print(json.dumps(out, indent=1, ensure_ascii=False))
json.dump(out, open(sys.argv[3],'w'), indent=1, ensure_ascii=False)
