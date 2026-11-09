import sys, warnings, numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
sys.path.insert(0, sys.argv[3]); from models import *
warnings.filterwarnings('ignore')
df = pd.read_pickle(sys.argv[1]); outdir = sys.argv[2]
df['bot'] = is_bot(df['source'])
d = prep(df[(df.window=='delta') & (~df.bot)])
cols = CORE + CTRL
print(f"non-bot Delta: N={len(d)}, zero share={(d.engagement==0).mean():.3f}")

# R1: drop heavy posters (users with >50 tweets in window)
cnt = d.user_name.map(d.user_name.value_counts())
d1 = d[cnt <= 50].reset_index(drop=True)
r1 = fit_nb(d1, cols); t1 = table(r1, cols); t1.to_csv(f"{outdir}/R1_nobot_le50tweets.csv")
print(f"\nR1 users<=50 tweets: N={len(d1)} AIC={r1.aic:.1f}"); print(t1.loc[CORE][['IRR','CI_lo','CI_hi','p']].round(4).to_string())

# R2: one random tweet per user (removes within-user dependence entirely)
d2 = d.sample(frac=1, random_state=42).drop_duplicates('user_name').reset_index(drop=True)
r2 = fit_nb(d2, cols, cluster=False); t2 = table(r2, cols); t2.to_csv(f"{outdir}/R2_nobot_one_per_user.csv")
print(f"\nR2 one tweet per user: N={len(d2)} AIC={r2.aic:.1f}"); print(t2.loc[CORE][['IRR','CI_lo','CI_hi','p']].round(4).to_string())

# R3: ZINB, count = CORE+CTRL, inflation = log_followers + tweet_len10, cluster SE
y = d['engagement'].values.astype(float)
X = sm.add_constant(d[cols].astype(float)); Z = sm.add_constant(d[['log_followers','tweet_len10']].astype(float))
nbr = fit_nb(d, cols, cluster=False)
sp = np.concatenate([np.zeros(Z.shape[1]), nbr.params.values])
groups = pd.factorize(d['user_name'].fillna('NA'))[0]
m = ZeroInflatedNegativeBinomialP(y, X, exog_infl=Z, inflation='logit', p=2)
r3 = m.fit(start_params=sp, method='bfgs', maxiter=2000, disp=0, cov_type='cluster', cov_kwds={'groups': groups})
names = ['infl_const','infl_log_followers','infl_tweet_len10','const'] + cols + ['alpha']
lo, hi = r3.conf_int().values.T
t3 = pd.DataFrame({'coef': r3.params.values, 'IRR': np.exp(r3.params.values), 'CI_lo': np.exp(lo), 'CI_hi': np.exp(hi), 'p': r3.pvalues.values}, index=names)
t3.to_csv(f"{outdir}/R3_zinb_nobot_cluster.csv")
print(f"\nR3 ZINB: AIC={r3.aic:.1f} (NB same data AIC={nbr.aic:.1f}) converged={r3.mle_retvals['converged']} anyNaN_SE={bool(np.isnan(r3.bse).any())}")
print(t3.loc[CORE + ['infl_const','infl_log_followers','infl_tweet_len10']][['coef','IRR','CI_lo','CI_hi','p']].round(4).to_string())
