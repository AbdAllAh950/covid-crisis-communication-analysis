"""Corrected models: account-level controls, cluster-robust SEs, bot exclusion."""
import sys, warnings, json, numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomialP, Poisson
warnings.filterwarnings('ignore')

BOT_RE = r"cowin|vaxblr|vaccinealert|vaccination availability|slothelp|alertbot|alertsbot"
def is_bot(s):
    s = s.fillna('')
    generic = s.str.contains('bot', case=False) & ~s.str.contains('tweetbot', case=False)
    return s.str.contains(BOT_RE, case=False, regex=True) | generic

CORE = ['vader', 'has_url', 'has_question', 'hashtag_count', 'mention_count']
CTRL = ['tweet_len10', 'log_followers', 'log_friends', 'verified', 'account_age_yrs', 'is_weekend']

def prep(d):
    d = d.copy()
    d['tweet_len10'] = d['tweet_length'] / 10.0
    d['account_age_yrs'] = d['account_age_yrs'].fillna(d['account_age_yrs'].median())
    return d.reset_index(drop=True)

def fit_nb(d, cols, cluster=True):
    y = d['engagement'].values.astype(float)
    X = sm.add_constant(d[cols].astype(float))
    sp = np.append(Poisson(y, X).fit(disp=0, maxiter=300).params.values, 1.0)
    kw = dict(cov_type='cluster', cov_kwds={'groups': pd.factorize(d['user_name'].fillna('NA'))[0]}) if cluster else {}
    r = NegativeBinomialP(y, X, p=2).fit(start_params=sp, method='newton', maxiter=300, disp=0, **kw)
    return r

def table(r, cols):
    names = ['const'] + cols + ['alpha']
    lo, hi = r.conf_int().values.T
    t = pd.DataFrame({'coef': r.params.values, 'IRR': np.exp(r.params.values), 'CI_lo': np.exp(lo), 'CI_hi': np.exp(hi),
                      'SE': r.bse.values, 'p': r.pvalues.values}, index=names)
    return t

if __name__ == '__main__':
    df = pd.read_pickle(sys.argv[1]); outdir = sys.argv[2]
    df['bot'] = is_bot(df['source'])
    d = prep(df[df.window == 'delta'])
    nb_ = d[~d.bot].reset_index(drop=True)
    print(f"Delta: N={len(d)} | bot tweets={d.bot.sum()} ({d.bot.mean()*100:.1f}%) | non-bot N={len(nb_)} | users={nb_.user_name.nunique()}")
    specs = {
      'M0_paper_spec_full':        (d,   CORE + ['tweet_len10'], False),
      'M1_controls_full_cluster':  (d,   CORE + CTRL, True),
      'M2_paper_spec_nobot_clust': (nb_, CORE + ['tweet_len10'], True),
      'M3_controls_nobot_cluster': (nb_, CORE + CTRL, True),
    }
    res = {}
    for name, (data, cols, cl) in specs.items():
        r = fit_nb(data, cols, cl)
        t = table(r, cols); res[name] = t
        t.to_csv(f"{outdir}/{name}.csv")
        print(f"\n=== {name} | N={len(data)} | AIC={r.aic:.2f} | converged={r.mle_retvals['converged']} | SE={'cluster(user)' if cl else 'model-based'}")
        print(t.loc[CORE + [c for c in cols if c not in CORE]][['IRR','CI_lo','CI_hi','p']].round(4).to_string())
