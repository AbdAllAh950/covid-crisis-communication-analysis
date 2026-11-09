import sys, warnings, numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomialP, Poisson
warnings.filterwarnings('ignore')
df = pd.read_pickle(sys.argv[1]); d = df[df.window=='delta'].reset_index(drop=True)
y = d['engagement'].values.astype(float)

def fit_nb(cols):
    X = sm.add_constant(d[cols].astype(float))
    pois = Poisson(y, X).fit(disp=0, maxiter=200)
    sp = np.append(pois.params.values, 1.0)
    r = NegativeBinomialP(y, X, p=2).fit(start_params=sp, method='newton', maxiter=200, disp=0)
    return r

print("Delta descriptives: N", len(d), "| mean eng", round(y.mean(),2), "| zero%", round((y==0).mean()*100,1),
      "| median", np.median(y), "| URL%", round(d.has_url.mean()*100,1), "| Q%", round(d.has_question.mean()*100,1),
      "| hashtags", round(d.hashtag_count.mean(),2), "| mentions", round(d.mention_count.mean(),2), "| VADER", round(d.vader.mean(),3))
r = fit_nb(['vader','has_url','has_question','hashtag_count','mention_count'])
print("\nBaseline NB (5 features) AIC:", round(r.aic,2), "| converged:", r.mle_retvals['converged'], "| notebook: 259082.23")
print(pd.DataFrame({'coef':r.params.round(4),'IRR':np.exp(r.params).round(3),'p':r.pvalues.round(4)}).to_string())

r6 = fit_nb(['vader','has_url','has_question','hashtag_count','mention_count','tweet_length'])
print("\nNotebook cell-67 spec (VADER + 4 features + tweet_length) AIC:", round(r6.aic,2), "| notebook: 259082.23")
print(pd.DataFrame({'coef':r6.params.round(4),'IRR':np.exp(r6.params).round(4),'p':r6.pvalues.round(4)}).to_string())
