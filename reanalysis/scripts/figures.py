import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
sys.path.insert(0, sys.argv[2]); from models import is_bot
df = pd.read_pickle(sys.argv[1]); out = sys.argv[3]; res = sys.argv[4]
df['bot'] = is_bot(df['source'])
BLUE, ORANGE, AQUA, GRAY, INK, INK2, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#a3a29c', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': '#9a9993', 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False})

# ---------- Figure 1: weekly volume, automated vs other ----------
wk = df.assign(week=df.date.dt.to_period('W-SUN').dt.start_time).groupby(['week','bot']).size().unstack(fill_value=0)
full = pd.date_range(wk.index.min(), wk.index.max(), freq='7D'); wk = wk.reindex(full, fill_value=0)
fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=300)
xc = wk.index + pd.Timedelta(days=3.5)
ax.bar(xc, wk[False], width=5.6, color=BLUE, label='Other accounts', zorder=3)
ax.bar(xc, wk[True], width=5.6, bottom=wk[False], color=GRAY, label='Automated vaccine-slot accounts', zorder=3, edgecolor='white', linewidth=0.6)
top = wk.sum(axis=1).max()
for (a, b, lab) in [('2021-04-01','2021-06-30','Pre-Delta'), ('2021-07-01','2021-09-30','Delta'), ('2021-10-01','2021-11-24','Post-Delta')]:
    A, B = pd.Timestamp(a), pd.Timestamp(b)
    ax.text(A + (B - A) / 2, top * 1.06, lab, ha='center', va='bottom', color=INK2, fontsize=8.5)
    ax.axvline(A, color='#9a9993', lw=0.8, ls=(0, (3, 3)), zorder=2)
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.tick_params(axis='x', labelsize=8)
ax.set_ylabel('Tweets per week'); ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
ax.set_ylim(0, top * 1.16)
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(f'{out}/fig1_weekly_volume.png', facecolor='white'); plt.close(fig)

# ---------- Figure 2: naive vs corrected IRRs (Delta) ----------
labels = {'has_url': 'Link or attached media', 'vader': 'Positive sentiment (VADER)', 'has_question': 'Question', 'hashtag_count': 'Hashtags (per tag)', 'mention_count': '@Mentions (per mention)'}
order = ['has_url', 'vader', 'has_question', 'hashtag_count', 'mention_count']
M0 = pd.read_csv(f'{res}/M0_paper_spec_full.csv', index_col=0); M3 = pd.read_csv(f'{res}/M3_controls_nobot_cluster.csv', index_col=0)
fig, ax = plt.subplots(figsize=(6.6, 3.4), dpi=300)
for j, (t, col, mk, lab) in enumerate([(M0, ORANGE, 's', 'Naive model: all tweets, no account controls'), (M3, BLUE, 'o', 'Corrected model: automated accounts removed, account controls, clustered SEs')]):
    y = np.arange(len(order)) + (0.17 if j == 0 else -0.17)
    irr = t.loc[order, 'IRR'].values; lo = t.loc[order, 'CI_lo'].values; hi = t.loc[order, 'CI_hi'].values
    ax.hlines(y, lo, hi, color=col, lw=1.6, zorder=3)
    ax.scatter(irr, y, s=34, marker=mk, color=col, edgecolor='white', linewidth=0.8, zorder=4, label=lab)
ax.axvline(1, color='#6f6e69', lw=0.9, ls=(0, (3, 3)), zorder=2)
ax.set_xscale('log'); ax.set_xlim(0.4, 6.5)
tv = [0.5, 0.75, 1, 1.5, 2, 3, 5]
ax.xaxis.set_major_locator(FixedLocator(tv)); ax.xaxis.set_major_formatter(FixedFormatter(['0.5', '0.75', '1', '1.5', '2', '3', '5'])); ax.xaxis.set_minor_locator(NullLocator())
ax.set_yticks(np.arange(len(order))); ax.set_yticklabels([labels[o] for o in order]); ax.invert_yaxis()
ax.set_xlabel('Incidence rate ratio for engagement (log scale, 95% CI)'); ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.42, -0.2), fontsize=7.5, ncol=1)
fig.tight_layout(); fig.savefig(f'{out}/fig2_naive_vs_corrected.png', facecolor='white'); plt.close(fig)

# ---------- Figure 3: corrected IRRs by window ----------
P = {w: pd.read_csv(f'{res}/P_{w}.csv', index_col=0) for w in ['pre', 'delta', 'post']}
fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=300)
spec = [('pre', AQUA, '^', 'Pre-Delta (Apr–Jun 2021)'), ('delta', BLUE, 'o', 'Delta (Jul–Sep 2021)'), ('post', ORANGE, 's', 'Post-Delta (Oct–Nov 2021)')]
for j, (w, col, mk, lab) in enumerate(spec):
    t = P[w]; y = np.arange(len(order)) + (j - 1) * 0.24
    ax.hlines(y, t.loc[order, 'CI_lo'], t.loc[order, 'CI_hi'], color=col, lw=1.6, zorder=3)
    ax.scatter(t.loc[order, 'IRR'], y, s=34, marker=mk, color=col, edgecolor='white', linewidth=0.8, zorder=4, label=lab)
ax.axvline(1, color='#6f6e69', lw=0.9, ls=(0, (3, 3)), zorder=2)
ax.set_xscale('log'); ax.set_xlim(0.5, 2.2)
tv = [0.5, 0.75, 1, 1.25, 1.5, 2]
ax.xaxis.set_major_locator(FixedLocator(tv)); ax.xaxis.set_major_formatter(FixedFormatter([str(v) for v in tv])); ax.xaxis.set_minor_locator(NullLocator())
ax.set_yticks(np.arange(len(order))); ax.set_yticklabels([labels[o] for o in order]); ax.invert_yaxis()
ax.set_xlabel('Incidence rate ratio (log scale, 95% CI) — corrected specification'); ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.42, -0.2), fontsize=7.5, ncol=3)
fig.tight_layout(); fig.savefig(f'{out}/fig3_by_window.png', facecolor='white'); plt.close(fig)
print('ok')
