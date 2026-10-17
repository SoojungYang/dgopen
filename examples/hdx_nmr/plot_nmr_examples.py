"""13-panel figure: measured HDX-NMR dG_open vs dgopen and Best-Vendruscolo, per domain.
Run after run_nmr_examples.py. Writes nmr_examples.pdf and nmr_examples.png here."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(here, 'nmr_comparison.csv'))
r = pd.read_csv(os.path.join(here, 'nmr_rho.csv')).set_index('domain')
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Lato', 'DejaVu Sans'],
                     'axes.linewidth': 1.0, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'axes.labelsize': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'xtick.major.size': 2.5,
                     'ytick.major.size': 2.5, 'legend.frameon': False})
NAVY, SKY, RED, BLACK = '#00386b', '#54cffb', '#f54b1a', '#110d1b'
GREEN = '#0f6852'                         # measured HDX-NMR
order = list(r.sort_values(['held_out', 'rho_dgopen'], ascending=[False, False]).index)
fig, axes = plt.subplots(4, 4, figsize=(6.5, 6.2), sharey=True)
for ax, dom in zip(axes.flat, order):
    t = d[d.domain == dom]; m = t.dropna(subset=['dg_open_measured_kcal'])
    ax.fill_between(t.resi, t.dgopen_kcal - t.dgopen_member_sd, t.dgopen_kcal + t.dgopen_member_sd,
                    color=SKY, alpha=0.35, lw=0)
    ax.plot(t.resi, t.dgopen_kcal, color=NAVY, lw=1.4)
    ax.plot(t.resi, t.best_kcal, color=RED, lw=1.0, ls='--')
    isb = m.lower_bound.astype(bool).to_numpy()
    pt, lb = m[~isb], m[isb]
    ax.scatter(pt.resi, pt.dg_open_measured_kcal, s=11, color=GREEN, edgecolor='white', lw=0.3, zorder=3)
    ax.scatter(lb.resi, lb.dg_open_measured_kcal, s=13, marker='^', facecolor='white', edgecolor=GREEN, lw=0.8, zorder=3)
    title = dom.replace('.1_', ' ').replace('_', ' ')
    for a, b in (('K31L E36V', 'K31L/E36V'), ('R35D G45L', 'R35D/G45L')):
        title = title.replace(a, b)
    held = bool(r.loc[dom, 'held_out'])
    ax.set_title(title, fontsize=7.2 if len(title) < 21 else 6.3, pad=2, fontweight='bold' if held else 'normal')
    if held:
        ax.text(0.97, 0.97, 'held out', transform=ax.transAxes, ha='right', va='top', fontsize=6.8,
                fontweight='bold', color=BLACK)
    ax.text(0.03, 0.97, f'$\\rho$ {r.loc[dom, "rho_dgopen"]:.2f}', color=NAVY, transform=ax.transAxes, va='top', fontsize=7.2)
    ax.text(0.03, 0.86, f'$\\rho$ {r.loc[dom, "rho_best"]:.2f}', color=RED, transform=ax.transAxes, va='top', fontsize=7.2)
    ax.set_ylim(-1, 15); ax.set_xlim(t.resi.min() - 1, t.resi.max() + 1)
for ax in axes.flat[len(order):]:
    ax.axis('off')
handles = [Line2D([], [], color=NAVY, lw=1.4, label='dgopen (12-member mean)'),
           Patch(color=SKY, alpha=0.35, label='member spread ($\\pm$1 SD)'),
           Line2D([], [], color=RED, lw=1.0, ls='--', label='Best-Vendruscolo'),
           Line2D([], [], color=GREEN, marker='o', ls='', ms=3.5, label='HDX-NMR measured'),
           Line2D([], [], color=GREEN, marker='^', ls='', ms=4, mfc='white', mew=0.8, label='measured, lower bound')]
axes.flat[len(order)].legend(handles=handles, loc='upper left', fontsize=7.8, bbox_to_anchor=(-0.05, 1.0), handlelength=1.6)
fig.supxlabel('residue number', fontsize=10, y=0.012)
fig.supylabel('$\\Delta G_\\mathrm{open}$ (kcal/mol)', fontsize=10, x=0.02)
fig.tight_layout(rect=(0.025, 0.02, 1, 1), h_pad=0.5, w_pad=0.3)
fig.savefig(os.path.join(here, 'nmr_examples.pdf'))
fig.savefig(os.path.join(here, 'nmr_examples.png'), dpi=200)
