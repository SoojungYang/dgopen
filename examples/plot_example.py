"""Per-residue dG_open track for the 5PTI example: ensemble mean +- member spread, with the
Best-Vendruscolo baseline. Usage: python examples/plot_example.py (after running the README
quickstart). Needs matplotlib."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
m = pd.read_csv(os.path.join(here, '5PTI_dgopen.csv'))
b = pd.read_csv(os.path.join(here, '5PTI_best.csv'))
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Lato', 'DejaVu Sans'],
                     'axes.linewidth': 1.5, 'axes.labelsize': 12, 'xtick.labelsize': 10,
                     'ytick.labelsize': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False})
fig, ax = plt.subplots(figsize=(6.5, 2.6))
ax.fill_between(m.resi, m.dg_open_kcal - m.dg_open_member_sd, m.dg_open_kcal + m.dg_open_member_sd,
                color='#54cffb', alpha=0.35, lw=0, label='member spread ($\\pm$1 SD)')
ax.plot(m.resi, m.dg_open_kcal, color='#00386b', lw=2.0, marker='o', ms=3, label='dgopen ensemble')
ax.plot(b.resi, b.dg_open_kcal, color='#f54b1a', lw=1.5, ls='--', label='Best-Vendruscolo')
ax.set_xlabel('residue number (5PTI chain A)')
ax.set_ylabel('$\\Delta G_\\mathrm{open}$ (kcal/mol)')
ax.legend(fontsize=9, loc='upper left', ncol=3, bbox_to_anchor=(0, 1.18))
fig.savefig(os.path.join(here, '5PTI_track.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(here, '5PTI_track.png'), dpi=200, bbox_inches='tight')
