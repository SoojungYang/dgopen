"""Training schematic for the README: what each training stage supervises.

Uses one real domain (EEHEE_rd4_0871, data in schematic_data.json):
  (a) pretraining: a synthetic target is known for every residue, so the loss pairs residue i
      with residue i (shown with the Best-Vendruscolo target);
  (b) fine-tuning: HDX-MS gives only a multiset of opening energies with no residue labels;
      predictions are sorted and matched to the sorted measurements rank by rank;
  (c) output: one assigned ΔG_open per residue, here compared with HDX-NMR, which is never used
      in training.
Writes training_schematic.pdf / .png next to this file."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

here = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(here, 'schematic_data.json')))
resi = np.array(D['resi']); ex = np.array(D['exchangeable'])
f = lambda k: np.array([np.nan if v is None else v for v in D[k]], float)
pred, best, nmr = f('dgopen'), f('best'), f('nmr')
ms = np.array(D['ms_multiset'])

NAVY, BLUE, SKY, ORANGE, RED, BLACK, GREY, LGREY = ('#00386b', '#1b9ad0', '#54cffb', '#fbac41',
                                                   '#f54b1a', '#110d1b', '#9dadc4', '#e6ebee')
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Lato', 'DejaVu Sans'],
                     'font.size': 8, 'axes.linewidth': 1.0, 'axes.labelsize': 8.5,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
                     'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'legend.frameon': False,
                     'mathtext.fontset': 'dejavusans'})
fig = plt.figure(figsize=(6.5, 5.5))
gs = fig.add_gridspec(3, 3, height_ratios=[0.72, 1.55, 1.2], width_ratios=[1.25, 0.55, 1.25],
                      hspace=0.55, wspace=0.42, left=0.07, right=0.985, top=0.985, bottom=0.075)

# ---------------------------------------------------------------- top: pipeline
ax0 = fig.add_subplot(gs[0, :]); ax0.set_xlim(0, 100); ax0.set_ylim(3, 27.6); ax0.axis('off')
def box(x, y, w, h, title, body, fc, ec, tc=BLACK):
    ax0.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.5', fc=fc, ec=ec, lw=1.2))
    ax0.text(x + w / 2, y + h - 2.2, title, ha='center', va='top', fontsize=8.3, fontweight='bold', color=tc)
    ax0.text(x + w / 2, y + h - 7.6, body, ha='center', va='top', fontsize=7.0, color=BLACK, linespacing=1.3)
def arrow(x0, x1, y):
    ax0.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>', mutation_scale=10, lw=1.3, color=BLACK))
box(0.8, 4, 19.5, 23, 'Input', 'structure + sequence\n39 per-residue features\nESM-2 650M embedding\nCA–CA distance bias', 'white', GREY)
box(25.3, 4, 23.5, 23, '1  Pretraining', 'per-residue synthetic target\n(MSE, residue i ↔ residue i)\nBEST · MD-mean · MD-mean-var\n(4th variant: random init)', '#fff3e0', ORANGE)
box(53.8, 4, 23.5, 23, '2  Fine-tuning', 'mHDX-MS, 3,034 domains\nunassigned multiset per domain\n(sorted L1, rank j ↔ rank j)\n3 seeds per variant', '#e3f4fd', BLUE)
box(82.3, 4, 16.9, 23, '3  Output', '12-member mean\none $\\Delta G_\\mathrm{open}$\nper residue\n(kcal/mol)', '#e8edf5', NAVY)
for a, b in ((20.7, 24.9), (49.2, 53.4), (77.7, 81.9)):
    arrow(a, b, 15.5)

def letter(ax, s, dx=-0.13, dy=1.07):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=11, fontweight='bold', va='bottom')

# ---------------------------------------------------------------- (a) pretraining
axa = fig.add_subplot(gs[1, 0])
axa.bar(resi[ex], best[ex], width=0.75, color=ORANGE, alpha=0.75, lw=0)
for r in (12, 24, 36):                                    # a few residue-to-residue pairings
    i = np.where(resi == r)[0][0]
    axa.annotate('', xy=(r, best[i] + 0.15), xytext=(r, best[i] + 1.6),
                 arrowprops=dict(arrowstyle='-|>', color=NAVY, lw=1.0, mutation_scale=7))
    axa.text(r, best[i] + 1.75, f'$\\hat{{y}}_{{{r}}}$', ha='center', va='bottom', fontsize=7, color=NAVY)
axa.set_xlabel('residue $i$'); axa.set_ylabel('target $\\Delta G_\\mathrm{open}$ (kcal/mol)')
axa.set_ylim(0, 18.5); axa.set_yticks([0, 5, 10, 15]); axa.set_xlim(0, resi.max() + 1)
axa.set_title('pretraining: every residue has a target', fontsize=8.3, loc='left', pad=4)
axa.text(0.02, 0.98, 'bars: synthetic target $t_i$ (BEST shown)', transform=axa.transAxes, ha='left', va='top',
         fontsize=7.2, color='#b86e00')
axa.text(0.02, 0.89, 'each prediction $\\hat{y}_i$ is compared with $t_i$\n$\\mathcal{L}=\\sum_i w\\,(\\hat{y}_i-t_i)^2$',
         transform=axa.transAxes, ha='left', va='top', fontsize=7.2, color=NAVY, linespacing=1.5)
letter(axa, 'a', dx=-0.2)

# ---------------------------------------------------------------- (b) the multiset
axb = fig.add_subplot(gs[1, 1])
rng = np.random.default_rng(3)
jit = rng.uniform(-0.32, 0.32, len(ms))
axb.scatter(jit, ms, s=11, facecolor=np.where(ms <= 1e-6, 'white', RED), edgecolor=RED, lw=0.8, zorder=3)
axb.set_xlim(-0.75, 0.75); axb.set_ylim(-0.6, 6.4); axb.set_xticks([])
axb.spines['bottom'].set_visible(False)
axb.set_ylabel('measured $\\Delta G_\\mathrm{open}$ (kcal/mol)')
axb.set_title('HDX-MS gives a\nbag of values', fontsize=8.3, pad=4)
axb.text(0, -0.05, f'{len(ms)} values,\nno residue labels', transform=axb.get_xaxis_transform(), ha='center', va='top', fontsize=7)
letter(axb, 'b', dx=-0.62)

# ---------------------------------------------------------------- (c) rank matching
axc = fig.add_subplot(gs[1, 2])
ps = np.sort(pred[ex])[::-1]; mm = np.sort(ms)[::-1]; k = min(len(ps), len(mm)); j = np.arange(1, k + 1)
for jj, a_, b_ in zip(j, ps[:k], mm[:k]):
    axc.plot([jj, jj], [a_, b_], color=GREY, lw=0.7, zorder=1)
floor = mm[:k] <= 1e-6
axc.scatter(j[~floor], mm[:k][~floor], s=11, color=RED, zorder=3, lw=0, label='measured, sorted $m_{[j]}$')
axc.scatter(j[floor], mm[:k][floor], s=11, facecolor='white', edgecolor=RED, lw=0.8, zorder=3, label='fast-exchange floor (hinge)')
axc.scatter(j, ps[:k], s=9, color=NAVY, zorder=4, lw=0, label='predicted, sorted $\\hat{y}_{[j]}$')
axc.axhline(1.5, color=RED, lw=0.7, ls=':', zorder=0)
axc.text(k + 0.5, 1.6, '1.5', fontsize=6.5, color=RED, va='bottom', ha='right')
axc.set_xlabel('rank $j$ (highest first)'); axc.set_ylim(-0.6, 6.4); axc.set_xlim(0, k + 1)
axc.set_ylabel('$\\Delta G_\\mathrm{open}$ (kcal/mol)')
axc.legend(fontsize=6.8, loc='upper right', handlelength=1.0, borderaxespad=0.1)
axc.set_title('fine-tuning: match by rank, not residue', fontsize=8.3, loc='left', pad=4)
axc.text(0.03, 0.05, '$\\mathcal{L}=\\mathrm{mean}_j\\,|\\hat{y}_{[j]}-m_{[j]}|$\n(after training)', transform=axc.transAxes, fontsize=7.4, color=NAVY, linespacing=1.4)
letter(axc, 'c', dx=-0.2)

# ---------------------------------------------------------------- (d) output vs HDX-NMR
axd = fig.add_subplot(gs[2, :])
axd.plot(resi, np.where(ex, pred, np.nan), color=NAVY, lw=1.6, marker='o', ms=2.8, label='dgopen: assigned $\\Delta G_\\mathrm{open}$ per residue', zorder=3)
ok = ~np.isnan(nmr)
axd.scatter(resi[ok], nmr[ok], s=16, facecolor='white', edgecolor=BLACK, lw=0.9, zorder=4,
            label='HDX-NMR measured (per residue; never used in training)')
axd.set_xlabel('residue number'); axd.set_ylabel('$\\Delta G_\\mathrm{open}$ (kcal/mol)')
axd.set_xlim(0, resi.max() + 1); axd.set_ylim(0, 7.5)
axd.legend(fontsize=7.2, loc='upper left', ncol=2, handlelength=1.4, borderaxespad=0.1)
axd.set_title('output: the residue assignment that HDX-MS lacks  (EEHEE_rd4_0871, after training)', fontsize=8.3, loc='left', pad=4)
letter(axd, 'd', dx=-0.075, dy=1.04)

fig.savefig(os.path.join(here, 'training_schematic.pdf'))
fig.savefig(os.path.join(here, 'training_schematic.png'), dpi=220)
