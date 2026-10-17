"""Distance-biased graph transformer (architecture of the released checkpoints).

Same parameter names and arithmetic as the development trainer (DistGraphTransformer in
src/track2_train_torch.py, 'large' preset), with the sizes passed explicitly instead of
read from module globals, and no dropout (inference only).

Per layer l, head h:   S_h = Q_h K_h^T / sqrt(d_h) + B_h,   B_h[i, j] = sum_k PF[i, j, k] wb[l][h, k]
so the only geometric input is the pair tensor PF (CA-CA distance RBFs + sequence
separation); there is no positional encoding.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ArchConfig:
    n_in: int = 1319          # 39 standardized node features + 1280 ESM-2 650M dims
    d_head: int = 32
    n_head: int = 8
    n_layer: int = 6
    n_pair: int = 22

    @property
    def hidden(self):
        return self.d_head * self.n_head

    @property
    def ffn(self):
        return 2 * self.hidden


class DistGraphTransformer(nn.Module):
    def __init__(self, cfg: ArchConfig = ArchConfig()):
        super().__init__()
        self.cfg = cfg
        H, L = cfg.hidden, cfg.n_layer
        self.inp = nn.Linear(cfg.n_in, H)
        self.q = nn.ModuleList(nn.Linear(H, H, bias=False) for _ in range(L))
        self.k = nn.ModuleList(nn.Linear(H, H, bias=False) for _ in range(L))
        self.v = nn.ModuleList(nn.Linear(H, H, bias=False) for _ in range(L))
        self.o = nn.ModuleList(nn.Linear(H, H, bias=False) for _ in range(L))
        self.wb = nn.ParameterList(nn.Parameter(torch.zeros(cfg.n_head, cfg.n_pair)) for _ in range(L))
        self.f1 = nn.ModuleList(nn.Linear(H, cfg.ffn) for _ in range(L))
        self.f2 = nn.ModuleList(nn.Linear(cfg.ffn, H) for _ in range(L))
        self.out = nn.Linear(H, 1)

    def forward(self, X, PF):
        """X (L, n_in), PF (L, L, n_pair) -> per-residue dG_open (L,), kcal/mol."""
        c = self.cfg
        L = X.shape[0]
        H = self.inp(X)
        for l in range(c.n_layer):
            Q = self.q[l](H).view(L, c.n_head, c.d_head)
            K = self.k[l](H).view(L, c.n_head, c.d_head)
            V = self.v[l](H).view(L, c.n_head, c.d_head)
            B = torch.einsum('ijk,hk->hij', PF, self.wb[l])
            S = torch.einsum('ihd,jhd->hij', Q, K) / (c.d_head ** 0.5) + B
            A = torch.softmax(S, dim=-1)
            ctx = torch.einsum('hij,jhd->ihd', A, V).reshape(L, c.hidden)
            H = H + self.o[l](ctx)
            H = H + self.f2[l](torch.relu(self.f1[l](H)))
        return self.out(H).squeeze(-1)
