"""
The module offers a flexible framework for creating diverse, realistic tabular datasets
with controlled properties, which can be used for training and evaluating in-context
learning models. Key features include:

- Controlled feature relationships and causal structures via multiple generation methods
- Customizable feature distributions with mixed continuous and categorical variables
- Flexible train/test splits optimized for in-context learning evaluation
- Batch generation capabilities with hierarchical parameter sharing
- Memory-efficient handling of variable-length datasets

The main class is PriorDataset, which provides an iterable interface for generating
an infinite stream of synthetic datasets with diverse characteristics.
"""

from __future__ import annotations

import os
import sys
import math
import warnings
from itertools import combinations
from typing import Dict, Tuple, Union, Optional, Any

import numpy as np
from scipy.stats import loguniform, t as student_t
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nested import nested_tensor
from torch.utils.data import IterableDataset

from .mlp_scm import MLPSCM
from .tree_scm import TreeSCM

from .hp_sampling import HpSamplerList
from .reg2cls import Reg2Cls
from .prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP


warnings.filterwarnings(
    "ignore", message=".*The PyTorch API of nested tensors is in prototype stage.*", category=UserWarning
)


def _partial_corr_pvalue(x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray]) -> float:
    """Return two-sided p-value for correlation between x and y conditioned on z."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(x.shape[0])
    if n < 4:
        return 1.0

    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom <= 1e-12:
            return 0.0
        corr = float(np.dot(a, b) / denom)
        return max(min(corr, 0.999999), -0.999999)

    if z is None or z.size == 0:
        x0 = x - x.mean()
        y0 = y - y.mean()
        r = _safe_corr(x0, y0)
        df = n - 2
    else:
        z = np.asarray(z, dtype=np.float64)
        if z.ndim == 1:
            z = z[:, None]
        if z.shape[0] != n:
            return 1.0
        zc = z - z.mean(axis=0, keepdims=True)
        x0 = x - x.mean()
        y0 = y - y.mean()
        try:
            bx, *_ = np.linalg.lstsq(zc, x0, rcond=None)
            by, *_ = np.linalg.lstsq(zc, y0, rcond=None)
        except np.linalg.LinAlgError:
            return 1.0
        rx = x0 - zc @ bx
        ry = y0 - zc @ by
        r = _safe_corr(rx, ry)
        df = n - zc.shape[1] - 2

    if df <= 0:
        return 1.0
    t_val = abs(r) * math.sqrt(df / max(1e-12, 1.0 - r * r))
    p = float(2.0 * student_t.sf(t_val, df))
    if not math.isfinite(p):
        return 1.0
    return min(max(p, 0.0), 1.0)


def _bh_selected_indices(pvals: list[float], alpha: float) -> list[int]:
    """Benjamini-Hochberg selected indices under FDR alpha."""
    if not pvals:
        return []
    p = np.asarray(pvals, dtype=np.float64)
    m = p.shape[0]
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = np.where(ranked <= thresh)[0]
    if passed.size == 0:
        return []
    kmax = int(passed[-1])
    return order[: kmax + 1].tolist()


def _iter_condition_subsets(indices: list[int], max_k: int):
    yield ()
    if max_k <= 0 or not indices:
        return
    upper = min(max_k, len(indices))
    for k in range(1, upper + 1):
        for subset in combinations(indices, k):
            yield subset


def _iamb_fdr_mb(X: np.ndarray, y: np.ndarray, alpha: float, max_k: int) -> set[int]:
    """Approximate IAMB-FDR with partial-correlation CI tests."""
    d = int(X.shape[1])
    mb: list[int] = []
    remain = list(range(d))

    while remain:
        cond_idx = mb[-max_k:] if (max_k > 0 and len(mb) > max_k) else mb
        z = X[:, cond_idx] if cond_idx else None
        pvals = [_partial_corr_pvalue(X[:, j], y, z) for j in remain]
        selected = _bh_selected_indices(pvals, alpha)
        if not selected:
            break
        best_local = min(selected, key=lambda idx: pvals[idx])
        best_feat = remain[best_local]
        mb.append(best_feat)
        remain.remove(best_feat)

    # Backward elimination
    changed = True
    while changed and mb:
        changed = False
        for feat in list(mb):
            cond = [j for j in mb if j != feat]
            if max_k > 0 and len(cond) > max_k:
                cond = cond[-max_k:]
            z = X[:, cond] if cond else None
            if _partial_corr_pvalue(X[:, feat], y, z) > alpha:
                mb.remove(feat)
                changed = True
    return set(mb)


def _mmpc_mb(X: np.ndarray, y: np.ndarray, alpha: float, max_k: int) -> set[int]:
    """Approximate MMPC-style selection with bounded conditioning sets."""
    d = int(X.shape[1])
    cpc: list[int] = []
    remain = list(range(d))

    while remain:
        best_feat = None
        best_worst_p = 1.0
        for feat in remain:
            worst_p = 0.0
            for subset in _iter_condition_subsets(cpc, max_k):
                z = X[:, subset] if subset else None
                p = _partial_corr_pvalue(X[:, feat], y, z)
                if p > worst_p:
                    worst_p = p
                if worst_p >= best_worst_p:
                    break
            if worst_p < best_worst_p:
                best_worst_p = worst_p
                best_feat = feat

        if best_feat is None or best_worst_p > alpha:
            break
        cpc.append(best_feat)
        remain.remove(best_feat)

    # Backward phase
    for feat in list(cpc):
        others = [j for j in cpc if j != feat]
        independent = False
        for subset in _iter_condition_subsets(others, max_k):
            z = X[:, subset] if subset else None
            if _partial_corr_pvalue(X[:, feat], y, z) > alpha:
                independent = True
                break
        if independent:
            cpc.remove(feat)
    return set(cpc)


def _is_discrete_target(y: np.ndarray) -> bool:
    """Heuristic check whether y should be treated as a categorical target."""
    y = np.asarray(y).reshape(-1)
    if y.size == 0:
        return False
    y_round = np.rint(y)
    int_like_ratio = float(np.mean(np.abs(y - y_round) < 1e-6))
    if int_like_ratio < 0.99:
        return False
    n_unique = int(np.unique(y_round).size)
    unique_cap = max(2, min(32, int(2 * math.sqrt(y.shape[0]))))
    return n_unique <= unique_cap


def _corr_fdr_mb(X: np.ndarray, y: np.ndarray, alpha: float) -> set[int]:
    """Marginal correlation + BH-FDR."""
    d = int(X.shape[1])
    pvals = [_partial_corr_pvalue(X[:, j], y, None) for j in range(d)]
    selected = _bh_selected_indices(pvals, alpha)
    return set(selected)


def _mi_fdr_mb(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    random_state: int = 0,
    num_null_permutations: int = 6,
) -> set[int]:
    """Mutual-information ranking with permutation null + BH-FDR."""
    try:
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    except Exception:
        return set()

    d = int(X.shape[1])
    if d == 0:
        return set()

    discrete = _is_discrete_target(y)
    if discrete:
        y_fit = np.rint(y).astype(np.int64)
        mi_fn = mutual_info_classif
    else:
        y_fit = y.astype(np.float64)
        mi_fn = mutual_info_regression

    try:
        scores = np.asarray(mi_fn(X, y_fit, random_state=random_state), dtype=np.float64)
    except Exception:
        return set()
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    if scores.size == 0:
        return set()

    rng = np.random.default_rng(random_state)
    null_chunks: list[np.ndarray] = []
    for ridx in range(max(1, int(num_null_permutations))):
        y_perm = rng.permutation(y_fit)
        try:
            null_s = np.asarray(mi_fn(X, y_perm, random_state=random_state + ridx + 1), dtype=np.float64)
        except Exception:
            continue
        null_s = np.nan_to_num(null_s, nan=0.0, posinf=0.0, neginf=0.0)
        null_chunks.append(null_s)

    if not null_chunks:
        # Fallback to top-magnitude MI if permutation null failed
        top = int(np.argmax(scores))
        return {top} if scores[top] > 0 else set()

    null = np.concatenate(null_chunks, axis=0)
    pvals = [float((1 + np.count_nonzero(null >= s)) / (1 + null.size)) for s in scores]
    selected = _bh_selected_indices(pvals, alpha)
    if selected:
        return set(selected)

    # Keep at least one feature if all p-values are conservative.
    top = int(np.argmax(scores))
    return {top} if scores[top] > 0 else set()


def _lasso_mb(X: np.ndarray, y: np.ndarray, alpha: float, random_state: int = 0) -> set[int]:
    """Sparse linear probe (L1 logistic / Lasso) as MB proxy."""
    try:
        from sklearn.linear_model import Lasso, LogisticRegression
    except Exception:
        return set()

    d = int(X.shape[1])
    if d == 0:
        return set()

    discrete = _is_discrete_target(y)
    try:
        if discrete:
            y_fit = np.rint(y).astype(np.int64)
            c_val = float(np.clip(1.0 / max(alpha, 1e-3), 0.2, 10.0))
            model = LogisticRegression(
                penalty="l1",
                solver="saga",
                C=c_val,
                max_iter=300,
                n_jobs=1,
                class_weight="balanced",
                random_state=random_state,
            )
            model.fit(X, y_fit)
            coef = np.asarray(model.coef_, dtype=np.float64)
            importance = np.abs(coef).mean(axis=0)
        else:
            # Use a fixed regularization level to keep inference stable/fast per dataset.
            l1_alpha = float(np.clip(alpha * 2.0, 0.005, 0.2))
            model = Lasso(alpha=l1_alpha, max_iter=2000, random_state=random_state)
            model.fit(X, y.astype(np.float64))
            importance = np.abs(np.asarray(model.coef_, dtype=np.float64))
    except Exception:
        return set()

    importance = np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
    selected = np.where(importance > 1e-8)[0].tolist()
    if selected:
        return set(int(i) for i in selected)
    top = int(np.argmax(importance))
    return {top} if importance[top] > 0 else set()


def _mlp_saliency_mb(
    X: Tensor,
    y: Tensor,
    alpha: float,
    random_state: int = 0,
    train_steps: int = 180,
    batch_size: int = 1024,
    grad_sparsity_lambda: float = 0.0,
) -> set[int]:
    """Tiny MLP probe + input-gradient saliency (GPU-capable deep baseline)."""
    n, d = int(X.shape[0]), int(X.shape[1])
    if n < 16 or d == 0:
        return set()

    device = X.device
    torch.manual_seed(int(random_state))
    hidden = min(96, max(16, 2 * d))
    model = nn.Sequential(
        nn.Linear(d, hidden),
        nn.SiLU(),
        nn.Linear(hidden, 1),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)

    X_t = X.detach().float()
    y_t = y.detach().float().reshape(-1, 1)
    bs = max(64, min(int(batch_size), n))

    try:
        for _ in range(max(1, int(train_steps))):
            opt.zero_grad(set_to_none=True)
            if n <= bs:
                xb, yb = X_t, y_t
            else:
                idx = torch.randint(0, n, (bs,), device=device)
                xb, yb = X_t[idx], y_t[idx]

            if float(grad_sparsity_lambda) > 0.0:
                xb_req = xb.detach().clone().requires_grad_(True)
                pred = model(xb_req)
                fit_loss = F.mse_loss(pred, yb)
                grad_x = torch.autograd.grad(pred.sum(), xb_req, create_graph=True)[0]
                sparse_loss = grad_x.abs().mean()
                loss = fit_loss + float(grad_sparsity_lambda) * sparse_loss
            else:
                pred = model(xb)
                loss = F.mse_loss(pred, yb)

            if not torch.isfinite(loss):
                return set()
            loss.backward()
            opt.step()

        sal_n = min(n, 4096)
        if sal_n < n:
            idx = torch.randint(0, n, (sal_n,), device=device)
            X_req = X_t[idx].detach().clone().requires_grad_(True)
            y_req = y_t[idx]
        else:
            X_req = X_t.detach().clone().requires_grad_(True)
            y_req = y_t
        pred = model(X_req)
        loss = F.mse_loss(pred, y_req)
        loss.backward()
        saliency = (X_req.grad.abs().mean(dim=0) * X_req.std(dim=0).clamp_min(1e-6)).detach().cpu().numpy()
    except Exception:
        return set()

    saliency = np.nan_to_num(saliency, nan=0.0, posinf=0.0, neginf=0.0)
    if saliency.size == 0 or float(np.max(saliency)) <= 0:
        return set()

    # Keep a small adaptive top-k set; alpha controls aggressiveness.
    ratio = float(np.clip(0.08 + 4.0 * alpha, 0.08, 0.35))
    k = max(1, int(round(ratio * d)))
    top_idx = np.argsort(-saliency)[:k]
    return set(int(i) for i in top_idx.tolist())


def _l0_resnet_probe_mb(
    X: Tensor,
    y: Tensor,
    alpha: float,
    random_state: int = 0,
    train_steps: int = 300,
    batch_size: int = 1024,
    l0_lambda: float = 1e-2,
    hidden_dim: int = 192,
    num_blocks: int = 3,
) -> set[int]:
    """Hard-concrete L0 gates + residual MLP probe for sparse MB discovery."""
    n, d = int(X.shape[0]), int(X.shape[1])
    if n < 32 or d == 0:
        return set()

    device = X.device
    torch.manual_seed(int(random_state))
    np.random.seed(int(random_state))

    class _ResBlock(nn.Module):
        def __init__(self, h: int):
            super().__init__()
            self.ln = nn.LayerNorm(h)
            self.fc1 = nn.Linear(h, h * 2)
            self.fc2 = nn.Linear(h * 2, h)

        def forward(self, x: Tensor) -> Tensor:
            h = self.ln(x)
            h = F.gelu(self.fc1(h))
            h = self.fc2(h)
            return x + h

    class _L0ResNetProbe(nn.Module):
        def __init__(self, in_dim: int, h: int, blocks: int, lam: float):
            super().__init__()
            self.in_dim = in_dim
            self.h = h
            self.gamma = -0.1
            self.zeta = 1.1
            self.temperature = 0.66
            self.l0_lambda = float(lam)

            self.log_alpha = nn.Parameter(torch.full((in_dim,), -0.3))
            self.in_proj = nn.Linear(in_dim, h)
            self.inter_u = nn.Linear(in_dim, h, bias=False)
            self.inter_v = nn.Linear(in_dim, h, bias=False)
            self.blocks = nn.ModuleList([_ResBlock(h) for _ in range(max(1, int(blocks)))])
            self.head = nn.Linear(h, 1)

        def _sample_z(self, batch_size_: int) -> Tensor:
            if self.training:
                u = torch.rand((batch_size_, self.in_dim), device=self.log_alpha.device).clamp_(1e-6, 1.0 - 1e-6)
                s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + self.log_alpha.unsqueeze(0)) / self.temperature)
            else:
                s = torch.sigmoid(self.log_alpha).unsqueeze(0).expand(batch_size_, -1)
            s_bar = s * (self.zeta - self.gamma) + self.gamma
            return s_bar.clamp(0.0, 1.0)

        def expected_l0(self) -> Tensor:
            logit = self.log_alpha - self.temperature * math.log(-self.gamma / self.zeta)
            prob_nonzero = torch.sigmoid(logit)
            return prob_nonzero.mean()

        def deterministic_gate(self) -> Tensor:
            s = torch.sigmoid(self.log_alpha)
            return (s * (self.zeta - self.gamma) + self.gamma).clamp(0.0, 1.0)

        def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
            z = self._sample_z(x.shape[0])
            xg = x * z
            h = self.in_proj(xg)
            h = h + torch.tanh(self.inter_u(xg)) * torch.tanh(self.inter_v(xg))
            for blk in self.blocks:
                h = blk(h)
            out = self.head(F.gelu(h)).squeeze(-1)
            return out, z

    model = _L0ResNetProbe(
        in_dim=d,
        h=max(64, int(hidden_dim)),
        blocks=max(1, int(num_blocks)),
        lam=float(l0_lambda),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-4)

    X_t = X.detach().float()
    y_t = y.detach().float().reshape(-1)
    bs = max(64, min(int(batch_size), n))

    try:
        model.train()
        for _ in range(max(1, int(train_steps))):
            opt.zero_grad(set_to_none=True)
            if n <= bs:
                xb, yb = X_t, y_t
            else:
                idx = torch.randint(0, n, (bs,), device=device)
                xb, yb = X_t[idx], y_t[idx]

            pred, _ = model(xb)
            fit_loss = F.mse_loss(pred, yb)
            sparse_loss = model.expected_l0()
            loss = fit_loss + model.l0_lambda * sparse_loss
            if not torch.isfinite(loss):
                return set()
            loss.backward()
            opt.step()

        model.eval()
        sal_n = min(n, 4096)
        if sal_n < n:
            idx = torch.randint(0, n, (sal_n,), device=device)
            xg = X_t[idx].detach().clone().requires_grad_(True)
            yg = y_t[idx]
        else:
            xg = X_t.detach().clone().requires_grad_(True)
            yg = y_t

        # Use deterministic gates for stable post-hoc saliency.
        gate_det = model.deterministic_gate().detach()
        pred, _ = model(xg)
        loss = F.mse_loss(pred, yg)
        loss.backward()
        saliency = (xg.grad.abs().mean(dim=0) * xg.std(dim=0).clamp_min(1e-6)).detach()

        gate_prob = torch.sigmoid(
            model.log_alpha.detach() - model.temperature * math.log(-model.gamma / model.zeta)
        ).clamp(0.0, 1.0)
        scores = (0.6 * saliency + 0.4 * saliency.mean() * gate_prob) * gate_det.clamp_min(1e-4)
        scores = scores.cpu().numpy()
    except Exception:
        return set()

    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    if scores.size == 0 or float(np.max(scores)) <= 0:
        return set()

    ratio = float(np.clip(0.06 + 4.0 * alpha, 0.06, 0.35))
    k = max(1, int(round(ratio * d)))
    top_idx = np.argsort(-scores)[:k]
    return set(int(i) for i in top_idx.tolist())


def _tabtransformer_probe_mb(
    X: Tensor,
    y: Tensor,
    alpha: float,
    random_state: int = 0,
    train_steps: int = 180,
    batch_size: int = 1024,
) -> set[int]:
    """Lightweight Transformer probe + gradient saliency (GPU-capable)."""
    n, d = int(X.shape[0]), int(X.shape[1])
    if n < 32 or d == 0:
        return set()

    device = X.device
    torch.manual_seed(int(random_state))
    np.random.seed(int(random_state))

    X_t = X.detach().float()
    y_t = y.detach().float().reshape(-1)

    y_round = torch.round(y_t)
    int_like_ratio = float((y_t - y_round).abs().lt(1e-6).float().mean().item())
    n_unique = int(torch.unique(y_round).numel())
    unique_cap = max(2, min(32, int(2 * math.sqrt(max(1, n)))))
    is_discrete = int_like_ratio >= 0.99 and n_unique <= unique_cap

    if is_discrete:
        uniq = torch.unique(y_round).sort()[0]
        y_cls = torch.bucketize(y_round, uniq, right=False).long()
        num_classes = int(uniq.numel())
        if num_classes < 2:
            return set()
    else:
        y_reg = y_t

    embed_dim = 64
    nhead = 4
    num_layers = 2
    ff_dim = 128

    class _FeatureTransformer(nn.Module):
        def __init__(self, num_features: int, out_dim: int):
            super().__init__()
            self.value_proj = nn.Linear(1, embed_dim)
            self.type_emb = nn.Parameter(torch.randn(num_features, embed_dim) * 0.02)
            self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=0.0,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.head = nn.Linear(embed_dim, out_dim)

        def forward(self, x: Tensor) -> Tensor:
            bsz = x.shape[0]
            toks = self.value_proj(x.unsqueeze(-1))
            toks = toks + self.type_emb.unsqueeze(0)
            cls = self.cls.expand(bsz, -1, -1)
            h = self.encoder(torch.cat([cls, toks], dim=1))
            return self.head(h[:, 0])

    out_dim = num_classes if is_discrete else 1
    model = _FeatureTransformer(d, out_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=5e-4)
    bs = max(64, min(int(batch_size), n))

    try:
        for _ in range(max(1, int(train_steps))):
            opt.zero_grad(set_to_none=True)
            if n <= bs:
                xb = X_t
                if is_discrete:
                    yb = y_cls
                else:
                    yb = y_reg
            else:
                idx = torch.randint(0, n, (bs,), device=device)
                xb = X_t[idx]
                if is_discrete:
                    yb = y_cls[idx]
                else:
                    yb = y_reg[idx]

            out = model(xb)
            if is_discrete:
                loss = F.cross_entropy(out, yb)
            else:
                loss = F.mse_loss(out.squeeze(-1), yb)
            if not torch.isfinite(loss):
                return set()
            loss.backward()
            opt.step()

        sal_n = min(n, 4096)
        if sal_n < n:
            idx = torch.randint(0, n, (sal_n,), device=device)
            xg = X_t[idx].detach().clone().requires_grad_(True)
            if is_discrete:
                yg = y_cls[idx]
            else:
                yg = y_reg[idx]
        else:
            xg = X_t.detach().clone().requires_grad_(True)
            if is_discrete:
                yg = y_cls
            else:
                yg = y_reg
        out = model(xg)
        if is_discrete:
            loss = F.cross_entropy(out, yg)
        else:
            loss = F.mse_loss(out.squeeze(-1), yg)
        loss.backward()
        saliency = (xg.grad.abs().mean(dim=0) * xg.std(dim=0).clamp_min(1e-6)).detach().cpu().numpy()
    except Exception:
        return set()

    saliency = np.nan_to_num(saliency, nan=0.0, posinf=0.0, neginf=0.0)
    if saliency.size == 0 or float(np.max(saliency)) <= 0:
        return set()

    ratio = float(np.clip(0.08 + 4.0 * alpha, 0.08, 0.35))
    k = max(1, int(round(ratio * d)))
    top_idx = np.argsort(-saliency)[:k]
    return set(int(i) for i in top_idx.tolist())


def infer_mb_pseudo_labels(
    X: Tensor,
    y: Tensor,
    method: str,
    ci_alpha: float,
    max_condition_set: int,
    nn_train_steps: int = 180,
    nn_batch_size: int = 1024,
    mlp_grad_sparse_lambda: float = 0.0,
    nn_l0_lambda: float = 1e-2,
    nn_hidden_dim: int = 192,
    nn_num_blocks: int = 3,
) -> Tensor:
    """Infer binary MB pseudo labels for each feature in X."""
    method = str(method).lower()
    if method == "none":
        return torch.zeros(X.shape[1], dtype=torch.long, device=X.device)

    x_t = X.detach().float()
    y_t = y.detach().float().reshape(-1)
    n, d = int(x_t.shape[0]), int(x_t.shape[1])
    pseudo = torch.zeros(d, dtype=torch.long, device=X.device)
    if n < 8 or d == 0:
        return pseudo

    y_std = y_t.std(unbiased=False)
    if (not torch.isfinite(y_std)) or float(y_std.item()) <= 1e-10:
        return pseudo
    y_t = (y_t - y_t.mean()) / (y_std + 1e-12)

    x_std = x_t.std(dim=0, unbiased=False)
    valid_mask = torch.isfinite(x_std) & (x_std > 1e-10)
    valid_idx = torch.where(valid_mask)[0]
    if valid_idx.numel() == 0:
        return pseudo

    x_valid_t = x_t[:, valid_idx]
    x_valid_t = (x_valid_t - x_valid_t.mean(dim=0, keepdim=True)) / (
        x_valid_t.std(dim=0, unbiased=False, keepdim=True) + 1e-12
    )

    try:
        if method == "iamb_fdr":
            x_valid = x_valid_t.detach().cpu().numpy()
            y_np = y_t.detach().cpu().numpy().reshape(-1)
            selected = _iamb_fdr_mb(x_valid, y_np, alpha=float(ci_alpha), max_k=int(max_condition_set))
        elif method == "mmpc":
            x_valid = x_valid_t.detach().cpu().numpy()
            y_np = y_t.detach().cpu().numpy().reshape(-1)
            selected = _mmpc_mb(x_valid, y_np, alpha=float(ci_alpha), max_k=int(max_condition_set))
        elif method == "corr_fdr":
            x_valid = x_valid_t.detach().cpu().numpy()
            y_np = y_t.detach().cpu().numpy().reshape(-1)
            selected = _corr_fdr_mb(x_valid, y_np, alpha=float(ci_alpha))
        elif method == "mi_fdr":
            x_valid = x_valid_t.detach().cpu().numpy()
            y_np = y_t.detach().cpu().numpy().reshape(-1)
            selected = _mi_fdr_mb(x_valid, y_np, alpha=float(ci_alpha), random_state=0)
        elif method == "lasso":
            x_valid = x_valid_t.detach().cpu().numpy()
            y_np = y_t.detach().cpu().numpy().reshape(-1)
            selected = _lasso_mb(x_valid, y_np, alpha=float(ci_alpha), random_state=0)
        elif method == "mlp_saliency":
            selected = _mlp_saliency_mb(
                x_valid_t,
                y_t,
                alpha=float(ci_alpha),
                random_state=0,
                train_steps=int(nn_train_steps),
                batch_size=int(nn_batch_size),
                grad_sparsity_lambda=float(mlp_grad_sparse_lambda),
            )
        elif method == "l0_resnet_probe":
            selected = _l0_resnet_probe_mb(
                x_valid_t,
                y_t,
                alpha=float(ci_alpha),
                random_state=0,
                train_steps=int(nn_train_steps),
                batch_size=int(nn_batch_size),
                l0_lambda=float(nn_l0_lambda),
                hidden_dim=int(nn_hidden_dim),
                num_blocks=int(nn_num_blocks),
            )
        elif method == "tabtransformer_probe":
            selected = _tabtransformer_probe_mb(
                x_valid_t,
                y_t,
                alpha=float(ci_alpha),
                random_state=0,
                train_steps=int(nn_train_steps),
                batch_size=int(nn_batch_size),
            )
        else:
            raise ValueError(f"Unknown pseudo label method: {method}")
    except Exception:
        selected = set()

    for idx in selected:
        if 0 <= idx < int(valid_idx.numel()):
            pseudo[int(valid_idx[idx].item())] = 1
    return pseudo


class Prior:
    """
    Abstract base class for dataset prior generators.

    Defines the interface and common functionality for different types of
    synthetic dataset generators.

    Parameters
    ----------
    batch_size : int, default=256
        Total number of datasets to generate per batch

    min_features : int, default=2
        Minimum number of features per dataset

    max_features : int, default=100
        Maximum number of features per dataset

    max_classes : int, default=10
        Maximum number of target classes

    min_seq_len : int, default=None
        Minimum samples per dataset. If None, uses max_seq_len

    max_seq_len : int, default=1024
        Maximum samples per dataset

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution

    min_train_size : int|float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int|float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    replay_small : bool, default=False
        If True, occasionally sample smaller sequence lengths with
        specific distributions to ensure model robustness on smaller datasets
    """

    def __init__(
        self,
        batch_size: int = 256,
        min_features: int = 2,
        max_features: int = 100,
        max_classes: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        min_train_size: Union[int, float] = 0.1,
        max_train_size: Union[int, float] = 0.9,
        replay_small: bool = False,
    ):
        self.batch_size = batch_size

        assert min_features <= max_features, "Invalid feature range"
        self.min_features = min_features
        self.max_features = max_features

        self.max_classes = max_classes
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.log_seq_len = log_seq_len

        self.validate_train_size_range(min_train_size, max_train_size)
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.replay_small = replay_small

    @staticmethod
    def validate_train_size_range(min_train_size: Union[int, float], max_train_size: Union[int, float]) -> None:
        """
        Checks if the training size range is valid.

        Parameters
        ----------
        min_train_size : int|float
            Minimum training size (position or ratio)

        max_train_size : int|float
            Maximum training size (position or ratio)

        Raises
        ------
        AssertionError
            If training size range is invalid
        ValueError
            If training size types are mismatched or invalid
        """
        # Check for numeric types only
        if not isinstance(min_train_size, (int, float)) or not isinstance(max_train_size, (int, float)):
            raise TypeError("Training sizes must be int or float")

        # Check for valid ranges based on type
        if isinstance(min_train_size, int) and isinstance(max_train_size, int):
            assert 0 < min_train_size < max_train_size, "0 < min_train_size < max_train_size"
        elif isinstance(min_train_size, float) and isinstance(max_train_size, float):
            assert 0 < min_train_size < max_train_size < 1, "0 < min_train_size < max_train_size < 1"
        else:
            raise ValueError("Both training sizes must be of the same type (int or float)")

    @staticmethod
    def sample_seq_len(
        min_seq_len: Optional[int], max_seq_len: int, log: bool = False, replay_small: bool = False
    ) -> int:
        """
        Selects a random sequence length within the specified range.

        This method provides flexible sampling strategies for dataset sizes, including
        occasional re-sampling of smaller sequence lengths for better training diversity.

        Parameters
        ----------
        min_seq_len : int, optional
            Minimum sequence length. If None, returns max_seq_len directly.

        max_seq_len : int
            Maximum sequence length

        log : bool, default=False
            If True, sample from a log-uniform distribution to better
            cover the range of possible sizes

        replay_small : bool, default=False
            If True, occasionally sample smaller sequence lengths with
            specific distributions to ensure model robustness on smaller datasets

        Returns
        -------
        int
            The sampled sequence length
        """
        if min_seq_len is None:
            return max_seq_len

        if log:
            seq_len = int(loguniform.rvs(min_seq_len, max_seq_len))
        else:
            seq_len = np.random.randint(min_seq_len, max_seq_len)

        if replay_small:
            p = np.random.random()
            if p < 0.05:
                return np.random.randint(200, 1000)
            elif p < 0.3:
                return int(loguniform.rvs(1000, 10000))
            else:
                return seq_len
        else:
            return seq_len

    @staticmethod
    def sample_train_size(min_train_size: Union[int, float], max_train_size: Union[int, float], seq_len: int) -> int:
        """
        Selects a random training size within the specified range.

        This method handles both absolute position and fractional ratio approaches
        for determining the training/test split point.

        Parameters
        ----------
        min_train_size : int|float
            Minimum training size. If int, used as absolute position.
            If float between 0 and 1, used as ratio of sequence length.

        max_train_size : int|float
            Maximum training size. If int, used as absolute position.
            If float between 0 and 1, used as ratio of sequence length.

        seq_len : int
            Total sequence length

        Returns
        -------
        int
            The sampled training size position

        Raises
        ------
        ValueError
            If training size range has incompatible types
        """
        if isinstance(min_train_size, int) and isinstance(max_train_size, int):
            train_size = np.random.randint(min_train_size, max_train_size)
        elif isinstance(min_train_size, float) and isinstance(min_train_size, float):
            train_size = np.random.uniform(min_train_size, max_train_size)
            train_size = int(seq_len * train_size)
        else:
            raise ValueError("Invalid training size range.")
        return train_size

    @staticmethod
    def adjust_max_features(seq_len: int, max_features: int) -> int:
        """
        Adjusts the maximum number of features based on the sequence length.

        This method implements an adaptive feature limit that scales inversely
        with sequence length. Longer sequences are restricted to fewer features
        to prevent memory issues and excessive computation times while still
        maintaining dataset diversity and learning difficulty.

        Parameters
        ----------
        seq_len : int
            Sequence length (number of samples)

        max_features : int
            Original maximum number of features

        Returns
        -------
        int
            Adjusted maximum number of features, ensuring computational feasibility
        """
        if seq_len <= 10240:
            return min(100, max_features)
        elif 10240 < seq_len <= 20000:
            return min(80, max_features)
        elif 20000 < seq_len <= 30000:
            return min(60, max_features)
        elif 30000 < seq_len <= 40000:
            return min(40, max_features)
        elif 40000 < seq_len <= 50000:
            return min(30, max_features)
        elif 50000 < seq_len <= 60000:
            return min(20, max_features)
        elif 60000 < seq_len <= 65000:
            return min(15, max_features)
        else:
            return 10

    @staticmethod
    def delete_unique_features(
        X: Tensor, d: Tensor, feature_binary: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Removes features that have only one unique value across all samples.

        Single-value features provide no useful information for learning since they
        have zero variance. This method identifies and removes such constant features
        to improve model training efficiency and stability. The removed features are
        replaced with zero padding to maintain tensor dimensions.

        Parameters
        ----------
        X : Tensor
            Input features tensor of shape (B, T, H) where:
            - B is batch size
            - T is sequence length
            - H is feature dimensionality

        d : Tensor
            Number of features per dataset of shape (B,), indicating how many
            features are actually used in each dataset (rest is padding)

        Returns
        -------
        tuple
            (X_new, d_new) where:
            - X_new is the filtered tensor with non-informative features removed
            - d_new is the updated feature count per dataset
        """

        def filter_unique_features(
            xi: Tensor, di: int, role_i: Optional[Tensor] = None
        ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
            """Filters features with only one unique value from a single dataset."""
            num_features = xi.shape[-1]
            # Only consider actual features (up to di, ignoring padding)
            xi = xi[:, :di]
            if role_i is not None:
                role_i = role_i[:di]
            # Identify features with more than one unique value (informative features)
            unique_mask = [len(torch.unique(xi[:, j])) > 1 for j in range(di)]
            di_new = sum(unique_mask)
            # Create new tensor with only informative features, padding the rest
            xi_new = F.pad(xi[:, unique_mask], pad=(0, num_features - di_new), mode="constant", value=0)
            if role_i is not None:
                role_new = F.pad(role_i[unique_mask], pad=(0, num_features - di_new), mode="constant", value=0)
            else:
                role_new = None
            return xi_new, torch.tensor(di_new, device=xi.device), role_new

        # Process each dataset in the batch independently
        if feature_binary is None:
            filtered_results = [filter_unique_features(xi, di, None) for xi, di in zip(X, d)]
        else:
            filtered_results = [filter_unique_features(xi, di, ri) for xi, di, ri in zip(X, d, feature_binary)]
        X_new, d_new, role_new = [list(res) for res in zip(*filtered_results)]
        X_new = torch.stack(X_new)
        d_new = torch.stack(d_new)
        role_tensor = torch.stack(role_new) if feature_binary is not None else None

        return X_new, d_new, role_tensor

    @staticmethod
    def sanity_check(X: Tensor, y: Tensor, train_size: int, n_attempts: int = 10, min_classes: int = 2) -> bool:
        """
        Verifies that both train and test sets contain all classes.

        For in-context learning to work properly, we need both the train and test
        sets to contain examples from all classes. This method checks this condition
        and attempts to fix invalid splits by randomly permuting the data.

        Parameters
        ----------
        X : Tensor
            Input features tensor of shape (B, T, H)

        y : Tensor
            Target labels tensor of shape (B, T)

        train_size : int
            Position to split the data into train and test sets

        n_attempts : int, default=10
            Number of random permutations to try for fixing invalid splits

        min_classes : int, default=2
            Minimum number of classes required in both train and test sets

        Returns
        -------
        bool
            True if all datasets have valid splits, False otherwise
        """

        def is_valid_split(yi: Tensor) -> bool:
            """Check if a single dataset has a valid train/test split."""
            # Guard against invalid train_size
            if train_size <= 0 or train_size >= yi.shape[0]:
                return False

            # A valid split requires both train and test sets to have the same classes
            # and at least min_classes different classes must be present
            unique_tr = torch.unique(yi[:train_size])
            unique_te = torch.unique(yi[train_size:])
            return set(unique_tr.tolist()) == set(unique_te.tolist()) and len(unique_tr) >= min_classes

        # Check each dataset in the batch
        for i, (xi, yi) in enumerate(zip(X, y)):
            if is_valid_split(yi):
                continue

            # If the dataset has an invalid split, try to fix it with random permutations
            succeeded = False
            for _ in range(n_attempts):
                # Generate a random permutation of the samples
                perm = torch.randperm(yi.shape[0])
                yi_perm = yi[perm]
                xi_perm = xi[perm]
                # Check if the permutation results in a valid split
                if is_valid_split(yi_perm):
                    X[i], y[i] = xi_perm, yi_perm
                    succeeded = True
                    break

            if not succeeded:  # No valid split was found after all attempts
                return False

        return True


class SCMPrior(Prior):
    """
    Generates synthetic datasets using Structural Causal Models (SCM).

    The data generation process follows a hierarchical structure:
    1. Generate a list of parameters for each dataset, respecting group/subgroup sharing.
    2. Process the parameter list to generate datasets, applying necessary transformations and checks.

    Parameters
    ----------
    batch_size : int, default=256
        Total number of datasets to generate per batch

    batch_size_per_gp : int, default=4
        Number of datasets per group, sharing similar characteristics

    batch_size_per_subgp : int, default=None
        Number of datasets per subgroup, with more similar causal structures
        If None, defaults to batch_size_per_gp

    min_features : int, default=2
        Minimum number of features per dataset

    max_features : int, default=100
        Maximum number of features per dataset

    max_classes : int, default=10
        Maximum number of target classes

    min_seq_len : int, default=None
        Minimum samples per dataset. If None, uses max_seq_len directly.

    max_seq_len : int, default=1024
        Maximum samples per dataset

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution

    seq_len_per_gp : bool = False
        If True, sample sequence length per group, allowing variable-sized datasets

    min_train_size : int|float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int|float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    replay_small : bool, default=False
        If True, occasionally sample smaller sequence lengths with
        specific distributions to ensure model robustness on smaller datasets

    prior_type : str, default="mlp_scm"
        Type of prior: 'mlp_scm' (default), 'tree_scm', or 'mix_scm'
        'mix_scm' randomly selects between 'mlp_scm' and 'tree_scm' based on probabilities.

    fixed_hp : dict, default=DEFAULT_FIXED_HP
        Fixed structural configuration parameters

    sampled_hp : dict, default=DEFAULT_SAMPLED_HP
        Parameters sampled during generation

    n_jobs : int, default=-1
        Number of parallel jobs to run (-1 means using all processors).

    num_threads_per_generate : int, default=1
        Number of threads per job for dataset generation

    device : str, default="cpu"
        Computation device ('cpu' or 'cuda')
    """

    def __init__(
        self,
        batch_size: int = 256,
        batch_size_per_gp: int = 4,
        batch_size_per_subgp: Optional[int] = None,
        min_features: int = 2,
        max_features: int = 100,
        max_classes: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        seq_len_per_gp: bool = False,
        min_train_size: Union[int, float] = 0.1,
        max_train_size: Union[int, float] = 0.9,
        replay_small: bool = False,
        prior_type: str = "mlp_scm",
        fixed_hp: Dict[str, Any] = DEFAULT_FIXED_HP,
        sampled_hp: Dict[str, Any] = DEFAULT_SAMPLED_HP,
        return_x_node_binary: bool = False,
        node_pseudo_label_method: str = "none",
        node_ci_alpha: float = 0.01,
        node_ci_max_condition_set: int = 2,
        n_jobs: int = -1,
        num_threads_per_generate: int = 1,
        device: str = "cpu",
    ):
        super().__init__(
            batch_size=batch_size,
            min_features=min_features,
            max_features=max_features,
            max_classes=max_classes,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            log_seq_len=log_seq_len,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
            replay_small=replay_small,
        )

        self.batch_size_per_gp = batch_size_per_gp
        self.batch_size_per_subgp = batch_size_per_subgp or batch_size_per_gp
        self.seq_len_per_gp = seq_len_per_gp
        self.prior_type = prior_type
        self.fixed_hp = fixed_hp
        self.sampled_hp = sampled_hp
        self.return_x_node_binary = return_x_node_binary
        self.node_pseudo_label_method = str(node_pseudo_label_method).lower()
        valid_methods = {
            "none",
            "iamb_fdr",
            "mmpc",
            "corr_fdr",
            "mi_fdr",
            "lasso",
            "mlp_saliency",
            "l0_resnet_probe",
            "tabtransformer_probe",
        }
        if self.node_pseudo_label_method not in valid_methods:
            raise ValueError(
                "node_pseudo_label_method must be one of: "
                "none, iamb_fdr, mmpc, corr_fdr, mi_fdr, lasso, mlp_saliency, l0_resnet_probe, tabtransformer_probe. "
                f"Got: {self.node_pseudo_label_method}"
            )
        if not (0.0 < float(node_ci_alpha) < 1.0):
            raise ValueError(f"node_ci_alpha must be in (0, 1), got {node_ci_alpha}")
        if int(node_ci_max_condition_set) < 0:
            raise ValueError(
                f"node_ci_max_condition_set must be >= 0, got {node_ci_max_condition_set}"
            )
        self.node_ci_alpha = float(node_ci_alpha)
        self.node_ci_max_condition_set = int(node_ci_max_condition_set)
        self.n_jobs = n_jobs
        self.num_threads_per_generate = num_threads_per_generate
        self.device = device

    def hp_sampling(self) -> Dict[str, Any]:
        """
        Sample hyperparameters for dataset generation.

        Returns
        -------
        dict
            Dictionary with sampled hyperparameters merged with fixed ones
        """
        hp_sampler = HpSamplerList(self.sampled_hp, device=self.device)
        return hp_sampler.sample()

    @torch.no_grad()
    def generate_dataset(self, params: Dict[str, Any]) -> Tuple[Tensor, ...]:
        """
        Generates a single valid dataset based on the provided parameters.

        Parameters
        ----------
        params : dict
            Hyperparameters for generating this specific dataset, including seq_len,
            train_size, num_features, num_classes, prior_type, device, etc.

        Returns
        -------
        tuple
            (X, y, d) where:
            - X: Features tensor of shape (seq_len, max_features)
            - y: Labels tensor of shape (seq_len,)
            - d: Number of active features after filtering (scalar Tensor)
        """

        if params["prior_type"] == "mlp_scm":
            prior_cls = MLPSCM
        elif params["prior_type"] == "tree_scm":
            prior_cls = TreeSCM
        else:
            raise ValueError(f"Unknown prior type {params['prior_type']}")

        while True:
            prior = prior_cls(**params)
            X, y = prior()
            feature_binary = getattr(prior, "last_x_binary_roles", None)
            pseudo_mb_acc = None

            if self.return_x_node_binary and params["prior_type"] == "mlp_scm" and feature_binary is not None:
                X, y, feature_binary = Reg2Cls(params)(X, y, feature_binary=feature_binary)
                feature_binary = feature_binary.long()
                if self.node_pseudo_label_method != "none":
                    pseudo_binary = infer_mb_pseudo_labels(
                        X,
                        y,
                        method=self.node_pseudo_label_method,
                        ci_alpha=self.node_ci_alpha,
                        max_condition_set=self.node_ci_max_condition_set,
                    )
                    pseudo_mb_acc = (pseudo_binary == feature_binary).float().mean()
                    feature_binary = pseudo_binary
            else:
                X, y = Reg2Cls(params)(X, y)
                feature_binary = torch.zeros(X.shape[1], device=X.device, dtype=torch.long) if self.return_x_node_binary else None

            # Add batch dim for single dataset to be compatible with delete_unique_features and sanity_check
            X, y = X.unsqueeze(0), y.unsqueeze(0)
            d = torch.tensor([params["num_features"]], device=self.device, dtype=torch.long)
            if feature_binary is not None:
                feature_binary = feature_binary.unsqueeze(0).long()

            # Only keep valid datasets with sufficient features and balanced classes
            X, d, feature_binary = self.delete_unique_features(X, d, feature_binary=feature_binary)
            if (d > 0).all() and self.sanity_check(X, y, params["train_size"]):
                if feature_binary is not None:
                    if self.node_pseudo_label_method != "none":
                        if pseudo_mb_acc is None:
                            pseudo_mb_acc = torch.tensor(float("nan"), device=X.device)
                        return X.squeeze(0), y.squeeze(0), d.squeeze(0), feature_binary.squeeze(0), pseudo_mb_acc
                    return X.squeeze(0), y.squeeze(0), d.squeeze(0), feature_binary.squeeze(0)
                return X.squeeze(0), y.squeeze(0), d.squeeze(0)

    @torch.no_grad()
    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, ...]:
        """
        Generates a batch of datasets by first creating a parameter list and then processing it.

        Parameters
        ----------
        batch_size : int, optional
            Batch size override. If None, uses self.batch_size

        Returns
        -------
        X : Tensor or NestedTensor
            Features tensor. If seq_len_per_gp=False, shape is (batch_size, seq_len, max_features).
            If seq_len_per_gp=True, returns a NestedTensor.

        y : Tensor or NestedTensor
            Labels tensor. If seq_len_per_gp=False, shape is (batch_size, seq_len).
            If seq_len_per_gp=True, returns a NestedTensor.

        d : Tensor
            Number of active features per dataset after filtering, shape (batch_size,)

        seq_lens : Tensor
            Sequence length for each dataset, shape (batch_size,)

        train_sizes : Tensor
            Position for train/test split for each dataset, shape (batch_size,)
        """
        batch_size = batch_size or self.batch_size

        # Calculate number of groups and subgroups
        size_per_gp = min(self.batch_size_per_gp, batch_size)
        num_gps = math.ceil(batch_size / size_per_gp)

        size_per_subgp = min(self.batch_size_per_subgp, size_per_gp)

        # Generate parameters list for all datasets, preserving group and subgroup structure
        param_list = []
        global_seq_len = None
        global_train_size = None

        # Determine global seq_len/train_size if not per-group
        if not self.seq_len_per_gp:
            global_seq_len = self.sample_seq_len(
                self.min_seq_len, self.max_seq_len, log=self.log_seq_len, replay_small=self.replay_small
            )
            global_train_size = self.sample_train_size(self.min_train_size, self.max_train_size, global_seq_len)

        # Generate parameters for each group
        for gp_idx in range(num_gps):
            # Determine actual size for this group (may be smaller for the last group)
            actual_gp_size = min(size_per_gp, batch_size - gp_idx * size_per_gp)
            if actual_gp_size <= 0:
                break

            group_sampled_hp = self.hp_sampling()
            # If per-group, sample seq_len and train_size for this group. Otherwise, use global ones
            if self.seq_len_per_gp:
                gp_seq_len = self.sample_seq_len(
                    self.min_seq_len, self.max_seq_len, log=self.log_seq_len, replay_small=self.replay_small
                )
                gp_train_size = self.sample_train_size(self.min_train_size, self.max_train_size, gp_seq_len)
                # Adjust max features based on seq_len for this group
                gp_max_features = self.adjust_max_features(gp_seq_len, self.max_features)
            else:
                gp_seq_len = global_seq_len
                gp_train_size = global_train_size
                gp_max_features = self.max_features

            # Calculate number of subgroups for this group
            num_subgps_in_gp = math.ceil(actual_gp_size / size_per_subgp)

            # Generate parameters for each subgroup
            for subgp_idx in range(num_subgps_in_gp):
                # Determine actual size for this subgroup
                actual_subgp_size = min(size_per_subgp, actual_gp_size - subgp_idx * size_per_subgp)
                if actual_subgp_size <= 0:
                    break

                # Subgroups share prior type, number of features, and sampled HPs
                subgp_prior_type = self.get_prior()
                subgp_num_features = round(np.random.uniform(self.min_features, gp_max_features))
                subgp_sampled_hp = {k: v() if callable(v) else v for k, v in group_sampled_hp.items()}

                # Generate parameters for each dataset in this subgroup
                for ds_idx in range(actual_subgp_size):
                    # Each dataset has its own number of classes
                    if np.random.random() > 0.5:
                        ds_num_classes = np.random.randint(2, self.max_classes + 1)
                    else:
                        ds_num_classes = 2

                    # Create parameters dictionary for this dataset
                    params = {
                        **self.fixed_hp,  # Fixed HPs
                        "seq_len": gp_seq_len,
                        "train_size": gp_train_size,
                        # If per-gp setting, use adjusted max features for this group because we use nested tensors
                        # If not per-gp setting, use global max features to fix size for concatenation
                        "max_features": gp_max_features if self.seq_len_per_gp else self.max_features,
                        **subgp_sampled_hp,  # sampled HPs for this group
                        "prior_type": subgp_prior_type,
                        "num_features": subgp_num_features,
                        "num_classes": ds_num_classes,
                        "device": self.device,
                    }
                    param_list.append(params)

        # Use joblib to generate datasets in parallel.
        # Note: the 'loky' backend does not support nested parallelism during DDP, whereas the 'threading' backend does.
        # However, 'threading' does not respect `inner_max_num_threads`.
        # Therefore, we stick with the 'loky' backend for parallelism, but this requires generating
        # the prior datasets separately from the training process and loading them from disk,
        # rather than generating them on-the-fly.
        if self.n_jobs > 1 and self.device == "cpu":
            with joblib.parallel_config(
                n_jobs=self.n_jobs, backend="loky", inner_max_num_threads=self.num_threads_per_generate
            ):
                results = joblib.Parallel()(joblib.delayed(self.generate_dataset)(params) for params in param_list)
        else:
            results = [self.generate_dataset(params) for params in param_list]

        if self.return_x_node_binary:
            if self.node_pseudo_label_method != "none":
                X_list, y_list, d_list, x_node_binary_list, pseudo_mb_acc_list = zip(*results)
            else:
                X_list, y_list, d_list, x_node_binary_list = zip(*results)
        else:
            X_list, y_list, d_list = zip(*results)

        # Combine Results
        if self.seq_len_per_gp:
            # Use nested tensors for variable sequence lengths
            X = nested_tensor([x.to(self.device) for x in X_list], device=self.device)
            y = nested_tensor([y.to(self.device) for y in y_list], device=self.device)
        else:
            # Stack into regular tensors for fixed sequence length
            X = torch.stack(X_list).to(self.device)  # (B, T, H)
            y = torch.stack(y_list).to(self.device)  # (B, T)

        # Metadata (always regular tensors)
        d = torch.stack(d_list).to(self.device)  # Actual number of features after filtering out constant ones
        seq_lens = torch.tensor([params["seq_len"] for params in param_list], device=self.device, dtype=torch.long)
        train_sizes = torch.tensor(
            [params["train_size"] for params in param_list], device=self.device, dtype=torch.long
        )

        if self.return_x_node_binary:
            x_node_binary = torch.stack(x_node_binary_list).to(self.device)  # (B, max_features), 0/1
            if self.node_pseudo_label_method != "none":
                pseudo_mb_acc = torch.stack(pseudo_mb_acc_list).to(self.device).float()
                return X, y, d, seq_lens, train_sizes, x_node_binary, pseudo_mb_acc
            return X, y, d, seq_lens, train_sizes, x_node_binary
        return X, y, d, seq_lens, train_sizes

    def get_prior(self) -> str:
        """
        Determine which prior type to use for generation.

        For 'mix_scm' prior type, randomly selects between available priors
        based on configured probabilities.

        Returns
        -------
        str
            The selected prior type name
        """
        if self.prior_type == "mix_scm":
            return np.random.choice(["mlp_scm", "tree_scm"], p=self.fixed_hp.get("mix_probas", [0.7, 0.3]))
        else:
            return self.prior_type


class DummyPrior(Prior):
    """This class creates purely random data. This is useful for testing and debugging
    without the computational overhead of SCM-based generation.

    Parameters
    ----------
    batch_size : int, default=256
        Number of datasets to generate

    min_features : int, default=2
        Minimum number of features per dataset

    max_features : int, default=100
        Maximum number of features per dataset

    max_classes : int, default=10
        Maximum number of target classes

    min_seq_len : int, default=None
        Minimum samples per dataset. If None, uses max_seq_len directly.

    max_seq_len : int, default=1024
        Maximum samples per dataset

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution

    min_train_size : int|float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int|float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    device : str, default="cpu"
        Computation device
    """

    def __init__(
        self,
        batch_size: int = 256,
        min_features: int = 2,
        max_features: int = 100,
        max_classes: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        min_train_size: Union[int, float] = 0.1,
        max_train_size: Union[int, float] = 0.9,
        device: str = "cpu",
    ):
        super().__init__(
            batch_size=batch_size,
            min_features=min_features,
            max_features=max_features,
            max_classes=max_classes,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            log_seq_len=log_seq_len,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
        )
        self.device = device

    @torch.no_grad()
    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Generates a batch of random datasets for testing purposes.

        Parameters
        ----------
        batch_size : int, optional
            Batch size override, if None, uses self.batch_size

        Returns
        -------
        X : Tensor
            Features tensor of shape (batch_size, seq_len, max_features).
            Contains random Gaussian values for all features.

        y : Tensor
            Labels tensor of shape (batch_size, seq_len).
            Contains randomly assigned class labels.

        d : Tensor
            Number of features per dataset of shape (batch_size,).
            Always set to max_features for DummyPrior.

        seq_lens : Tensor
            Sequence length for each dataset of shape (batch_size,).
            All datasets share the same sequence length.

        train_sizes : Tensor
            Position for train/test split for each dataset of shape (batch_size,).
            All datasets share the same split position.
        """

        batch_size = batch_size or self.batch_size
        seq_len = self.sample_seq_len(self.min_seq_len, self.max_seq_len, log=self.log_seq_len)
        train_size = self.sample_train_size(self.min_train_size, self.max_train_size, seq_len)

        X = torch.randn(batch_size, seq_len, self.max_features, device=self.device)

        num_classes = np.random.randint(2, self.max_classes + 1)
        y = torch.randint(0, num_classes, (batch_size, seq_len), device=self.device)

        d = torch.full((batch_size,), self.max_features, device=self.device)
        seq_lens = torch.full((batch_size,), seq_len, device=self.device)
        train_sizes = torch.full((batch_size,), train_size, device=self.device)

        return X, y, d, seq_lens, train_sizes


class PriorDataset(IterableDataset):
    """
    Main dataset class that provides an infinite iterator over synthetic tabular datasets.

    Parameters
    ----------
    batch_size : int, default=256
        Total number of datasets to generate per batch

    batch_size_per_gp : int, default=4
        Number of datasets per group, sharing similar characteristics

    batch_size_per_subgp : int, default=None
        Number of datasets per subgroup, with more similar causal structures
        If None, defaults to batch_size_per_gp

    min_features : int, default=2
        Minimum number of features per dataset

    max_features : int, default=100
        Maximum number of features per dataset

    max_classes : int, default=10
        Maximum number of target classes

    min_seq_len : int, default=None
        Minimum samples per dataset. If None, uses max_seq_len directly.

    max_seq_len : int, default=1024
        Maximum samples per dataset

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution

    seq_len_per_gp : bool = False
        If True, sample sequence length per group, allowing variable-sized datasets

    min_train_size : int|float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int|float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    replay_small : bool, default=False
        If True, occasionally sample smaller sequence lengths with
        specific distributions to ensure model robustness on smaller datasets

    prior_type : str, default="mlp_scm"
        Type of prior: 'mlp_scm' (default), 'tree_scm', 'mix_scm', or 'dummy'

        1. SCM-based: Structural causal models with complex feature relationships
         - 'mlp_scm': MLP-based causal models
         - 'tree_scm': Tree-based causal models
         - 'mix_scm': Probabilistic mix of the above models

        2. Dummy: Randomly generated datasets for debugging

    scm_fixed_hp : dict, default=DEFAULT_FIXED_HP
        Fixed parameters for SCM-based priors

    scm_sampled_hp : dict, default=DEFAULT_SAMPLED_HP
        Parameters sampled during generation

    n_jobs : int, default=-1
        Number of parallel jobs to run (-1 means using all processors)

    num_threads_per_generate : int, default=1
        Number of threads per job for dataset generation

    device : str, default="cpu"
        Computation device ('cpu' or 'cuda')
    """

    def __init__(
        self,
        batch_size: int = 256,
        batch_size_per_gp: int = 4,
        batch_size_per_subgp: Optional[int] = None,
        min_features: int = 2,
        max_features: int = 100,
        max_classes: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        seq_len_per_gp: bool = False,
        min_train_size: Union[int, float] = 0.1,
        max_train_size: Union[int, float] = 0.9,
        replay_small: bool = False,
        prior_type: str = "mlp_scm",
        scm_fixed_hp: Dict[str, Any] = DEFAULT_FIXED_HP,
        scm_sampled_hp: Dict[str, Any] = DEFAULT_SAMPLED_HP,
        return_x_node_binary: bool = False,
        node_pseudo_label_method: str = "none",
        node_ci_alpha: float = 0.01,
        node_ci_max_condition_set: int = 2,
        n_jobs: int = -1,
        num_threads_per_generate: int = 1,
        device: str = "cpu",
    ):
        super().__init__()
        if prior_type == "dummy":
            self.prior = DummyPrior(
                batch_size=batch_size,
                min_features=min_features,
                max_features=max_features,
                max_classes=max_classes,
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                log_seq_len=log_seq_len,
                min_train_size=min_train_size,
                max_train_size=max_train_size,
                device=device,
            )
        elif prior_type in ["mlp_scm", "tree_scm", "mix_scm"]:
            self.prior = SCMPrior(
                batch_size=batch_size,
                batch_size_per_gp=batch_size_per_gp,
                batch_size_per_subgp=batch_size_per_subgp,
                min_features=min_features,
                max_features=max_features,
                max_classes=max_classes,
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                log_seq_len=log_seq_len,
                seq_len_per_gp=seq_len_per_gp,
                min_train_size=min_train_size,
                max_train_size=max_train_size,
                replay_small=replay_small,
                prior_type=prior_type,
                fixed_hp=scm_fixed_hp,
                sampled_hp=scm_sampled_hp,
                return_x_node_binary=return_x_node_binary,
                node_pseudo_label_method=node_pseudo_label_method,
                node_ci_alpha=node_ci_alpha,
                node_ci_max_condition_set=node_ci_max_condition_set,
                n_jobs=n_jobs,
                num_threads_per_generate=num_threads_per_generate,
                device=device,
            )
        else:
            raise ValueError(
                f"Unknown prior type '{prior_type}'. Available options: 'mlp_scm', 'tree_scm', 'mix_scm', or 'dummy'."
            )

        self.batch_size = batch_size
        self.batch_size_per_gp = batch_size_per_gp
        self.batch_size_per_subgp = batch_size_per_subgp or batch_size_per_gp
        self.min_features = min_features
        self.max_features = max_features
        self.max_classes = max_classes
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.log_seq_len = log_seq_len
        self.seq_len_per_gp = seq_len_per_gp
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.device = device
        self.prior_type = prior_type
        self.return_x_node_binary = return_x_node_binary
        self.node_pseudo_label_method = str(node_pseudo_label_method).lower()
        self.node_ci_alpha = float(node_ci_alpha)
        self.node_ci_max_condition_set = int(node_ci_max_condition_set)

    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Generate a new batch of datasets.

        Parameters
        ----------
        batch_size : int, optional
            If provided, overrides the default batch size for this call

        Returns
        -------
        X : Tensor or NestedTensor
            1. For SCM-based priors:
             - If seq_len_per_gp=False, shape is (batch_size, seq_len, max_features).
             - If seq_len_per_gp=True, returns a NestedTensor.

            2. For DummyPrior, random Gaussian values of (batch_size, seq_len, max_features).

        X : Tensor or NestedTensor
            1. For SCM-based priors:
             - If seq_len_per_gp=False, shape is (batch_size, seq_len).
             - If seq_len_per_gp=True, returns a NestedTensor.

            2. For DummyPrior, random class labels of (batch_size, seq_len).

        d : Tensor
            Number of active features per dataset of shape (batch_size,).

        seq_lens : Tensor
            Sequence length for each dataset of shape (batch_size,).

        train_sizes : Tensor
            Position for train/test split for each dataset of shape (batch_size,).
        """
        return self.prior.get_batch(batch_size)

    def __iter__(self) -> "PriorDataset":
        """
        Returns an iterator that yields batches indefinitely.

        Returns
        -------
        self
            Returns self as an iterator
        """
        return self

    def __next__(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Returns the next batch from the iterator. Since this is an infinite
        iterator, it never raises StopIteration and instead continuously generates
        new synthetic data batches.
        """
        with DisablePrinting():
            return self.get_batch()

    def __repr__(self) -> str:
        """
        Returns a string representation of the dataset.

        Provides a detailed view of the dataset configuration for debugging
        and logging purposes.

        Returns
        -------
        str
            A formatted string with dataset parameters
        """
        return (
            f"PriorDataset(\n"
            f"  prior_type: {self.prior_type}\n"
            f"  batch_size: {self.batch_size}\n"
            f"  batch_size_per_gp: {self.batch_size_per_gp}\n"
            f"  features: {self.min_features} - {self.max_features}\n"
            f"  max classes: {self.max_classes}\n"
            f"  seq_len: {self.min_seq_len or 'None'} - {self.max_seq_len}\n"
            f"  sequence length varies across groups: {self.seq_len_per_gp}\n"
            f"  train_size: {self.min_train_size} - {self.max_train_size}\n"
            f"  node_pseudo_label_method: {self.node_pseudo_label_method}\n"
            f"  device: {self.device}\n"
            f")"
        )


class DisablePrinting:
    """Context manager to temporarily suppress printed output."""

    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self.original_stdout

# """
# The module offers a flexible framework for creating diverse, realistic tabular datasets
# with controlled properties, which can be used for training and evaluating in-context
# learning models. Key features include:

# - Controlled feature relationships and causal structures via multiple generation methods
# - Customizable feature distributions with mixed continuous and categorical variables
# - Flexible train/test splits optimized for in-context learning evaluation
# - Batch generation capabilities with hierarchical parameter sharing
# - Memory-efficient handling of variable-length datasets

# The main class is PriorDataset, which provides an iterable interface for generating
# an infinite stream of synthetic datasets with diverse characteristics.
# """

# from __future__ import annotations

# import os
# import sys
# import math
# import warnings
# from collections import defaultdict
# from typing import Dict, Tuple, Union, Optional, Any, List

# import numpy as np
# from scipy.stats import loguniform
# import joblib

# import torch
# import torch.nn.functional as F
# from torch import Tensor
# from torch.nested import nested_tensor
# from torch.utils.data import IterableDataset

# from .mlp_scm import MLPSCM
# from .tree_scm import TreeSCM

# from .hp_sampling import HpSamplerList
# from .reg2cls import Reg2Cls
# from .prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP


# warnings.filterwarnings(
#     "ignore", message=".*The PyTorch API of nested tensors is in prototype stage.*", category=UserWarning
# )


# class Prior:
#     """
#     Abstract base class for dataset prior generators.

#     Defines the interface and common functionality for different types of
#     synthetic dataset generators.

#     Parameters
#     ----------
#     batch_size : int, default=256
#         Total number of datasets to generate per batch

#     min_features : int, default=2
#         Minimum number of features per dataset

#     max_features : int, default=100
#         Maximum number of features per dataset

#     max_classes : int, default=10
#         Maximum number of target classes

#     min_seq_len : int, default=None
#         Minimum samples per dataset. If None, uses max_seq_len

#     max_seq_len : int, default=1024
#         Maximum samples per dataset

#     log_seq_len : bool, default=False
#         If True, sample sequence length from a log-uniform distribution

#     min_train_size : int|float, default=0.1
#         Position or ratio for train/test split start. If int, absolute position.
#         If float between 0 and 1, specifies a fraction of sequence length.

#     max_train_size : int|float, default=0.9
#         Position or ratio for train/test split end. If int, absolute position.
#         If float between 0 and 1, specifies a fraction of sequence length.

#     replay_small : bool, default=False
#         If True, occasionally sample smaller sequence lengths with
#         specific distributions to ensure model robustness on smaller datasets
#     """

#     def __init__(
#         self,
#         batch_size: int = 256,
#         min_features: int = 2,
#         max_features: int = 100,
#         max_classes: int = 10,
#         min_seq_len: Optional[int] = None,
#         max_seq_len: int = 1024,
#         log_seq_len: bool = False,
#         min_train_size: Union[int, float] = 0.1,
#         max_train_size: Union[int, float] = 0.9,
#         replay_small: bool = False,
#     ):
#         self.batch_size = batch_size

#         assert min_features <= max_features, "Invalid feature range"
#         self.min_features = min_features
#         self.max_features = max_features

#         self.max_classes = max_classes
#         self.min_seq_len = min_seq_len
#         self.max_seq_len = max_seq_len
#         self.log_seq_len = log_seq_len

#         self.validate_train_size_range(min_train_size, max_train_size)
#         self.min_train_size = min_train_size
#         self.max_train_size = max_train_size
#         self.replay_small = replay_small

#     @staticmethod
#     def validate_train_size_range(min_train_size: Union[int, float], max_train_size: Union[int, float]) -> None:
#         """
#         Checks if the training size range is valid.

#         Parameters
#         ----------
#         min_train_size : int|float
#             Minimum training size (position or ratio)

#         max_train_size : int|float
#             Maximum training size (position or ratio)

#         Raises
#         ------
#         AssertionError
#             If training size range is invalid
#         ValueError
#             If training size types are mismatched or invalid
#         """
#         # Check for numeric types only
#         if not isinstance(min_train_size, (int, float)) or not isinstance(max_train_size, (int, float)):
#             raise TypeError("Training sizes must be int or float")

#         # Check for valid ranges based on type
#         if isinstance(min_train_size, int) and isinstance(max_train_size, int):
#             assert 0 < min_train_size < max_train_size, "0 < min_train_size < max_train_size"
#         elif isinstance(min_train_size, float) and isinstance(max_train_size, float):
#             assert 0 < min_train_size < max_train_size < 1, "0 < min_train_size < max_train_size < 1"
#         else:
#             raise ValueError("Both training sizes must be of the same type (int or float)")

#     @staticmethod
#     def sample_seq_len(
#         min_seq_len: Optional[int], max_seq_len: int, log: bool = False, replay_small: bool = False
#     ) -> int:
#         """
#         Selects a random sequence length within the specified range.

#         This method provides flexible sampling strategies for dataset sizes, including
#         occasional re-sampling of smaller sequence lengths for better training diversity.

#         Parameters
#         ----------
#         min_seq_len : int, optional
#             Minimum sequence length. If None, returns max_seq_len directly.

#         max_seq_len : int
#             Maximum sequence length

#         log : bool, default=False
#             If True, sample from a log-uniform distribution to better
#             cover the range of possible sizes

#         replay_small : bool, default=False
#             If True, occasionally sample smaller sequence lengths with
#             specific distributions to ensure model robustness on smaller datasets

#         Returns
#         -------
#         int
#             The sampled sequence length
#         """
#         if min_seq_len is None:
#             return max_seq_len

#         if log:
#             seq_len = int(loguniform.rvs(min_seq_len, max_seq_len))
#         else:
#             seq_len = np.random.randint(min_seq_len, max_seq_len)

#         if replay_small:
#             p = np.random.random()
#             if p < 0.05:
#                 return np.random.randint(200, 1000)
#             elif p < 0.3:
#                 return int(loguniform.rvs(1000, 10000))
#             else:
#                 return seq_len
#         else:
#             return seq_len

#     @staticmethod
#     def sample_train_size(min_train_size: Union[int, float], max_train_size: Union[int, float], seq_len: int) -> int:
#         """
#         Selects a random training size within the specified range.

#         This method handles both absolute position and fractional ratio approaches
#         for determining the training/test split point.

#         Parameters
#         ----------
#         min_train_size : int|float
#             Minimum training size. If int, used as absolute position.
#             If float between 0 and 1, used as ratio of sequence length.

#         max_train_size : int|float
#             Maximum training size. If int, used as absolute position.
#             If float between 0 and 1, used as ratio of sequence length.

#         seq_len : int
#             Total sequence length

#         Returns
#         -------
#         int
#             The sampled training size position

#         Raises
#         ------
#         ValueError
#             If training size range has incompatible types
#         """
#         if isinstance(min_train_size, int) and isinstance(max_train_size, int):
#             train_size = np.random.randint(min_train_size, max_train_size)
#         elif isinstance(min_train_size, float) and isinstance(min_train_size, float):
#             train_size = np.random.uniform(min_train_size, max_train_size)
#             train_size = int(seq_len * train_size)
#         else:
#             raise ValueError("Invalid training size range.")
#         return train_size

#     @staticmethod
#     def adjust_max_features(seq_len: int, max_features: int) -> int:
#         """
#         Adjusts the maximum number of features based on the sequence length.

#         This method implements an adaptive feature limit that scales inversely
#         with sequence length. Longer sequences are restricted to fewer features
#         to prevent memory issues and excessive computation times while still
#         maintaining dataset diversity and learning difficulty.

#         Parameters
#         ----------
#         seq_len : int
#             Sequence length (number of samples)

#         max_features : int
#             Original maximum number of features

#         Returns
#         -------
#         int
#             Adjusted maximum number of features, ensuring computational feasibility
#         """
#         if seq_len <= 10240:
#             return min(100, max_features)
#         elif 10240 < seq_len <= 20000:
#             return min(80, max_features)
#         elif 20000 < seq_len <= 30000:
#             return min(60, max_features)
#         elif 30000 < seq_len <= 40000:
#             return min(40, max_features)
#         elif 40000 < seq_len <= 50000:
#             return min(30, max_features)
#         elif 50000 < seq_len <= 60000:
#             return min(20, max_features)
#         elif 60000 < seq_len <= 65000:
#             return min(15, max_features)
#         else:
#             return 10

#     @staticmethod
#     def delete_unique_features(X: Tensor, d: Tensor) -> Tuple[Tensor, Tensor]:
#         """
#         Removes features that have only one unique value across all samples.

#         Single-value features provide no useful information for learning since they
#         have zero variance. This method identifies and removes such constant features
#         to improve model training efficiency and stability. The removed features are
#         replaced with zero padding to maintain tensor dimensions.

#         Parameters
#         ----------
#         X : Tensor
#             Input features tensor of shape (B, T, H) where:
#             - B is batch size
#             - T is sequence length
#             - H is feature dimensionality

#         d : Tensor
#             Number of features per dataset of shape (B,), indicating how many
#             features are actually used in each dataset (rest is padding)

#         Returns
#         -------
#         tuple
#             (X_new, d_new) where:
#             - X_new is the filtered tensor with non-informative features removed
#             - d_new is the updated feature count per dataset
#         """

#         def filter_unique_features(xi: Tensor, di: int) -> Tuple[Tensor, Tensor]:
#             """Filters features with only one unique value from a single dataset."""
#             num_features = xi.shape[-1]
#             # Only consider actual features (up to di, ignoring padding)
#             xi = xi[:, :di]
#             # Identify features with more than one unique value (informative features)
#             unique_mask = [len(torch.unique(xi[:, j])) > 1 for j in range(di)]
#             di_new = sum(unique_mask)
#             # Create new tensor with only informative features, padding the rest
#             xi_new = F.pad(xi[:, unique_mask], pad=(0, num_features - di_new), mode="constant", value=0)
#             return xi_new, torch.tensor(di_new, device=xi.device)

#         # Process each dataset in the batch independently
#         filtered_results = [filter_unique_features(xi, di) for xi, di in zip(X, d)]
#         X_new, d_new = [torch.stack(res) for res in zip(*filtered_results)]

#         return X_new, d_new

#     @staticmethod
#     def sanity_check(X: Tensor, y: Tensor, train_size: int, n_attempts: int = 10, min_classes: int = 2) -> bool:
#         """
#         Verifies that both train and test sets contain all classes.

#         For in-context learning to work properly, we need both the train and test
#         sets to contain examples from all classes. This method checks this condition
#         and attempts to fix invalid splits by randomly permuting the data.

#         Parameters
#         ----------
#         X : Tensor
#             Input features tensor of shape (B, T, H)

#         y : Tensor
#             Target labels tensor of shape (B, T)

#         train_size : int
#             Position to split the data into train and test sets

#         n_attempts : int, default=10
#             Number of random permutations to try for fixing invalid splits

#         min_classes : int, default=2
#             Minimum number of classes required in both train and test sets

#         Returns
#         -------
#         bool
#             True if all datasets have valid splits, False otherwise
#         """

#         def is_valid_split(yi: Tensor) -> bool:
#             """Check if a single dataset has a valid train/test split."""
#             # Guard against invalid train_size
#             if train_size <= 0 or train_size >= yi.shape[0]:
#                 return False

#             # A valid split requires both train and test sets to have the same classes
#             # and at least min_classes different classes must be present
#             unique_tr = torch.unique(yi[:train_size])
#             unique_te = torch.unique(yi[train_size:])
#             return set(unique_tr.tolist()) == set(unique_te.tolist()) and len(unique_tr) >= min_classes

#         # Check each dataset in the batch
#         for i, (xi, yi) in enumerate(zip(X, y)):
#             if is_valid_split(yi):
#                 continue

#             # If the dataset has an invalid split, try to fix it with random permutations
#             succeeded = False
#             for _ in range(n_attempts):
#                 # Generate a random permutation of the samples
#                 perm = torch.randperm(yi.shape[0])
#                 yi_perm = yi[perm]
#                 xi_perm = xi[perm]
#                 # Check if the permutation results in a valid split
#                 if is_valid_split(yi_perm):
#                     X[i], y[i] = xi_perm, yi_perm
#                     succeeded = True
#                     break

#             if not succeeded:  # No valid split was found after all attempts
#                 return False

#         return True


# class SCMPrior(Prior):
#     """
#     Generates synthetic datasets using Structural Causal Models (SCM).

#     The data generation process follows a hierarchical structure:
#     1. Generate a list of parameters for each dataset, respecting group/subgroup sharing.
#     2. Process the parameter list to generate datasets, applying necessary transformations and checks.

#     Parameters
#     ----------
#     batch_size : int, default=256
#         Total number of datasets to generate per batch

#     batch_size_per_gp : int, default=4
#         Number of datasets per group, sharing similar characteristics

#     batch_size_per_subgp : int, default=None
#         Number of datasets per subgroup, with more similar causal structures
#         If None, defaults to batch_size_per_gp

#     min_features : int, default=2
#         Minimum number of features per dataset

#     max_features : int, default=100
#         Maximum number of features per dataset

#     max_classes : int, default=10
#         Maximum number of target classes

#     min_seq_len : int, default=None
#         Minimum samples per dataset. If None, uses max_seq_len directly.

#     max_seq_len : int, default=1024
#         Maximum samples per dataset

#     log_seq_len : bool, default=False
#         If True, sample sequence length from a log-uniform distribution

#     seq_len_per_gp : bool = False
#         If True, sample sequence length per group, allowing variable-sized datasets

#     min_train_size : int|float, default=0.1
#         Position or ratio for train/test split start. If int, absolute position.
#         If float between 0 and 1, specifies a fraction of sequence length.

#     max_train_size : int|float, default=0.9
#         Position or ratio for train/test split end. If int, absolute position.
#         If float between 0 and 1, specifies a fraction of sequence length.

#     replay_small : bool, default=False
#         If True, occasionally sample smaller sequence lengths with
#         specific distributions to ensure model robustness on smaller datasets

#     prior_type : str, default="mlp_scm"
#         Type of prior: 'mlp_scm' (default), 'tree_scm', or 'mix_scm'
#         'mix_scm' randomly selects between 'mlp_scm' and 'tree_scm' based on probabilities.

#     fixed_hp : dict, default=DEFAULT_FIXED_HP
#         Fixed structural configuration parameters

#     sampled_hp : dict, default=DEFAULT_SAMPLED_HP
#         Parameters sampled during generation

#     n_jobs : int, default=-1
#         Number of parallel jobs to run (-1 means using all processors).

#     num_threads_per_generate : int, default=1
#         Number of threads per job for dataset generation

#     device : str, default="cpu"
#         Computation device ('cpu' or 'cuda')
#     """

#     def __init__(
#         self,
#         batch_size: int = 256,
#         batch_size_per_gp: int = 4,
#         batch_size_per_subgp: Optional[int] = None,
#         min_features: int = 2,
#         max_features: int = 100,
#         max_classes: int = 10,
#         min_seq_len: Optional[int] = None,
#         max_seq_len: int = 1024,
#         log_seq_len: bool = False,
#         seq_len_per_gp: bool = False,
#         min_train_size: Union[int, float] = 0.1,
#         max_train_size: Union[int, float] = 0.9,
#         replay_small: bool = False,
#         prior_type: str = "mlp_scm",
#         fixed_hp: Dict[str, Any] = DEFAULT_FIXED_HP,
#         sampled_hp: Dict[str, Any] = DEFAULT_SAMPLED_HP,
#         n_jobs: int = -1,
#         num_threads_per_generate: int = 1,
#         device: str = "cpu",
#     ):
#         super().__init__(
#             batch_size=batch_size,
#             min_features=min_features,
#             max_features=max_features,
#             max_classes=max_classes,
#             min_seq_len=min_seq_len,
#             max_seq_len=max_seq_len,
#             log_seq_len=log_seq_len,
#             min_train_size=min_train_size,
#             max_train_size=max_train_size,
#             replay_small=replay_small,
#         )

#         self.batch_size_per_gp = batch_size_per_gp
#         self.batch_size_per_subgp = batch_size_per_subgp or batch_size_per_gp
#         self.seq_len_per_gp = seq_len_per_gp
#         self.prior_type = prior_type
#         self.fixed_hp = fixed_hp
#         self.sampled_hp = sampled_hp
#         self.n_jobs = n_jobs
#         self.num_threads_per_generate = num_threads_per_generate
#         self.device = device
#         # Use a single num_features per batch by default to stabilize d
#         self.fixed_num_features_per_batch = True

#     def hp_sampling(self) -> Dict[str, Any]:
#         """
#         Sample hyperparameters for dataset generation.

#         Returns
#         -------
#         dict
#             Dictionary with sampled hyperparameters merged with fixed ones
#         """
#         hp_sampler = HpSamplerList(self.sampled_hp, device=self.device)
#         return hp_sampler.sample()

#     @torch.no_grad()
#     def generate_dataset(self, params: Dict[str, Any]) -> Tuple[Tensor, Tensor, Tensor]:
#         """
#         Generates a single valid dataset based on the provided parameters.

#         Parameters
#         ----------
#         params : dict
#             Hyperparameters for generating this specific dataset, including seq_len,
#             train_size, num_features, num_classes, prior_type, device, etc.

#         Returns
#         -------
#         tuple
#             (X, y, d) where:
#             - X: Features tensor of shape (seq_len, max_features)
#             - y: Labels tensor of shape (seq_len,)
#             - d: Number of active features (after filtering if enabled) (scalar Tensor)
#         """

#         if params["prior_type"] == "mlp_scm":
#             prior_cls = MLPSCM
#         elif params["prior_type"] == "tree_scm":
#             prior_cls = TreeSCM
#         else:
#             raise ValueError(f"Unknown prior type {params['prior_type']}")

#         while True:
#             X, y = prior_cls(**params)()
#             X, y = Reg2Cls(params)(X, y)

#             # Add batch dim for single dataset to be compatible with delete_unique_features and sanity_check
#             X, y = X.unsqueeze(0), y.unsqueeze(0)
#             d = torch.tensor([params["num_features"]], device=self.device, dtype=torch.long)

#             # Only keep valid datasets with sufficient features and balanced classes
#             if not self.fixed_num_features_per_batch:
#                 X, d = self.delete_unique_features(X, d)
#             if (d > 0).all() and self.sanity_check(X, y, params["train_size"]):
#                 return X.squeeze(0), y.squeeze(0), d.squeeze(0)

#     @torch.no_grad()
#     def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
#         """
#         Generates a batch of datasets by first creating a parameter list and then processing it.

#         Parameters
#         ----------
#         batch_size : int, optional
#             Batch size override. If None, uses self.batch_size

#         Returns
#         -------
#         X : Tensor or NestedTensor
#             Features tensor. If seq_len_per_gp=False, shape is (batch_size, seq_len, max_features).
#             If seq_len_per_gp=True, returns a NestedTensor.

#         y : Tensor or NestedTensor
#             Labels tensor. If seq_len_per_gp=False, shape is (batch_size, seq_len).
#             If seq_len_per_gp=True, returns a NestedTensor.

#         d : Tensor
#             Effective number of features per dataset (after filtering if enabled), shape (batch_size,)

#         seq_lens : Tensor
#             Sequence length for each dataset, shape (batch_size,)

#         train_sizes : Tensor
#             Position for train/test split for each dataset, shape (batch_size,)
#         """
#         batch_size = batch_size or self.batch_size

#         # Calculate number of groups and subgroups
#         size_per_gp = min(self.batch_size_per_gp, batch_size)
#         num_gps = math.ceil(batch_size / size_per_gp)

#         size_per_subgp = min(self.batch_size_per_subgp, size_per_gp)

#         # Generate parameters list for all datasets, preserving group and subgroup structure
#         param_list = []
#         global_seq_len = None
#         global_train_size = None

#         # Determine global seq_len/train_size if not per-group
#         if not self.seq_len_per_gp:
#             global_seq_len = self.sample_seq_len(
#                 self.min_seq_len, self.max_seq_len, log=self.log_seq_len, replay_small=self.replay_small
#             )
#             global_train_size = self.sample_train_size(self.min_train_size, self.max_train_size, global_seq_len)

#         # Sample a single num_features for the whole batch when seq_len is fixed
#         batch_num_features = None
#         if not self.seq_len_per_gp:
#             batch_num_features = round(np.random.uniform(self.min_features, self.max_features))

#         # Generate parameters for each group
#         for gp_idx in range(num_gps):
#             # Determine actual size for this group (may be smaller for the last group)
#             actual_gp_size = min(size_per_gp, batch_size - gp_idx * size_per_gp)
#             if actual_gp_size <= 0:
#                 break

#             group_sampled_hp = self.hp_sampling()
#             # If per-group, sample seq_len and train_size for this group. Otherwise, use global ones
#             if self.seq_len_per_gp:
#                 gp_seq_len = self.sample_seq_len(
#                     self.min_seq_len, self.max_seq_len, log=self.log_seq_len, replay_small=self.replay_small
#                 )
#                 gp_train_size = self.sample_train_size(self.min_train_size, self.max_train_size, gp_seq_len)
#                 # Adjust max features based on seq_len for this group
#                 gp_max_features = self.adjust_max_features(gp_seq_len, self.max_features)
#             else:
#                 gp_seq_len = global_seq_len
#                 gp_train_size = global_train_size
#                 gp_max_features = self.max_features

#             # Choose a single num_features for this group
#             if self.seq_len_per_gp:
#                 gp_num_features = round(np.random.uniform(self.min_features, gp_max_features))
#             else:
#                 gp_num_features = batch_num_features

#             # Calculate number of subgroups for this group
#             num_subgps_in_gp = math.ceil(actual_gp_size / size_per_subgp)

#             # Generate parameters for each subgroup
#             for subgp_idx in range(num_subgps_in_gp):
#                 # Determine actual size for this subgroup
#                 actual_subgp_size = min(size_per_subgp, actual_gp_size - subgp_idx * size_per_subgp)
#                 if actual_subgp_size <= 0:
#                     break

#                 # Subgroups share prior type, number of features, and sampled HPs
#                 subgp_prior_type = self.get_prior()
#                 subgp_num_features = gp_num_features
#                 subgp_sampled_hp = {k: v() if callable(v) else v for k, v in group_sampled_hp.items()}

#                 # Generate parameters for each dataset in this subgroup
#                 for ds_idx in range(actual_subgp_size):
#                     # Each dataset has its own number of classes
#                     if np.random.random() > 0.5:
#                         ds_num_classes = np.random.randint(2, self.max_classes + 1)
#                     else:
#                         ds_num_classes = 2

#                     # Create parameters dictionary for this dataset
#                     params = {
#                         **self.fixed_hp,  # Fixed HPs
#                         "seq_len": gp_seq_len,
#                         "train_size": gp_train_size,
#                         # If per-gp setting, use adjusted max features for this group because we use nested tensors
#                         # If not per-gp setting, use global max features to fix size for concatenation
#                         "max_features": gp_max_features if self.seq_len_per_gp else self.max_features,
#                         **subgp_sampled_hp,  # sampled HPs for this group
#                         "prior_type": subgp_prior_type,
#                         "num_features": subgp_num_features,
#                         "num_classes": ds_num_classes,
#                         "device": self.device,
#                     }
#                     param_list.append(params)

#         # Use joblib to generate datasets in parallel.
#         # Note: the 'loky' backend does not support nested parallelism during DDP, whereas the 'threading' backend does.
#         # However, 'threading' does not respect `inner_max_num_threads`.
#         # Therefore, we stick with the 'loky' backend for parallelism, but this requires generating
#         # the prior datasets separately from the training process and loading them from disk,
#         # rather than generating them on-the-fly.
#         if self.n_jobs > 1 and self.device == "cpu":
#             with joblib.parallel_config(
#                 n_jobs=self.n_jobs, backend="loky", inner_max_num_threads=self.num_threads_per_generate
#             ):
#                 results = joblib.Parallel()(joblib.delayed(self.generate_dataset)(params) for params in param_list)
#         else:
#             results = [self.generate_dataset(params) for params in param_list]

#         X_list, y_list, d_list = zip(*results)

#         # Combine Results
#         if self.seq_len_per_gp:
#             # Use nested tensors for variable sequence lengths
#             X = nested_tensor([x.to(self.device) for x in X_list], device=self.device)
#             y = nested_tensor([y.to(self.device) for y in y_list], device=self.device)
#         else:
#             # Stack into regular tensors for fixed sequence length
#             X = torch.stack(X_list).to(self.device)  # (B, T, H)
#             y = torch.stack(y_list).to(self.device)  # (B, T)

#         # Metadata (always regular tensors)
#         d = torch.stack(d_list).to(self.device)  # Effective number of features (after filtering if enabled)
#         seq_lens = torch.tensor([params["seq_len"] for params in param_list], device=self.device, dtype=torch.long)
#         train_sizes = torch.tensor(
#             [params["train_size"] for params in param_list], device=self.device, dtype=torch.long
#         )

#         return X, y, d, seq_lens, train_sizes

#     def get_prior(self) -> str:
#         """
#         Determine which prior type to use for generation.

#         For 'mix_scm' prior type, randomly selects between available priors
#         based on configured probabilities.

#         Returns
#         -------
#         str
#             The selected prior type name
#         """
#         if self.prior_type == "mix_scm":
#             return np.random.choice(["mlp_scm", "tree_scm"], p=self.fixed_hp.get("mix_probas", [0.7, 0.3]))
#         else:
#             return self.prior_type


# class SCMPrior2(SCMPrior):
#     """Variant of :class:`SCMPrior` that mirrors the modular generation pipeline
#     used by :class:`TemporalCausalPrior`.

#     This refactoring separates parameter construction from dataset materialisation
#     so that feature generation follows the same staged workflow as the temporal
#     causal prior while preserving the original SCM-based label generation.
#     """

#     def __init__(
#         self,
#         batch_size: int = 256,
#         batch_size_per_gp: int = 4,
#         batch_size_per_subgp: Optional[int] = None,
#         min_features: int = 2,
#         max_features: int = 100,
#         max_classes: int = 10,
#         min_seq_len: Optional[int] = None,
#         max_seq_len: int = 1024,
#         log_seq_len: bool = False,
#         seq_len_per_gp: bool = False,
#         min_train_size: Union[int, float] = 0.1,
#         max_train_size: Union[int, float] = 0.9,
#         replay_small: bool = False,
#         prior_type: str = "mlp_scm",
#         fixed_hp: Dict[str, Any] = DEFAULT_FIXED_HP,
#         sampled_hp: Dict[str, Any] = DEFAULT_SAMPLED_HP,
#         n_jobs: int = -1,
#         num_threads_per_generate: int = 1,
#         device: str = "cpu",
#     ):
#         from .temporal_prior import TemporalCausalPrior, TemporalPriorParams

#         super().__init__(
#             batch_size=batch_size,
#             batch_size_per_gp=batch_size_per_gp,
#             batch_size_per_subgp=batch_size_per_subgp,
#             min_features=min_features,
#             max_features=max_features,
#             max_classes=max_classes,
#             min_seq_len=min_seq_len,
#             max_seq_len=max_seq_len,
#             log_seq_len=log_seq_len,
#             seq_len_per_gp=seq_len_per_gp,
#             min_train_size=min_train_size,
#             max_train_size=max_train_size,
#             replay_small=replay_small,
#             prior_type=prior_type,
#             fixed_hp=fixed_hp,
#             sampled_hp=sampled_hp,
#             n_jobs=n_jobs,
#             num_threads_per_generate=num_threads_per_generate,
#             device=device,
#         )

#         self.temporal_edge_prob = 0.3
#         self.temporal_noise_scale = 0.05
#         self.temporal_nonlinearity = "leaky_relu"
#         self.temporal_neg_slope = 0.2
#         self._TemporalPriorParams = TemporalPriorParams
#         self._temporal_prior = TemporalCausalPrior(
#             batch_size=1,
#             batch_size_per_gp=1,
#             min_features=min_features,
#             max_features=max_features,
#             min_seq_len=min_seq_len,
#             max_seq_len=max_seq_len,
#             log_seq_len=log_seq_len,
#             seq_len_per_gp=seq_len_per_gp,
#             min_train_size=min_train_size,
#             max_train_size=max_train_size,
#             edge_prob=self.temporal_edge_prob,
#             min_noise_scale=self.temporal_noise_scale,
#             max_noise_scale=self.temporal_noise_scale,
#             nonlinearity=self.temporal_nonlinearity,
#             neg_slope=self.temporal_neg_slope,
#             device=device,
#         )

#     def _build_param_list(self, batch_size: int) -> List[Dict[str, Any]]:
#         """Construct the list of per-dataset hyper-parameters."""

#         size_per_gp = min(self.batch_size_per_gp, batch_size)
#         num_gps = math.ceil(batch_size / size_per_gp)
#         size_per_subgp = min(self.batch_size_per_subgp, size_per_gp)

#         param_list: List[Dict[str, Any]] = []
#         global_seq_len = None
#         global_train_size = None

#         if not self.seq_len_per_gp:
#             global_seq_len = self.sample_seq_len(
#                 self.min_seq_len, self.max_seq_len, log=self.log_seq_len, replay_small=self.replay_small
#             )
#             global_train_size = self.sample_train_size(self.min_train_size, self.max_train_size, global_seq_len)

#         for gp_idx in range(num_gps):
#             actual_gp_size = min(size_per_gp, batch_size - gp_idx * size_per_gp)
#             if actual_gp_size <= 0:
#                 break

#             group_sampled_hp = self.hp_sampling()
#             if self.seq_len_per_gp:
#                 gp_seq_len = self.sample_seq_len(
#                     self.min_seq_len, self.max_seq_len, log=self.log_seq_len, replay_small=self.replay_small
#                 )
#                 gp_train_size = self.sample_train_size(self.min_train_size, self.max_train_size, gp_seq_len)
#                 gp_max_features = self.adjust_max_features(gp_seq_len, self.max_features)
#             else:
#                 gp_seq_len = global_seq_len
#                 gp_train_size = global_train_size
#                 gp_max_features = self.max_features

#             num_subgps_in_gp = math.ceil(actual_gp_size / size_per_subgp)
#             for subgp_idx in range(num_subgps_in_gp):
#                 actual_subgp_size = min(size_per_subgp, actual_gp_size - subgp_idx * size_per_subgp)
#                 if actual_subgp_size <= 0:
#                     break

#                 subgp_prior_type = self.get_prior()
#                 subgp_num_features = round(np.random.uniform(self.min_features, gp_max_features))
#                 subgp_sampled_hp = {k: v() if callable(v) else v for k, v in group_sampled_hp.items()}

#                 for _ in range(actual_subgp_size):
#                     if np.random.random() > 0.5:
#                         ds_num_classes = np.random.randint(2, self.max_classes + 1)
#                     else:
#                         ds_num_classes = 2

#                     params = {
#                         **self.fixed_hp,
#                         "seq_len": gp_seq_len,
#                         "train_size": gp_train_size,
#                         "max_features": gp_max_features if self.seq_len_per_gp else self.max_features,
#                         **subgp_sampled_hp,
#                         "prior_type": subgp_prior_type,
#                         "num_features": subgp_num_features,
#                         "num_classes": ds_num_classes,
#                         "device": self.device,
#                     }
#                     param_list.append(params)

#         return param_list[:batch_size]

#     def _generate_dataset(self, params: Dict[str, Any]) -> Tuple[Tensor, Tensor, Tensor]:
#         """Generate a single dataset mirroring TemporalCausalPrior's two-step process."""

#         noise_scale = params.get("noise_scale", self.temporal_noise_scale)
#         temporal_params = self._TemporalPriorParams(
#             seq_len=params["seq_len"],
#             train_size=params["train_size"],
#             num_features=params["num_features"],
#             noise_scale=noise_scale,
#             device=torch.device(params["device"]),
#         )
#         inputs, _, feature_count, _, _ = self._temporal_prior._generate_sequence(temporal_params)

#         X, y, _ = SCMPrior.generate_dataset(self, params)
#         y = y.to(inputs.device)

#         return inputs.to(inputs.device), y, feature_count

#     @torch.no_grad()
#     def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
#         """Generate a batch following the TemporalCausalPrior-style workflow."""
#         batch_size = batch_size or self.batch_size
#         param_list = self._build_param_list(batch_size)

#         results = [self._generate_dataset(params) for params in param_list]
#         X_list, y_list, d_list = zip(*results)

#         if self.seq_len_per_gp:
#             X = nested_tensor([x.to(self.device) for x in X_list], device=self.device)
#             y = nested_tensor([y.to(self.device) for y in y_list], device=self.device)
#         else:
#             X = torch.stack(X_list).to(self.device)
#             y = torch.stack(y_list).to(self.device)

#         d = torch.stack(d_list).to(self.device)
#         seq_lens = torch.tensor([params["seq_len"] for params in param_list], device=self.device, dtype=torch.long)
#         train_sizes = torch.tensor(
#             [params["train_size"] for params in param_list], device=self.device, dtype=torch.long
#         )

#         return X, y, d, seq_lens, train_sizes


# class DummyPrior(Prior):
#     """This class creates purely random data. This is useful for testing and debugging
#     without the computational overhead of SCM-based generation.

#     Parameters
#     ----------
#     batch_size : int, default=256
#         Number of datasets to generate

#     min_features : int, default=2
#         Minimum number of features per dataset

#     max_features : int, default=100
#         Maximum number of features per dataset

#     max_classes : int, default=10
#         Maximum number of target classes

#     min_seq_len : int, default=None
#         Minimum samples per dataset. If None, uses max_seq_len directly.

#     max_seq_len : int, default=1024
#         Maximum samples per dataset

#     log_seq_len : bool, default=False
#         If True, sample sequence length from a log-uniform distribution

#     min_train_size : int|float, default=0.1
#         Position or ratio for train/test split start. If int, absolute position.
#         If float between 0 and 1, specifies a fraction of sequence length.

#     max_train_size : int|float, default=0.9
#         Position or ratio for train/test split end. If int, absolute position.
#         If float between 0 and 1, specifies a fraction of sequence length.

#     device : str, default="cpu"
#         Computation device
#     """

#     def __init__(
#         self,
#         batch_size: int = 256,
#         min_features: int = 2,
#         max_features: int = 100,
#         max_classes: int = 10,
#         min_seq_len: Optional[int] = None,
#         max_seq_len: int = 1024,
#         log_seq_len: bool = False,
#         min_train_size: Union[int, float] = 0.1,
#         max_train_size: Union[int, float] = 0.9,
#         device: str = "cpu",
#     ):
#         super().__init__(
#             batch_size=batch_size,
#             min_features=min_features,
#             max_features=max_features,
#             max_classes=max_classes,
#             min_seq_len=min_seq_len,
#             max_seq_len=max_seq_len,
#             log_seq_len=log_seq_len,
#             min_train_size=min_train_size,
#             max_train_size=max_train_size,
#         )
#         self.device = device

#     @torch.no_grad()
#     def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
#         """
#         Generates a batch of random datasets for testing purposes.

#         Parameters
#         ----------
#         batch_size : int, optional
#             Batch size override, if None, uses self.batch_size

#         Returns
#         -------
#         X : Tensor
#             Features tensor of shape (batch_size, seq_len, max_features).
#             Contains random Gaussian values for all features.

#         y : Tensor
#             Labels tensor of shape (batch_size, seq_len).
#             Contains randomly assigned class labels.

#         d : Tensor
#             Number of features per dataset of shape (batch_size,).
#             Always set to max_features for DummyPrior.

#         seq_lens : Tensor
#             Sequence length for each dataset of shape (batch_size,).
#             All datasets share the same sequence length.

#         train_sizes : Tensor
#             Position for train/test split for each dataset of shape (batch_size,).
#             All datasets share the same split position.
#         """

#         batch_size = batch_size or self.batch_size
#         seq_len = self.sample_seq_len(self.min_seq_len, self.max_seq_len, log=self.log_seq_len)
#         train_size = self.sample_train_size(self.min_train_size, self.max_train_size, seq_len)

#         X = torch.randn(batch_size, seq_len, self.max_features, device=self.device)

#         num_classes = np.random.randint(2, self.max_classes + 1)
#         y = torch.randint(0, num_classes, (batch_size, seq_len), device=self.device)

#         d = torch.full((batch_size,), self.max_features, device=self.device)
#         seq_lens = torch.full((batch_size,), seq_len, device=self.device)
#         train_sizes = torch.full((batch_size,), train_size, device=self.device)

#         return X, y, d, seq_lens, train_sizes


# class PriorDataset(IterableDataset):
#     """
#     Main dataset class that provides an infinite iterator over synthetic tabular datasets.

#     Parameters
#     ----------
#     batch_size : int, default=256
#         Total number of datasets to generate per batch

#     batch_size_per_gp : int, default=4
#         Number of datasets per group, sharing similar characteristics

#     batch_size_per_subgp : int, default=None
#         Number of datasets per subgroup, with more similar causal structures
#         If None, defaults to batch_size_per_gp

#     min_features : int, default=2
#         Minimum number of features per dataset

#     max_features : int, default=100
#         Maximum number of features per dataset

#     max_classes : int, default=10
#         Maximum number of target classes

#     min_seq_len : int, default=None
#         Minimum samples per dataset. If None, uses max_seq_len directly.

#     max_seq_len : int, default=1024
#         Maximum samples per dataset

#     log_seq_len : bool, default=False
#         If True, sample sequence length from a log-uniform distribution

#     seq_len_per_gp : bool = False
#         If True, sample sequence length per group, allowing variable-sized datasets

#     min_train_size : int|float, default=0.1
#         Position or ratio for train/test split start. If int, absolute position.
#         If float between 0 and 1, specifies a fraction of sequence length.

#     max_train_size : int|float, default=0.9
#         Position or ratio for train/test split end. If int, absolute position.
#         If float between 0 and 1, specifies a fraction of sequence length.

#     replay_small : bool, default=False
#         If True, occasionally sample smaller sequence lengths with
#         specific distributions to ensure model robustness on smaller datasets

#     prior_type : str, default="mlp_scm"
#         Type of prior: 'mlp_scm' (default), 'tree_scm', 'mix_scm', or 'dummy'

#         1. SCM-based: Structural causal models with complex feature relationships
#          - 'mlp_scm': MLP-based causal models
#          - 'tree_scm': Tree-based causal models
#          - 'mix_scm': Probabilistic mix of the above models

#         2. Dummy: Randomly generated datasets for debugging

#     scm_fixed_hp : dict, default=DEFAULT_FIXED_HP
#         Fixed parameters for SCM-based priors

#     scm_sampled_hp : dict, default=DEFAULT_SAMPLED_HP
#         Parameters sampled during generation

#     n_jobs : int, default=-1
#         Number of parallel jobs to run (-1 means using all processors)

#     num_threads_per_generate : int, default=1
#         Number of threads per job for dataset generation

#     device : str, default="cpu"
#         Computation device ('cpu' or 'cuda')

#     bucket_by_d : bool, default=False
#         If True, bucket samples by their effective feature count `d` and only
#         return batches drawn from a single bucket.

#     d_bucket_size : int, default=1
#         Bucket width for `d`. If 1, each distinct `d` value is its own bucket.
#         Larger values create wider buckets to reduce buffering.

#     bucket_max_buffer : int, default=0
#         Maximum number of samples to keep per bucket. If 0, no cap is applied.
#     """

#     def __init__(
#         self,
#         batch_size: int = 256,
#         batch_size_per_gp: int = 4,
#         batch_size_per_subgp: Optional[int] = None,
#         min_features: int = 2,
#         max_features: int = 100,
#         max_classes: int = 10,
#         min_seq_len: Optional[int] = None,
#         max_seq_len: int = 1024,
#         log_seq_len: bool = False,
#         seq_len_per_gp: bool = False,
#         min_train_size: Union[int, float] = 0.1,
#         max_train_size: Union[int, float] = 0.9,
#         replay_small: bool = False,
#         prior_type: str = "mlp_scm",
#         scm_fixed_hp: Dict[str, Any] = DEFAULT_FIXED_HP,
#         scm_sampled_hp: Dict[str, Any] = DEFAULT_SAMPLED_HP,
#         n_jobs: int = -1,
#         num_threads_per_generate: int = 1,
#         device: str = "cpu",
#         bucket_by_d: bool = False,
#         d_bucket_size: int = 1,
#         bucket_max_buffer: int = 0,
#     ):
#         super().__init__()
#         if prior_type == "dummy":
#             self.prior = DummyPrior(
#                 batch_size=batch_size,
#                 min_features=min_features,
#                 max_features=max_features,
#                 max_classes=max_classes,
#                 min_seq_len=min_seq_len,
#                 max_seq_len=max_seq_len,
#                 log_seq_len=log_seq_len,
#                 min_train_size=min_train_size,
#                 max_train_size=max_train_size,
#                 device=device,
#             )
#         elif prior_type in ["mlp_scm", "tree_scm", "mix_scm"]:
#             self.prior = SCMPrior(
#                 batch_size=batch_size,
#                 batch_size_per_gp=batch_size_per_gp,
#                 batch_size_per_subgp=batch_size_per_subgp,
#                 min_features=min_features,
#                 max_features=max_features,
#                 max_classes=max_classes,
#                 min_seq_len=min_seq_len,
#                 max_seq_len=max_seq_len,
#                 log_seq_len=log_seq_len,
#                 seq_len_per_gp=seq_len_per_gp,
#                 min_train_size=min_train_size,
#                 max_train_size=max_train_size,
#                 replay_small=replay_small,
#                 prior_type=prior_type,
#                 fixed_hp=scm_fixed_hp,
#                 sampled_hp=scm_sampled_hp,
#                 n_jobs=n_jobs,
#                 num_threads_per_generate=num_threads_per_generate,
#                 device=device,
#             )
#         else:
#             raise ValueError(
#                 f"Unknown prior type '{prior_type}'. Available options: 'mlp_scm', 'tree_scm', 'mix_scm', or 'dummy'."
#             )

#         self.batch_size = batch_size
#         self.batch_size_per_gp = batch_size_per_gp
#         self.batch_size_per_subgp = batch_size_per_subgp or batch_size_per_gp
#         self.min_features = min_features
#         self.max_features = max_features
#         self.max_classes = max_classes
#         self.min_seq_len = min_seq_len
#         self.max_seq_len = max_seq_len
#         self.log_seq_len = log_seq_len
#         self.seq_len_per_gp = seq_len_per_gp
#         self.min_train_size = min_train_size
#         self.max_train_size = max_train_size
#         self.device = device
#         self.prior_type = prior_type
#         self.bucket_by_d = bucket_by_d
#         self.d_bucket_size = max(int(d_bucket_size), 1)
#         self.bucket_max_buffer = max(int(bucket_max_buffer), 0)
#         self._bucket_buffers: Dict[int, List[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]] = defaultdict(list)

#     def _bucket_id(self, d_val: int, seq_len: int, train_size: int) -> tuple[int, int, int]:
#         if self.d_bucket_size <= 1:
#             d_bucket = d_val
#         else:
#             d_bucket = d_val // self.d_bucket_size
#         return (seq_len, train_size, d_bucket)

#     def _split_batch(
#         self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
#     ) -> List[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
#         X, y, d, seq_lens, train_sizes = batch
#         batch_size = d.shape[0]
#         samples = []
#         for i in range(batch_size):
#             samples.append((X[i], y[i], d[i], seq_lens[i], train_sizes[i]))
#         return samples

#     def _collate_samples(
#         self, samples: List[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]
#     ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
#         xs, ys, ds, seqs, trains = zip(*samples)
#         if self.seq_len_per_gp:
#             X = nested_tensor(list(xs), device=self.device)
#             y = nested_tensor(list(ys), device=self.device)
#         else:
#             X = torch.stack(xs).to(self.device)
#             y = torch.stack(ys).to(self.device)
#         d = torch.stack(ds).to(self.device)
#         seq_lens = torch.stack(seqs).to(self.device)
#         train_sizes = torch.stack(trains).to(self.device)
#         return X, y, d, seq_lens, train_sizes

#     def _get_bucketed_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
#         target_bs = batch_size or self.batch_size
#         max_buffer = self.bucket_max_buffer if self.bucket_max_buffer >= target_bs else 0

#         while True:
#             # Check for any bucket ready to emit
#             ready_bucket = None
#             ready_size = 0
#             for bucket_id, buf in self._bucket_buffers.items():
#                 if len(buf) >= target_bs and len(buf) > ready_size:
#                     ready_bucket = bucket_id
#                     ready_size = len(buf)

#             if ready_bucket is not None:
#                 buf = self._bucket_buffers[ready_bucket]
#                 samples = buf[:target_bs]
#                 del buf[:target_bs]
#                 return self._collate_samples(samples)

#             # Otherwise, generate more data and fill buckets
#             batch = self.prior.get_batch(target_bs)
#             for sample in self._split_batch(batch):
#                 d_val = int(sample[2].item())
#                 seq_len = int(sample[3].item())
#                 train_size = int(sample[4].item())
#                 bucket_id = self._bucket_id(d_val, seq_len, train_size)
#                 buf = self._bucket_buffers[bucket_id]
#                 buf.append(sample)
#                 if max_buffer and len(buf) > max_buffer:
#                     # Drop oldest to cap memory usage
#                     buf.pop(0)

#     def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
#         """
#         Generate a new batch of datasets.

#         Parameters
#         ----------
#         batch_size : int, optional
#             If provided, overrides the default batch size for this call

#         Returns
#         -------
#         X : Tensor or NestedTensor
#             1. For SCM-based priors:
#              - If seq_len_per_gp=False, shape is (batch_size, seq_len, max_features).
#              - If seq_len_per_gp=True, returns a NestedTensor.

#             2. For DummyPrior, random Gaussian values of (batch_size, seq_len, max_features).

#         X : Tensor or NestedTensor
#             1. For SCM-based priors:
#              - If seq_len_per_gp=False, shape is (batch_size, seq_len).
#              - If seq_len_per_gp=True, returns a NestedTensor.

#             2. For DummyPrior, random class labels of (batch_size, seq_len).

#         d : Tensor
#             Number of active features per dataset of shape (batch_size,).

#         seq_lens : Tensor
#             Sequence length for each dataset of shape (batch_size,).

#         train_sizes : Tensor
#             Position for train/test split for each dataset of shape (batch_size,).
#         """
#         if self.bucket_by_d:
#             return self._get_bucketed_batch(batch_size)
#         return self.prior.get_batch(batch_size)

#     def __iter__(self) -> "PriorDataset":
#         """
#         Returns an iterator that yields batches indefinitely.

#         Returns
#         -------
#         self
#             Returns self as an iterator
#         """
#         return self

#     def __next__(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
#         """
#         Returns the next batch from the iterator. Since this is an infinite
#         iterator, it never raises StopIteration and instead continuously generates
#         new synthetic data batches.
#         """
#         with DisablePrinting():
#             return self.get_batch()

#     def __repr__(self) -> str:
#         """
#         Returns a string representation of the dataset.

#         Provides a detailed view of the dataset configuration for debugging
#         and logging purposes.

#         Returns
#         -------
#         str
#             A formatted string with dataset parameters
#         """
#         return (
#             f"PriorDataset(\n"
#             f"  prior_type: {self.prior_type}\n"
#             f"  batch_size: {self.batch_size}\n"
#             f"  batch_size_per_gp: {self.batch_size_per_gp}\n"
#             f"  features: {self.min_features} - {self.max_features}\n"
#             f"  max classes: {self.max_classes}\n"
#             f"  seq_len: {self.min_seq_len or 'None'} - {self.max_seq_len}\n"
#             f"  sequence length varies across groups: {self.seq_len_per_gp}\n"
#             f"  train_size: {self.min_train_size} - {self.max_train_size}\n"
#             f"  bucket_by_d: {self.bucket_by_d}\n"
#             f"  d_bucket_size: {self.d_bucket_size}\n"
#             f"  bucket_max_buffer: {self.bucket_max_buffer}\n"
#             f"  device: {self.device}\n"
#             f")"
#         )


# class DisablePrinting:
#     """Context manager to temporarily suppress printed output."""

#     def __enter__(self):
#         self.original_stdout = sys.stdout
#         sys.stdout = open(os.devnull, "w")

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         sys.stdout.close()
#         sys.stdout = self.original_stdout
