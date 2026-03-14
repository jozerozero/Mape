from __future__ import annotations

import random
import warnings
from typing import Any, Dict, List, Set, Tuple

import torch
from torch import nn

from .utils import XSampler


class MLPSCM(nn.Module):
    """DAG-based structural causal model.

    This implementation applies sparsity directly on the DAG edges instead of
    post-hoc feature row permutation.
    """

    def __init__(
        self,
        seq_len: int = 1024,
        num_features: int = 100,
        num_outputs: int = 1,
        is_causal: bool = True,
        num_causes: int = 10,
        y_is_effect: bool = True,
        in_clique: bool = False,
        sort_features: bool = True,
        num_layers: int = 10,
        hidden_dim: int = 20,
        mlp_activations: Any = nn.Tanh,
        init_std: float = 0.1,
        block_wise_dropout: bool = True,
        mlp_dropout_prob: float = 0.1,
        scale_init_std_by_dropout: bool = True,
        sampling: str = "normal",
        pre_sample_cause_stats: bool = False,
        noise_std: float = 0.01,
        pre_sample_noise_std: bool = False,
        device: str = "cpu",
        num_nodes: int | None = None,
        edge_prob: float = 0.3,
        edge_drop_prob: float = 0.0,
        mb_hops: int = 1,
        **kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.is_causal = is_causal
        self.num_causes = num_causes
        self.y_is_effect = y_is_effect
        self.in_clique = in_clique
        self.sort_features = sort_features
        self.init_std = init_std
        self.noise_std = noise_std
        self.pre_sample_noise_std = pre_sample_noise_std
        self.sampling = sampling
        self.pre_sample_cause_stats = pre_sample_cause_stats
        self.device = device
        self.mb_hops = max(0, int(mb_hops))

        # keep compatibility with historical args that are not used by DAG mode
        _ = (num_layers, hidden_dim, mlp_activations, block_wise_dropout, mlp_dropout_prob, scale_init_std_by_dropout)

        graph_sparsity = float(kwargs.get("graph_sparsity", 0.0))
        if graph_sparsity > 0.0:
            warnings.warn(
                "graph_sparsity row-permutation is deprecated in DAG mode and will be ignored.",
                stacklevel=2,
            )

        if not self.is_causal:
            self.num_causes = self.num_features

        if num_nodes is None:
            num_nodes = max(self.num_causes + 2 * self.num_features + self.num_outputs, self.num_causes + 8)
        if num_nodes < self.num_causes + self.num_features + self.num_outputs:
            raise ValueError(
                "num_nodes must be >= num_causes + num_features + num_outputs "
                f"({self.num_causes + self.num_features + self.num_outputs}), got {num_nodes}"
            )
        self.num_nodes = int(num_nodes)

        self.edge_prob = float(edge_prob)
        self.edge_drop_prob = float(edge_drop_prob)
        if not (0.0 <= self.edge_prob <= 1.0):
            raise ValueError(f"edge_prob must be in [0, 1], got {self.edge_prob}")
        if not (0.0 <= self.edge_drop_prob <= 1.0):
            raise ValueError(f"edge_drop_prob must be in [0, 1], got {self.edge_drop_prob}")

        self.xsampler = XSampler(
            self.seq_len,
            self.num_causes,
            pre_stats=self.pre_sample_cause_stats,
            sampling=self.sampling,
            device=self.device,
        )

        self.parents, self.children = self._build_sparse_dag(
            num_nodes=self.num_nodes,
            num_roots=self.num_causes,
            edge_prob=self.edge_prob,
            edge_drop_prob=self.edge_drop_prob,
        )
        self.topo_order = list(range(self.num_nodes))
        self.activations = self._sample_node_activations()
        self.weights = self._init_edge_weights()
        self.last_x_binary_roles: torch.Tensor | None = None

        if self.pre_sample_noise_std:
            self.node_noise_std = {
                i: float(abs(torch.normal(mean=torch.tensor(0.0), std=float(self.noise_std)).item()))
                for i in range(self.num_nodes)
            }
        else:
            self.node_noise_std = {i: float(self.noise_std) for i in range(self.num_nodes)}

    def _build_sparse_dag(
        self,
        num_nodes: int,
        num_roots: int,
        edge_prob: float,
        edge_drop_prob: float,
    ) -> Tuple[List[Set[int]], List[Set[int]]]:
        """Build an acyclic graph then prune edges and repair non-root parent constraints."""
        parents: List[Set[int]] = [set() for _ in range(num_nodes)]
        children: List[Set[int]] = [set() for _ in range(num_nodes)]

        # Step 1: sample edges with i < j to guarantee DAG.
        for child in range(num_roots, num_nodes):
            for parent in range(child):
                if random.random() < edge_prob:
                    parents[child].add(parent)
                    children[parent].add(child)

        # Step 2: ensure every non-root has at least one parent.
        for child in range(num_roots, num_nodes):
            if not parents[child]:
                parent = random.randint(0, child - 1)
                parents[child].add(parent)
                children[parent].add(child)

        # Step 3: edge drop for sparsification.
        edges = [(p, c) for c in range(num_roots, num_nodes) for p in list(parents[c])]
        for parent, child in edges:
            if random.random() < edge_drop_prob and parent in parents[child]:
                parents[child].remove(parent)
                children[parent].remove(child)

        # Step 4: re-repair parent constraints after dropping edges.
        for child in range(num_roots, num_nodes):
            if not parents[child]:
                parent = random.randint(0, child - 1)
                parents[child].add(parent)
                children[parent].add(child)

        return parents, children

    def _sample_node_activations(self) -> Dict[int, nn.Module]:
        choices = [nn.Identity(), nn.Tanh(), nn.ReLU(), nn.Sigmoid()]
        activations: Dict[int, nn.Module] = {}
        for node in range(self.num_nodes):
            if node < self.num_causes:
                activations[node] = nn.Identity()
            else:
                activations[node] = random.choice(choices)
        return activations

    def _init_edge_weights(self) -> Dict[Tuple[int, int], float]:
        weights: Dict[Tuple[int, int], float] = {}
        for child in range(self.num_causes, self.num_nodes):
            for parent in self.parents[child]:
                weights[(parent, child)] = float(torch.normal(mean=torch.tensor(0.0), std=float(self.init_std)).item())
        return weights

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        node_values: Dict[int, torch.Tensor] = {}

        causes = torch.clamp(self.xsampler.sample(), -5.0, 5.0)
        for i in range(self.num_causes):
            node_values[i] = causes[:, i : i + 1]

        for node in self.topo_order:
            if node < self.num_causes:
                continue

            parent_ids = self.parents[node]
            agg = torch.zeros(self.seq_len, 1, device=self.device)
            for parent in parent_ids:
                w = self.weights[(parent, node)]
                agg = agg + node_values[parent] * w

            eps = torch.normal(
                mean=torch.zeros(self.seq_len, 1, device=self.device),
                std=float(self.node_noise_std[node]),
            )
            z = agg + eps
            node_values[node] = self.activations[node](z)

        all_nodes = torch.cat([node_values[i] for i in range(self.num_nodes)], dim=1)
        X, y, x_binary_roles = self._select_X_y(all_nodes)
        self.last_x_binary_roles = x_binary_roles.to(dtype=torch.long, device=self.device)

        X = torch.nan_to_num(X, nan=0.0)
        y = torch.nan_to_num(y, nan=0.0)
        X = torch.clamp(X, -15.0, 15.0)

        if self.num_outputs == 1:
            y = y.squeeze(-1)

        return X, y

    def _select_X_y(self, all_nodes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate_nodes = [n for n in range(self.num_causes, self.num_nodes)]
        if not candidate_nodes:
            candidate_nodes = list(range(self.num_nodes))

        # y candidates: effects from sinks or broad random fallback
        if self.y_is_effect:
            sinks = [n for n in candidate_nodes if len(self.children[n]) == 0]
            y_candidates = sinks if len(sinks) >= self.num_outputs else candidate_nodes
        else:
            y_candidates = candidate_nodes

        if len(y_candidates) >= self.num_outputs:
            y_indices = random.sample(y_candidates, self.num_outputs)
        else:
            y_indices = random.sample(list(range(self.num_nodes)), self.num_outputs)

        preferred_pool = self._k_hop_neighbors(set(y_indices), self.mb_hops)
        preferred_pool = [n for n in preferred_pool if n not in y_indices and n >= self.num_causes]
        x_pool = [n for n in candidate_nodes if n not in y_indices]

        x_indices: List[int] = []
        if len(preferred_pool) >= self.num_features:
            if self.in_clique:
                topo_pref = [n for n in self.topo_order if n in preferred_pool]
                start = random.randint(0, len(topo_pref) - self.num_features)
                x_indices = topo_pref[start : start + self.num_features]
            else:
                x_indices = random.sample(preferred_pool, self.num_features)
        else:
            x_indices.extend(preferred_pool)
            remain = [n for n in x_pool if n not in x_indices]
            need = self.num_features - len(x_indices)
            if len(remain) >= need:
                x_indices.extend(random.sample(remain, need))
            else:
                x_indices.extend(remain)
                all_remain = [n for n in range(self.num_nodes) if n not in y_indices and n not in x_indices]
                if all_remain and len(x_indices) < self.num_features:
                    x_indices.extend(random.sample(all_remain, min(len(all_remain), self.num_features - len(x_indices))))

        if self.sort_features:
            x_indices = sorted(x_indices)

        # Build binary labels for X dimensions:
        # 1 -> in Y's Markov blanket neighborhood (parent/child/sibling), 0 -> other.
        mb_set = self._markov_blanket_union(y_indices)
        x_binary_roles = torch.tensor([1 if idx in mb_set else 0 for idx in x_indices], device=self.device)

        X = all_nodes[:, x_indices]
        y = all_nodes[:, y_indices]
        return X, y, x_binary_roles

    def _k_hop_neighbors(self, seeds: Set[int], hops: int) -> Set[int]:
        visited = set(seeds)
        frontier = set(seeds)
        for _ in range(hops):
            if not frontier:
                break
            nxt: Set[int] = set()
            for node in frontier:
                for parent in self.parents[node]:
                    if parent not in visited:
                        visited.add(parent)
                        nxt.add(parent)
                for child in self.children[node]:
                    if child not in visited:
                        visited.add(child)
                        nxt.add(child)
            frontier = nxt
        return visited

    def _markov_blanket_union(self, y_indices: List[int]) -> Set[int]:
        blanket: Set[int] = set()
        for y in y_indices:
            parents = set(self.parents[y])
            children = set(self.children[y])
            siblings: Set[int] = set()
            for c in children:
                siblings |= set(self.parents[c])
            siblings.discard(y)
            siblings -= parents
            siblings -= children
            blanket |= parents | children | siblings
        return blanket
