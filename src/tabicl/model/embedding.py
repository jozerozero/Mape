from __future__ import annotations

from collections import OrderedDict
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .encoders import SetTransformer
from .inference import InferenceManager
from .inference_config import InferenceConfig, MgrConfig
from .kv_cache import KVCache
from .layers import OneHotAndLinear, SkippableLinear


class ColEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        affine: bool = True,
        feature_group: Union[bool, str] = False,
        feature_group_size: int = 3,
        target_aware: bool = False,
        max_classes: int = 10,
        reserve_cls_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.reserve_cls_tokens = reserve_cls_tokens
        self.affine = bool(affine and not feature_group)
        self.feature_group = feature_group
        self.feature_group_size = feature_group_size
        self.target_aware = target_aware
        self.max_classes = max_classes

        in_dim = feature_group_size if feature_group else 1
        self.in_linear = SkippableLinear(in_dim, embed_dim)
        self.tf_col = SetTransformer(
            num_blocks=num_blocks,
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_inds=num_inds,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
        )

        if affine:
            self.out_w = SkippableLinear(embed_dim, embed_dim)
            self.ln_w = nn.LayerNorm(embed_dim, bias=not bias_free_ln) if norm_first else nn.Identity()
            self.out_b = SkippableLinear(embed_dim, embed_dim)
            self.ln_b = nn.LayerNorm(embed_dim, bias=not bias_free_ln) if norm_first else nn.Identity()

        if target_aware:
            self.y_encoder = OneHotAndLinear(max_classes, embed_dim)

        self.inference_mgr = InferenceManager(enc_name="tf_col", out_dim=embed_dim)

    @staticmethod
    def map_feature_shuffle(reference_pattern: List[int], other_pattern: List[int]) -> List[int]:
        orig_to_other = {feature: idx for idx, feature in enumerate(other_pattern)}
        return [orig_to_other[feature] for feature in reference_pattern]

    def effective_feature_count(self, d: Tensor) -> Tensor:
        if not self.feature_group:
            return d
        mode = "same" if self.feature_group is True else self.feature_group
        if mode == "same" and len(d.unique()) == 1:
            return d
        return torch.div(d + self.feature_group_size - 1, self.feature_group_size, rounding_mode="floor")

    def _resolved_group_mode(self, d: Optional[Tensor] = None) -> Union[bool, str]:
        if not self.feature_group:
            return False
        if self.feature_group is True:
            if d is not None and len(d.unique()) != 1:
                return "valid"
            return "same"
        if self.feature_group == "same" and d is not None and len(d.unique()) != 1:
            return "valid"
        return self.feature_group

    def feature_grouping(self, x: Tensor, d: Optional[Tensor] = None) -> Tensor:
        if not self.feature_group:
            return x.unsqueeze(-1)

        mode = self._resolved_group_mode(d)
        bsz, seq_len, n_features = x.shape
        group_size = self.feature_group_size

        if mode == "same":
            idxs = torch.arange(n_features, dtype=torch.long, device=x.device)
            return torch.stack([x[:, :, (idxs + 2**i) % n_features] for i in range(group_size)], dim=-1)

        pad_cols = (group_size - n_features % group_size) % group_size
        if pad_cols > 0:
            x = F.pad(x, (0, pad_cols), value=0.0)
        return x.reshape(bsz, seq_len, -1, group_size)

    def _should_use_target_aware(self, y_train: Optional[Tensor]) -> bool:
        if not self.target_aware or y_train is None:
            return False
        if self.max_classes <= 0:
            return False
        num_classes = int(y_train.max().item()) + 1
        return num_classes <= self.max_classes

    def _compute_embeddings(
        self,
        features: Tensor,
        train_size: Optional[int] = None,
        y_train: Optional[Tensor] = None,
        embed_with_test: bool = False,
    ) -> Tensor:
        src = self.in_linear(features)
        if self._should_use_target_aware(y_train):
            y_emb = self.y_encoder(y_train.float())
            src[..., :train_size, :] = src[..., :train_size, :] + y_emb

        src = self.tf_col(src, None if embed_with_test else train_size)
        if not self.affine:
            return src

        weights = self.ln_w(self.out_w(src))
        biases = self.ln_b(self.out_b(src))
        return features * weights + biases

    def _compute_embeddings_with_cache(
        self,
        features: Tensor,
        col_cache: KVCache,
        train_size: Optional[int] = None,
        y_train: Optional[Tensor] = None,
        use_cache: bool = False,
        store_cache: bool = True,
    ) -> Tensor:
        src = self.in_linear(features)
        if store_cache and self._should_use_target_aware(y_train):
            y_emb = self.y_encoder(y_train.float())
            src[..., :train_size, :] = src[..., :train_size, :] + y_emb

        src = self.tf_col.forward_with_cache(
            src,
            col_cache=col_cache,
            train_size=train_size,
            use_cache=use_cache,
            store_cache=store_cache,
        )
        if not self.affine:
            return src

        weights = self.ln_w(self.out_w(src))
        biases = self.ln_b(self.out_b(src))
        return features * weights + biases

    def _prepare_features(
        self,
        x: Tensor,
        d: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor], int]:
        if not self.feature_group:
            if self.reserve_cls_tokens > 0:
                x = F.pad(x, (self.reserve_cls_tokens, 0), value=-100.0)
            if d is None:
                return x.transpose(1, 2).unsqueeze(-1), None, None, x.shape[-1]

            valid_d = d + self.reserve_cls_tokens
            bsz, _, hc = x.shape
            indices = torch.arange(hc, device=x.device).unsqueeze(0).expand(bsz, hc)
            mask = indices < valid_d.unsqueeze(1)
            return x.transpose(1, 2)[mask].unsqueeze(-1), mask, valid_d, hc

        grouped = self.feature_grouping(x, d)
        if self.reserve_cls_tokens > 0:
            grouped = F.pad(grouped, (0, 0, self.reserve_cls_tokens, 0), value=-100.0)
        if d is None:
            return grouped.transpose(1, 2), None, None, grouped.shape[2]

        valid_groups = self.effective_feature_count(d) + self.reserve_cls_tokens
        bsz, _, gc, _ = grouped.shape
        indices = torch.arange(gc, device=x.device).unsqueeze(0).expand(bsz, gc)
        mask = indices < valid_groups.unsqueeze(1)
        return grouped.transpose(1, 2)[mask], mask, valid_groups, gc

    def _expand_targets(self, y_train: Optional[Tensor], width: int, mask: Optional[Tensor]) -> Optional[Tensor]:
        if y_train is None:
            return None
        expanded = y_train.unsqueeze(1).expand(-1, width, -1)
        if mask is not None:
            expanded = expanded[mask]
        return expanded

    def _rebuild_embeddings(
        self,
        effective_embeddings: Tensor,
        mask: Optional[Tensor],
        width: int,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Tensor:
        if mask is None:
            return effective_embeddings
        embeddings = torch.zeros(batch_size, width, seq_len, self.embed_dim, device=device, dtype=effective_embeddings.dtype)
        embeddings[mask] = effective_embeddings
        return embeddings

    def _train_forward(
        self,
        x: Tensor,
        d: Optional[Tensor] = None,
        train_size: Optional[int] = None,
        y_train: Optional[Tensor] = None,
        embed_with_test: bool = False,
    ) -> Tensor:
        if train_size is None and y_train is not None and not embed_with_test:
            train_size = y_train.shape[1]

        features, mask, _, width = self._prepare_features(x, d)
        y_features = self._expand_targets(y_train, width, mask)
        effective_embeddings = self._compute_embeddings(features, train_size, y_features, embed_with_test)
        embeddings = self._rebuild_embeddings(
            effective_embeddings,
            mask,
            width,
            x.shape[0],
            x.shape[1],
            x.device,
        )
        return embeddings.transpose(1, 2)

    def _inference_forward(
        self,
        x: Tensor,
        train_size: Optional[int] = None,
        feature_shuffles: Optional[List[List[int]]] = None,
        mgr_config: MgrConfig = None,
        y_train: Optional[Tensor] = None,
        embed_with_test: bool = False,
    ) -> Tensor:
        if mgr_config is None:
            mgr_config = InferenceConfig().COL_CONFIG
        self.inference_mgr.configure(**mgr_config)

        if feature_shuffles is None or self.feature_group:
            features, _, _, width = self._prepare_features(x)
            y_features = self._expand_targets(y_train, width, None)
            embeddings = self.inference_mgr(
                self._compute_embeddings,
                inputs=OrderedDict(
                    [
                        ("features", features),
                        ("train_size", train_size),
                        ("y_train", y_features),
                        ("embed_with_test", embed_with_test),
                    ]
                ),
            )
            return embeddings.transpose(1, 2)

        batch = x.shape[0]
        first_table = x[0]
        if self.reserve_cls_tokens > 0:
            first_table = F.pad(first_table, (self.reserve_cls_tokens, 0), value=-100.0)
        features = first_table.transpose(0, 1).unsqueeze(-1)
        y_features = None
        if y_train is not None:
            y_features = y_train[0].unsqueeze(0).expand(features.shape[0], -1)
        first_embeddings = self.inference_mgr(
            self._compute_embeddings,
            inputs=OrderedDict(
                [
                    ("features", features),
                    ("train_size", train_size),
                    ("y_train", y_features),
                    ("embed_with_test", embed_with_test),
                ]
            ),
            output_repeat=batch,
        )
        embeddings = first_embeddings.unsqueeze(0).repeat(batch, 1, 1, 1)
        first_pattern = feature_shuffles[0]
        for i in range(1, batch):
            mapping = self.map_feature_shuffle(first_pattern, feature_shuffles[i])
            if self.reserve_cls_tokens > 0:
                mapping = [m + self.reserve_cls_tokens for m in mapping]
                mapping = list(range(self.reserve_cls_tokens)) + mapping
            embeddings[i] = first_embeddings[mapping]
        return embeddings.transpose(1, 2)

    def forward(
        self,
        x: Tensor,
        d: Optional[Tensor] = None,
        train_size: Optional[int] = None,
        y_train: Optional[Tensor] = None,
        embed_with_test: bool = False,
        feature_shuffles: Optional[List[List[int]]] = None,
        mgr_config: MgrConfig = None,
    ) -> Tensor:
        if self.training:
            return self._train_forward(x, d, train_size, y_train, embed_with_test)
        return self._inference_forward(x, train_size, feature_shuffles, mgr_config, y_train, embed_with_test)

    def forward_with_cache(
        self,
        x: Tensor,
        col_cache: KVCache,
        y_train: Optional[Tensor] = None,
        d: Optional[Tensor] = None,
        use_cache: bool = False,
        store_cache: bool = True,
        mgr_config: MgrConfig = None,
    ) -> Tensor:
        if use_cache == store_cache:
            raise ValueError("Exactly one of use_cache or store_cache must be True")
        if store_cache and y_train is None:
            raise ValueError("y_train must be provided when store_cache=True")

        if mgr_config is None:
            mgr_config = InferenceConfig().COL_CONFIG
        self.inference_mgr.configure(**mgr_config)

        train_size = y_train.shape[1] if store_cache else None
        features, mask, _, width = self._prepare_features(x, d)
        y_features = self._expand_targets(y_train, width, mask) if store_cache else None
        effective_embeddings = self.inference_mgr(
            self._compute_embeddings_with_cache,
            inputs=OrderedDict(
                [
                    ("features", features),
                    ("col_cache", col_cache),
                    ("train_size", train_size),
                    ("y_train", y_features),
                    ("use_cache", use_cache),
                    ("store_cache", store_cache),
                ]
            ),
        )
        embeddings = self._rebuild_embeddings(
            effective_embeddings,
            mask,
            width,
            x.shape[0],
            x.shape[1],
            x.device,
        )
        return embeddings.transpose(1, 2)
