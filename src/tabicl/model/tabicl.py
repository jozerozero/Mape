from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor, nn

from .embedding import ColEmbedding
from .inference_config import InferenceConfig
from .interaction import RowInteraction
from .kv_cache import TabICLCache
from .learning import ICLearning


class TabICL(nn.Module):
    def __init__(
        self,
        max_classes: int = 10,
        embed_dim: int = 128,
        col_num_blocks: int = 3,
        col_nhead: int = 4,
        col_num_inds: int = 128,
        col_affine: bool = True,
        col_feature_group: bool | str = False,
        col_feature_group_size: int = 3,
        col_target_aware: bool = False,
        row_num_blocks: int = 3,
        row_nhead: int = 8,
        row_num_cls: int = 4,
        row_rope_base: float = 100000,
        row_last_cls_only: bool = False,
        icl_num_blocks: int = 12,
        icl_nhead: int = 4,
        ff_factor: int = 2,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        arch_mode: str = "v2",
    ):
        super().__init__()
        self.max_classes = max_classes
        self.embed_dim = embed_dim
        self.col_num_blocks = col_num_blocks
        self.col_nhead = col_nhead
        self.col_num_inds = col_num_inds
        self.col_affine = col_affine
        self.col_feature_group = col_feature_group
        self.col_feature_group_size = col_feature_group_size
        self.col_target_aware = col_target_aware
        self.row_num_blocks = row_num_blocks
        self.row_nhead = row_nhead
        self.row_num_cls = row_num_cls
        self.row_rope_base = row_rope_base
        self.row_last_cls_only = row_last_cls_only
        self.icl_num_blocks = icl_num_blocks
        self.icl_nhead = icl_nhead
        self.ff_factor = ff_factor
        self.dropout = dropout
        self.activation = activation
        self.norm_first = norm_first
        self.bias_free_ln = bias_free_ln
        self.arch_mode = arch_mode

        self.col_embedder = ColEmbedding(
            embed_dim=embed_dim,
            num_blocks=col_num_blocks,
            nhead=col_nhead,
            num_inds=col_num_inds,
            dim_feedforward=embed_dim * ff_factor,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            affine=col_affine,
            feature_group=col_feature_group,
            feature_group_size=col_feature_group_size,
            target_aware=col_target_aware,
            max_classes=max_classes,
            reserve_cls_tokens=row_num_cls,
        )
        self.row_interactor = RowInteraction(
            embed_dim=embed_dim,
            num_blocks=row_num_blocks,
            nhead=row_nhead,
            dim_feedforward=embed_dim * ff_factor,
            num_cls=row_num_cls,
            rope_base=row_rope_base,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            row_last_cls_only=row_last_cls_only,
        )

        icl_dim = embed_dim * row_num_cls
        self.icl_predictor = ICLearning(
            max_classes=max_classes,
            d_model=icl_dim,
            num_blocks=icl_num_blocks,
            nhead=icl_nhead,
            dim_feedforward=icl_dim * ff_factor,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
        )
        self._cache: Optional[TabICLCache] = None

    @property
    def has_cache(self) -> bool:
        return self._cache is not None and not self._cache.is_empty()

    def clear_cache(self) -> None:
        self._cache = None

    def _effective_row_d(self, d: Optional[Tensor], n_features: int) -> Optional[Tensor]:
        if d is None:
            return None
        if len(d.unique()) == 1 and d[0] == n_features:
            return None
        return self.col_embedder.effective_feature_count(d)

    def _train_forward(
        self,
        X: Tensor,
        y_train: Tensor,
        d: Optional[Tensor] = None,
        embed_with_test: bool = False,
    ) -> Tensor:
        _, _, n_features = X.shape
        train_size = y_train.shape[1]
        assert train_size <= X.shape[1], "Number of training samples exceeds total samples"

        row_d = self._effective_row_d(d, n_features)
        representations = self.row_interactor(
            self.col_embedder(
                X,
                d=d,
                train_size=None if embed_with_test else train_size,
                y_train=y_train,
                embed_with_test=embed_with_test,
            ),
            d=row_d,
        )
        return self.icl_predictor(representations, y_train=y_train)

    def _inference_forward(
        self,
        X: Tensor,
        y_train: Tensor,
        feature_shuffles: Optional[List[List[int]]] = None,
        embed_with_test: bool = False,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        inference_config: InferenceConfig = None,
    ) -> Tensor:
        train_size = y_train.shape[1]
        assert train_size <= X.shape[1], "Number of training samples exceeds total samples"

        if inference_config is None:
            inference_config = InferenceConfig()

        representations = self.row_interactor(
            self.col_embedder(
                X,
                train_size=None if embed_with_test else train_size,
                y_train=y_train,
                embed_with_test=embed_with_test,
                feature_shuffles=feature_shuffles,
                mgr_config=inference_config.COL_CONFIG,
            ),
            mgr_config=inference_config.ROW_CONFIG,
        )
        return self.icl_predictor(
            representations,
            y_train=y_train,
            return_logits=return_logits,
            softmax_temperature=softmax_temperature,
            mgr_config=inference_config.ICL_CONFIG,
        )

    def forward(
        self,
        X: Tensor,
        y_train: Tensor,
        d: Optional[Tensor] = None,
        feature_shuffles: Optional[List[List[int]]] = None,
        embed_with_test: bool = False,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        inference_config: InferenceConfig = None,
    ) -> Tensor:
        if self.training:
            return self._train_forward(X, y_train, d=d, embed_with_test=embed_with_test)
        return self._inference_forward(
            X,
            y_train,
            feature_shuffles=feature_shuffles,
            embed_with_test=embed_with_test,
            return_logits=return_logits,
            softmax_temperature=softmax_temperature,
            inference_config=inference_config,
        )

    def forward_with_cache(
        self,
        X_train: Optional[Tensor] = None,
        y_train: Optional[Tensor] = None,
        X_test: Optional[Tensor] = None,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        use_cache: bool = False,
        store_cache: bool = True,
        cache: Optional[TabICLCache] = None,
        cache_mode: str = "kv",
        inference_config: Optional[InferenceConfig] = None,
    ) -> Optional[Tensor]:
        if cache is not None:
            use_cache = True
            store_cache = False
            self._cache = cache

        if use_cache == store_cache:
            raise ValueError("use_cache and store_cache must be mutually exclusive")

        if inference_config is None:
            inference_config = InferenceConfig()

        if self.arch_mode == "legacy" and cache_mode == "repr":
            cache_mode = "kv"

        if use_cache and self._cache is not None and self._cache.cache_type == "repr":
            cache_mode = "repr"

        if store_cache:
            if X_train is None or y_train is None:
                raise ValueError("X_train and y_train must be provided when store_cache=True")
            num_classes = len(y_train[0].unique())
            self._cache = TabICLCache(train_shape=X_train.shape, num_classes=num_classes)
            X = X_train if X_test is None else torch.cat([X_train, X_test], dim=1)
        else:
            if X_test is None:
                raise ValueError("X_test must be provided when use_cache=True")
            if self._cache is None or self._cache.is_empty():
                raise ValueError("No cache available. Call once with store_cache=True first.")
            X = X_test
            y_train = None

        col_embeddings = self.col_embedder.forward_with_cache(
            X,
            col_cache=self._cache.col_cache,
            y_train=y_train,
            use_cache=use_cache,
            store_cache=store_cache,
            mgr_config=inference_config.COL_CONFIG,
        )
        representations = self.row_interactor(col_embeddings, mgr_config=inference_config.ROW_CONFIG)

        if cache_mode == "repr":
            if store_cache:
                train_size = y_train.shape[1]
                representations = self.icl_predictor.prepare_repr_cache(representations, y_train)
                self._cache.row_repr = representations[:, :train_size]
                if X_test is None:
                    return None
            else:
                train_repr = self._cache.row_repr.to(representations.device)
                train_size = train_repr.shape[1]
                representations = torch.cat([train_repr, representations], dim=1)

            return self.icl_predictor.forward_with_repr_cache(
                representations,
                train_size=train_size,
                num_classes=self._cache.num_classes,
                return_logits=return_logits,
                softmax_temperature=softmax_temperature,
                mgr_config=inference_config.ICL_CONFIG,
            )

        out = self.icl_predictor.forward_with_cache(
            representations,
            icl_cache=self._cache.icl_cache,
            y_train=y_train,
            num_classes=self._cache.num_classes,
            return_logits=return_logits,
            softmax_temperature=softmax_temperature,
            use_cache=use_cache,
            store_cache=store_cache,
            mgr_config=inference_config.ICL_CONFIG,
        )
        if X_test is None and store_cache:
            return None
        return out
