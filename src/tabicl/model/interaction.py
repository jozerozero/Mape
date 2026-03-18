from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import torch
from torch import Tensor, nn

from .encoders import Encoder
from .inference import InferenceManager
from .inference_config import InferenceConfig, MgrConfig


class RowInteraction(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        num_cls: int = 4,
        rope_base: float = 100000,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        row_last_cls_only: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_cls = num_cls
        self.norm_first = norm_first
        self.row_last_cls_only = row_last_cls_only

        self.tf_row = Encoder(
            num_blocks=num_blocks,
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            use_rope=False,
            rope_base=rope_base,
        )
        self.cls_tokens = nn.Parameter(torch.empty(num_cls, embed_dim))
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)

        self.out_ln = nn.LayerNorm(embed_dim, bias=not bias_free_ln) if norm_first else nn.Identity()
        self.repr_ln = nn.Identity() if row_last_cls_only else nn.LayerNorm(embed_dim * self.num_cls)
        self.inference_mgr = InferenceManager(enc_name="tf_row", out_dim=embed_dim * self.num_cls, out_no_seq=True)

    def _aggregate_embeddings(self, embeddings: Tensor, key_mask: Optional[Tensor] = None) -> Tensor:
        if not self.row_last_cls_only:
            outputs = self.tf_row(embeddings, key_padding_mask=key_mask)
            cls_outputs = outputs[..., : self.num_cls, :].clone()
            del outputs
            cls_outputs = self.out_ln(cls_outputs)
            return cls_outputs.flatten(-2)

        rope = self.tf_row.rope
        hidden = embeddings
        for block in self.tf_row.blocks[:-1]:
            hidden = block(q=hidden, key_padding_mask=key_mask, rope=rope)
        last_block = self.tf_row.blocks[-1]
        cls_outputs = last_block(
            q=hidden[..., : self.num_cls, :],
            k=hidden,
            v=hidden,
            key_padding_mask=key_mask,
            rope=rope,
        )
        cls_outputs = self.out_ln(cls_outputs)
        return cls_outputs.flatten(-2)

    def _train_forward(self, embeddings: Tensor, d: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len, n_features, _ = embeddings.shape
        device = embeddings.device

        cls_tokens = self.cls_tokens.expand(batch_size, seq_len, self.num_cls, self.embed_dim)
        embeddings[:, :, : self.num_cls] = cls_tokens.to(embeddings.device)

        if d is None:
            key_mask = None
        else:
            d = d + self.num_cls
            indices = torch.arange(n_features, device=device).view(1, 1, n_features).expand(batch_size, seq_len, n_features)
            key_mask = indices >= d.view(batch_size, 1, 1)

        representations = self._aggregate_embeddings(embeddings, key_mask)
        return self.repr_ln(representations)

    def _inference_forward(self, embeddings: Tensor, mgr_config: MgrConfig = None) -> Tensor:
        if mgr_config is None:
            mgr_config = InferenceConfig().ROW_CONFIG
        self.inference_mgr.configure(**mgr_config)

        batch_size, seq_len = embeddings.shape[:2]
        cls_tokens = self.cls_tokens.expand(batch_size, seq_len, self.num_cls, self.embed_dim)
        embeddings[:, :, : self.num_cls] = cls_tokens.to(embeddings.device)
        representations = self.inference_mgr(self._aggregate_embeddings, inputs=OrderedDict([("embeddings", embeddings)]))
        return self.repr_ln(representations)

    def forward(self, embeddings: Tensor, d: Optional[Tensor] = None, mgr_config: MgrConfig = None) -> Tensor:
        if self.training:
            return self._train_forward(embeddings, d)
        return self._inference_forward(embeddings, mgr_config)
