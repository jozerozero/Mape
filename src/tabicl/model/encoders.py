from __future__ import annotations

from typing import Optional

from torch import Tensor, nn

from .kv_cache import KVCache, KVCacheEntry
from .layers import InducedSelfAttentionBlock, MultiheadAttentionBlock
from .rope import RotaryEmbedding


class Encoder(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        use_rope: bool = False,
        rope_base: int = 100000,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.blocks = nn.ModuleList(
            [
                MultiheadAttentionBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    norm_first=norm_first,
                    bias_free_ln=bias_free_ln,
                )
                for _ in range(num_blocks)
            ]
        )
        self.rope = RotaryEmbedding(dim=d_model // nhead, theta=rope_base) if use_rope else None

    def forward(
        self,
        src: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor | int] = None,
        train_size: Optional[int] = None,
    ) -> Tensor:
        out = src
        for block in self.blocks:
            out = block(
                q=out,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                train_size=train_size,
                rope=self.rope,
            )
        return out

    def forward_with_cache(
        self,
        src: Tensor,
        icl_cache: KVCache,
        train_size: Optional[int] = None,
        use_cache: bool = False,
        store_cache: bool = True,
    ) -> Tensor:
        if use_cache == store_cache:
            raise ValueError("Exactly one of use_cache or store_cache must be True")
        if store_cache and train_size is None:
            raise ValueError("train_size must be provided when store_cache=True")

        out = src
        for layer_idx, block in enumerate(self.blocks):
            if use_cache:
                out = block(q=out, rope=self.rope, cached_kv=icl_cache.kv[layer_idx])
            else:
                out, k_proj, v_proj = block(q=out, train_size=train_size, rope=self.rope, need_kv=True)
                icl_cache.kv[layer_idx] = KVCacheEntry(key=k_proj, value=v_proj)
        return out


class SetTransformer(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int = 16,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.blocks = nn.ModuleList(
            [
                InducedSelfAttentionBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    num_inds=num_inds,
                    dropout=dropout,
                    activation=activation,
                    norm_first=norm_first,
                    bias_free_ln=bias_free_ln,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, src: Tensor, train_size: Optional[int] = None) -> Tensor:
        out = src
        for block in self.blocks:
            out = block(out, train_size)
        return out

    def forward_with_cache(
        self,
        src: Tensor,
        col_cache: KVCache,
        train_size: Optional[int] = None,
        use_cache: bool = False,
        store_cache: bool = True,
    ) -> Tensor:
        if use_cache == store_cache:
            raise ValueError("Exactly one of use_cache or store_cache must be True")
        if store_cache and train_size is None:
            raise ValueError("train_size must be provided when store_cache=True")

        out = src
        for block_idx, block in enumerate(self.blocks):
            out = block.forward_with_cache(out, col_cache, block_idx, train_size, use_cache, store_cache)
        return out
