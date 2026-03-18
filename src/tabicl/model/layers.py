from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .attention import multi_head_attention_forward
from .kv_cache import KVCache, KVCacheEntry
from .rope import RotaryEmbedding


class ClassNode:
    def __init__(self, depth=0):
        self.depth = depth
        self.is_leaf = False
        self.classes_ = None
        self.child_nodes = []
        self.class_mapping = {}
        self.group_indices = None
        self.R = None
        self.y = None


class OneHotAndLinear(nn.Linear):
    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__(num_classes, embed_dim)
        self.num_classes = num_classes
        self.embed_dim = embed_dim

    def forward(self, src: Tensor) -> Tensor:
        one_hot = F.one_hot(src.long(), self.num_classes).to(src.dtype)
        return F.linear(one_hot.float(), self.weight, self.bias)


class SkippableLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, skip_value: float = -100.0):
        super().__init__(in_features, out_features, bias)
        self.skip_value = skip_value

    def forward(self, src: Tensor) -> Tensor:
        out = F.linear(src, self.weight, self.bias)
        skip_mask = (src == self.skip_value).all(dim=-1)
        if skip_mask.any():
            out[skip_mask] = self.skip_value
        return out


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        hidden_dims: List[int] = [256, 256, 256],
        activation: str = "gelu",
        bias: bool = True,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = in_dim
        act = self.get_activation(activation)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=bias))
            layers.append(act())
            prev_dim = hidden_dim
        if out_dim is not None:
            layers.append(nn.Linear(prev_dim, out_dim, bias=bias))
        self.net = nn.Sequential(*layers)

    @staticmethod
    def get_activation(activation: str):
        mapping = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
        if activation not in mapping:
            raise ValueError(f"Unknown activation: {activation}. Supported: {list(mapping.keys())}")
        return mapping[activation]

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MultiheadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__(embed_dim, num_heads, dropout, batch_first=True)

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        cached_kv: Optional[KVCacheEntry] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        rope: Optional[RotaryEmbedding] = None,
        need_kv: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="src_mask",
            target_type=query.dtype,
        )
        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )
        return multi_head_attention_forward(
            query,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            key=key,
            value=value,
            cached_kv=cached_kv,
            training=self.training,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            rope=rope,
            need_kv=need_kv,
        )


class MultiheadAttentionBlock(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
    ):
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=True,
        )
        if bias_free_ln:
            self.norm1 = nn.LayerNorm(d_model, bias=False)
            self.norm2 = nn.LayerNorm(d_model, bias=False)
        del self.self_attn
        self.attn = MultiheadAttention(d_model, nhead, dropout)
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(
        self,
        q: Tensor,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        cached_kv: Optional[KVCacheEntry] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        train_size: Optional[int] = None,
        rope: Optional[RotaryEmbedding] = None,
        need_kv: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        if train_size is None:
            k = q if k is None else k
            v = q if v is None else v
        else:
            assert k is None and v is None, "k and v must be None when train_size is provided"
            k = v = q[..., :train_size, :]

        k_proj = v_proj = None
        if self.norm_first:
            q_normed = self.norm1(q)
            if cached_kv is not None:
                attn = self._attn_block(
                    q_normed,
                    cached_kv=cached_kv,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                    rope=rope,
                )
            else:
                if train_size is None:
                    k_normed = self.norm1(k) if k is not q else q_normed
                    v_normed = self.norm1(v) if v is not k else k_normed
                else:
                    k_normed = v_normed = q_normed[..., :train_size, :]
                attn_result = self._attn_block(
                    q_normed,
                    k_normed,
                    v_normed,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                    rope=rope,
                    need_kv=need_kv,
                )
                if need_kv and isinstance(attn_result, tuple):
                    attn, k_proj, v_proj = attn_result
                else:
                    attn = attn_result
            x = q + attn
            x = x + self._ff_block(self.norm2(x))
        else:
            if cached_kv is not None:
                attn = self._attn_block(
                    q,
                    cached_kv=cached_kv,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                    rope=rope,
                )
            else:
                attn_result = self._attn_block(
                    q,
                    k,
                    v,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                    rope=rope,
                    need_kv=need_kv,
                )
                if need_kv and isinstance(attn_result, tuple):
                    attn, k_proj, v_proj = attn_result
                else:
                    attn = attn_result
            x = self.norm1(q + attn)
            x = self.norm2(x + self._ff_block(x))

        if need_kv and k_proj is not None:
            return x, k_proj, v_proj
        return x

    def _attn_block(
        self,
        q: Tensor,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        cached_kv: Optional[KVCacheEntry] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        rope: Optional[RotaryEmbedding] = None,
        need_kv: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        result = self.attn(
            q,
            k,
            v,
            cached_kv=cached_kv,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            rope=rope,
            need_kv=need_kv,
        )
        if need_kv and isinstance(result, tuple):
            attn, k_proj, v_proj = result
            return self.dropout1(attn), k_proj, v_proj
        return self.dropout1(result)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class InducedSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        skip_value: float = -100.0,
    ):
        super().__init__()
        self.skip_value = skip_value
        self.num_inds = num_inds
        self.multihead_attn1 = MultiheadAttentionBlock(
            d_model, nhead, dim_feedforward, dropout, activation, norm_first, bias_free_ln
        )
        self.multihead_attn2 = MultiheadAttentionBlock(
            d_model, nhead, dim_feedforward, dropout, activation, norm_first, bias_free_ln
        )
        self.ind_vectors = nn.Parameter(torch.empty(num_inds, d_model))
        nn.init.trunc_normal_(self.ind_vectors, std=0.02)

    def induced_attention(self, src: Tensor, train_size: Optional[int] = None) -> Tensor:
        *batch_shape, _, d_model = src.shape
        ind_vectors = self.ind_vectors.expand(*batch_shape, self.num_inds, d_model)
        if train_size is None:
            hidden = self.multihead_attn1(ind_vectors, src, src)
        else:
            hidden = self.multihead_attn1(ind_vectors, src[..., :train_size, :], src[..., :train_size, :])
        return self.multihead_attn2(src, hidden, hidden)

    def forward(self, src: Tensor, train_size: Optional[int] = None) -> Tensor:
        skip_mask = (src == self.skip_value).all(dim=(-2, -1))
        if skip_mask.any():
            if skip_mask.all():
                return torch.full_like(src, self.skip_value)
            out = torch.empty_like(src)
            out[~skip_mask] = self.induced_attention(src[~skip_mask], train_size)
            out[skip_mask] = self.skip_value
            return out
        return self.induced_attention(src, train_size)

    def induced_attention_with_cache(
        self,
        src: Tensor,
        col_cache: KVCache,
        block_idx: int,
        train_size: Optional[int] = None,
        use_cache: bool = False,
        store_cache: bool = True,
    ) -> Tensor:
        *batch_shape, _, d_model = src.shape
        ind_vectors = self.ind_vectors.expand(*batch_shape, self.num_inds, d_model)

        if use_cache:
            assert block_idx in col_cache.kv, f"Cache miss for block {block_idx}"
            return self.multihead_attn2(src, cached_kv=col_cache.kv[block_idx])

        assert train_size is not None, "train_size must be provided when store_cache=True"
        hidden = self.multihead_attn1(ind_vectors, src[..., :train_size, :], src[..., :train_size, :])
        out, k_proj, v_proj = self.multihead_attn2(src, hidden, hidden, need_kv=True)
        col_cache.kv[block_idx] = KVCacheEntry(key=k_proj, value=v_proj)
        return out

    def forward_with_cache(
        self,
        src: Tensor,
        col_cache: KVCache,
        block_idx: int,
        train_size: Optional[int] = None,
        use_cache: bool = False,
        store_cache: bool = True,
    ) -> Tensor:
        if use_cache == store_cache:
            raise ValueError("Exactly one of use_cache or store_cache must be True")
        if store_cache and train_size is None:
            raise ValueError("train_size must be provided when store_cache=True")

        skip_mask = (src == self.skip_value).all(dim=(-2, -1))
        if skip_mask.all():
            return torch.full_like(src, self.skip_value)

        out = self.induced_attention_with_cache(src, col_cache, block_idx, train_size, use_cache, store_cache)
        if skip_mask.any():
            out[skip_mask] = self.skip_value
        return out
