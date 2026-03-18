from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class KVCacheEntry:
    key: Optional[Tensor] = None
    value: Optional[Tensor] = None

    def is_valid(self) -> bool:
        return self.key is not None and self.value is not None

    def __getitem__(self, indices) -> "KVCacheEntry":
        if not self.is_valid():
            return KVCacheEntry()
        return KVCacheEntry(key=self.key[indices], value=self.value[indices])

    def __setitem__(self, indices, other: "KVCacheEntry") -> None:
        if self.is_valid() and other.is_valid():
            self.key[indices] = other.key
            self.value[indices] = other.value

    def to(self, device) -> "KVCacheEntry":
        if not self.is_valid():
            return KVCacheEntry()
        return KVCacheEntry(key=self.key.to(device), value=self.value.to(device))

    @staticmethod
    def concat(entries: List["KVCacheEntry"], dim: int = 0) -> "KVCacheEntry":
        keys = [entry.key for entry in entries if entry.is_valid()]
        values = [entry.value for entry in entries if entry.is_valid()]
        if not keys:
            return KVCacheEntry()
        return KVCacheEntry(key=torch.cat(keys, dim=dim), value=torch.cat(values, dim=dim))


@dataclass
class KVCache:
    kv: Dict[int, KVCacheEntry] = field(default_factory=dict)

    def is_populated(self) -> bool:
        return any(entry.is_valid() for entry in self.kv.values())

    def __getitem__(self, indices) -> "KVCache":
        return self.__class__(kv={idx: entry[indices] for idx, entry in self.kv.items()})

    def __setitem__(self, indices, other: "KVCache") -> None:
        for idx, other_entry in other.kv.items():
            if idx in self.kv:
                device = self.kv[idx].key.device
                self.kv[idx][indices] = other_entry.to(device)

    def to(self, device) -> "KVCache":
        return self.__class__(kv={idx: entry.to(device) for idx, entry in self.kv.items()})

    @staticmethod
    def concat(caches: List["KVCache"], dim: int = 0) -> "KVCache":
        all_indices = set()
        for cache in caches:
            all_indices.update(cache.kv.keys())

        merged = {}
        for idx in sorted(all_indices):
            entries = [cache.kv[idx] for cache in caches if idx in cache.kv]
            merged[idx] = KVCacheEntry.concat(entries, dim=dim)
        return KVCache(kv=merged)

    def preallocate(self, reference: "KVCache", batch_shape: tuple, device="cpu") -> None:
        for idx, ref_entry in reference.kv.items():
            if ref_entry.is_valid():
                key_shape = batch_shape + ref_entry.key.shape[-3:]
                value_shape = batch_shape + ref_entry.value.shape[-3:]
                self.kv[idx] = KVCacheEntry(
                    key=torch.zeros(key_shape, dtype=ref_entry.key.dtype, device=device),
                    value=torch.zeros(value_shape, dtype=ref_entry.value.dtype, device=device),
                )


@dataclass
class TabICLCache:
    col_cache: Optional[KVCache] = None
    row_repr: Optional[Tensor] = None
    icl_cache: Optional[KVCache] = None
    train_shape: Tuple[int, int, int] = (0, 0, 0)
    num_classes: Optional[int] = None

    def __post_init__(self) -> None:
        if self.col_cache is None:
            self.col_cache = KVCache()
        if self.icl_cache is None:
            self.icl_cache = KVCache()

    @property
    def cache_type(self) -> str:
        if self.row_repr is not None:
            return "repr"
        if (self.col_cache and self.col_cache.kv) or (self.icl_cache and self.icl_cache.kv):
            return "kv"
        return "empty"

    def is_empty(self) -> bool:
        return (
            (self.col_cache is None or not self.col_cache.kv)
            and self.row_repr is None
            and (self.icl_cache is None or not self.icl_cache.kv)
        )

    def cache_size_mb(self) -> int:
        total = 0
        if self.col_cache:
            for kv in self.col_cache.kv.values():
                if kv.key is not None:
                    total += kv.key.numel() * kv.key.element_size()
                if kv.value is not None:
                    total += kv.value.numel() * kv.value.element_size()
        if self.row_repr is not None:
            total += self.row_repr.numel() * self.row_repr.element_size()
        if self.icl_cache:
            for kv in self.icl_cache.kv.values():
                if kv.key is not None:
                    total += kv.key.numel() * kv.key.element_size()
                if kv.value is not None:
                    total += kv.value.numel() * kv.value.element_size()
        return total // (1024 * 1024)

    def slice_batch(self, start: int, end: int) -> "TabICLCache":
        indices = slice(start, end)
        return TabICLCache(
            col_cache=self.col_cache[indices] if self.col_cache else KVCache(),
            row_repr=self.row_repr[indices] if self.row_repr is not None else None,
            icl_cache=self.icl_cache[indices] if self.icl_cache else KVCache(),
            train_shape=(end - start, self.train_shape[1], self.train_shape[2]),
            num_classes=self.num_classes,
        )

    def to(self, device) -> "TabICLCache":
        return TabICLCache(
            col_cache=self.col_cache.to(device) if self.col_cache else KVCache(),
            row_repr=self.row_repr.to(device) if self.row_repr is not None else None,
            icl_cache=self.icl_cache.to(device) if self.icl_cache else KVCache(),
            train_shape=self.train_shape,
            num_classes=self.num_classes,
        )

    @staticmethod
    def concat(caches: List["TabICLCache"], dim: int = 0) -> "TabICLCache":
        col_caches = [cache.col_cache for cache in caches if cache.col_cache is not None]
        row_reprs = [cache.row_repr for cache in caches if cache.row_repr is not None]
        icl_caches = [cache.icl_cache for cache in caches if cache.icl_cache is not None]

        total_batch = sum(cache.train_shape[0] for cache in caches)
        train_size = caches[0].train_shape[1]
        n_features = caches[0].train_shape[2]

        return TabICLCache(
            col_cache=KVCache.concat(col_caches, dim=dim) if col_caches else KVCache(),
            row_repr=torch.cat(row_reprs, dim=dim) if row_reprs else None,
            icl_cache=KVCache.concat(icl_caches, dim=dim) if icl_caches else KVCache(),
            train_shape=(total_batch, train_size, n_features),
            num_classes=caches[0].num_classes,
        )
