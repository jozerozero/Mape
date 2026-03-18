from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification
from sklearn.utils.estimator_checks import parametrize_with_checks

from tabicl import TabICL, TabICLClassifier


# use n_estimators=2 to test other preprocessing as well
@parametrize_with_checks([TabICLClassifier(n_estimators=2)])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


def _save_checkpoint(path: Path, config: dict) -> Path:
    model = TabICL(**config)
    torch.save({"config": config, "state_dict": model.state_dict()}, path)
    return path


def _legacy_config() -> dict:
    return {
        "max_classes": 2,
        "embed_dim": 32,
        "col_num_blocks": 1,
        "col_nhead": 4,
        "col_num_inds": 8,
        "row_num_blocks": 1,
        "row_nhead": 4,
        "row_num_cls": 2,
        "icl_num_blocks": 2,
        "icl_nhead": 4,
        "ff_factor": 2,
        "dropout": 0.0,
        "activation": "gelu",
        "norm_first": True,
    }


def _v2_config() -> dict:
    return {
        **_legacy_config(),
        "arch_mode": "v2",
        "col_affine": False,
        "col_feature_group": False,
        "col_feature_group_size": 3,
        "col_target_aware": False,
        "row_last_cls_only": True,
        "bias_free_ln": False,
    }


def _make_classifier(checkpoint_path: Path) -> TabICLClassifier:
    return TabICLClassifier(
        n_estimators=2,
        model_path=checkpoint_path,
        allow_auto_download=False,
        device="cpu",
        use_amp=False,
        batch_size=2,
    )


def test_legacy_checkpoint_load_sets_legacy_mode(tmp_path):
    checkpoint_path = _save_checkpoint(tmp_path / "legacy.ckpt", _legacy_config())
    clf = _make_classifier(checkpoint_path)
    clf._load_model()
    assert clf.model_.arch_mode == "legacy"
    assert clf.model_.row_last_cls_only is False
    assert clf.model_.col_feature_group is False


def test_legacy_checkpoint_repr_cache_falls_back_to_kv(tmp_path):
    checkpoint_path = _save_checkpoint(tmp_path / "legacy_cache.ckpt", _legacy_config())
    X, y = make_classification(n_samples=50, n_features=5, n_informative=3, n_redundant=0, random_state=42)
    X_train, X_test = X[:40], X[40:]
    y_train = y[:40]

    clf_plain = _make_classifier(checkpoint_path)
    clf_plain.fit(X_train, y_train)
    pred_plain = clf_plain.predict_proba(X_test)

    clf_cached = _make_classifier(checkpoint_path)
    clf_cached.fit(X_train, y_train, kv_cache="repr")
    pred_cached = clf_cached.predict_proba(X_test)

    assert clf_cached.cache_mode_ == "kv"
    np.testing.assert_allclose(pred_plain, pred_cached, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("kv_cache", ["kv", "repr"])
def test_v2_checkpoint_cache_matches_no_cache(tmp_path, kv_cache):
    checkpoint_path = _save_checkpoint(tmp_path / f"v2_{kv_cache}.ckpt", _v2_config())
    X, y = make_classification(n_samples=50, n_features=5, n_informative=3, n_redundant=0, random_state=0)
    X_train, X_test = X[:40], X[40:]
    y_train = y[:40]

    clf_plain = _make_classifier(checkpoint_path)
    clf_plain.fit(X_train, y_train)
    pred_plain = clf_plain.predict_proba(X_test)

    clf_cached = _make_classifier(checkpoint_path)
    clf_cached.fit(X_train, y_train, kv_cache=kv_cache)
    pred_cached = clf_cached.predict_proba(X_test)

    np.testing.assert_allclose(pred_plain, pred_cached, rtol=1e-4, atol=1e-4)
