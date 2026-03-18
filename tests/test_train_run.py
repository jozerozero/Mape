import sys
import types
from types import SimpleNamespace

import torch
from torch import nn

sys.modules.setdefault("wandb", types.SimpleNamespace(init=lambda *args, **kwargs: None))
sys.modules.setdefault("xgboost", types.SimpleNamespace(XGBRegressor=object, XGBClassifier=object))
if "transformers" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.get_constant_schedule = lambda *args, **kwargs: None
    transformers_stub.get_linear_schedule_with_warmup = lambda *args, **kwargs: None
    transformers_stub.get_cosine_schedule_with_warmup = lambda *args, **kwargs: None
    transformers_stub.get_polynomial_decay_schedule_with_warmup = lambda *args, **kwargs: None
    optimization_stub = types.ModuleType("transformers.optimization")
    optimization_stub.Adafactor = object
    transformers_stub.optimization = optimization_stub
    sys.modules["transformers"] = transformers_stub
    sys.modules["transformers.optimization"] = optimization_stub

from tabicl.train.run import Trainer


class DummyTrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.col_embedder = nn.Linear(1, 1)
        self.row_interactor = nn.Linear(1, 1)
        self.icl_predictor = nn.Linear(1, 1)


class FakeOptimizer:
    def __init__(self, params, lr=0.1):
        self.params = list(params)
        self.param_groups = [{"lr": lr}]
        self.step_calls = 0
        self.zero_grad_calls = 0

    def zero_grad(self, set_to_none=False):
        self.zero_grad_calls += 1
        for param in self.params:
            if param.grad is None:
                continue
            if set_to_none:
                param.grad = None
            else:
                param.grad.zero_()

    def step(self):
        self.step_calls += 1


class FakeScheduler:
    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1


class FakeScaler:
    def __init__(self):
        self.unscale_calls = 0
        self.step_calls = 0
        self.update_calls = 0

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        self.unscale_calls += 1

    def step(self, optimizer):
        self.step_calls += 1
        optimizer.step()

    def update(self):
        self.update_calls += 1


def _make_batch(batch_size=2, seq_len=4, train_size=2, features=3):
    return (
        torch.randn(batch_size, seq_len, features),
        torch.zeros(batch_size, seq_len, dtype=torch.long),
        torch.full((batch_size,), features, dtype=torch.long),
        torch.full((batch_size,), seq_len, dtype=torch.long),
        torch.full((batch_size,), train_size, dtype=torch.long),
    )


def _make_trainer(micro_batch_size=1):
    trainer = Trainer.__new__(Trainer)
    trainer.config = SimpleNamespace(micro_batch_size=micro_batch_size, gradient_clipping=1.0)
    trainer.model = DummyTrainModel()
    trainer.raw_model = trainer.model
    trainer.optimizer = FakeOptimizer(trainer.model.parameters())
    trainer.scheduler = FakeScheduler()
    trainer.scaler = FakeScaler()
    trainer.ddp = False
    trainer.master_process = False
    trainer.curr_step = 0
    trainer.amp_ctx = torch.no_grad()
    return trainer


def test_run_batch_skips_oom_without_unscale_or_step(monkeypatch):
    trainer = _make_trainer()
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    def always_oom(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("oom")

    trainer.run_micro_batch = always_oom

    results = trainer.run_batch(_make_batch())

    assert results["skipped"] is True
    assert results["skip_reason"] == "oom"
    assert results["successful_micro_batches"] == 0
    assert results["failed_micro_batches"] == 1
    assert trainer.scaler.unscale_calls == 0
    assert trainer.scaler.step_calls == 0
    assert trainer.scaler.update_calls == 0
    assert trainer.scheduler.step_calls == 0
    assert trainer.optimizer.step_calls == 0


def test_run_batch_clears_partial_grads_and_skips_step_after_oom(monkeypatch):
    trainer = _make_trainer()
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    calls = {"count": 0}

    def success_then_oom(*args, **kwargs):
        if calls["count"] == 0:
            calls["count"] += 1
            for param in trainer.model.parameters():
                param.grad = torch.ones_like(param)
            return {"ce": 0.5, "accuracy": 0.25}
        raise torch.cuda.OutOfMemoryError("oom")

    trainer.run_micro_batch = success_then_oom

    results = trainer.run_batch(_make_batch())

    assert results["skipped"] is True
    assert results["skip_reason"] == "oom"
    assert results["successful_micro_batches"] == 1
    assert results["failed_micro_batches"] == 1
    assert results["ce"] == 0.5
    assert results["accuracy"] == 0.25
    assert all(param.grad is None for param in trainer.model.parameters())
    assert trainer.scaler.unscale_calls == 0
    assert trainer.scaler.step_calls == 0
    assert trainer.scaler.update_calls == 0
    assert trainer.scheduler.step_calls == 0
    assert trainer.optimizer.step_calls == 0


def test_configure_amp_disables_grad_scaler_for_bfloat16(monkeypatch):
    trainer = Trainer.__new__(Trainer)
    trainer.config = SimpleNamespace(amp=True, device="cuda:0", dtype="bfloat16")
    trainer.master_process = False
    scaler_calls = []
    autocast_calls = []

    class CapturedScaler:
        pass

    def fake_grad_scaler(device, enabled):
        scaler_calls.append((device, enabled))
        return CapturedScaler()

    def fake_autocast(device_type, dtype):
        autocast_calls.append((device_type, dtype))
        return "autocast_ctx"

    monkeypatch.setattr(torch, "GradScaler", fake_grad_scaler)
    monkeypatch.setattr(torch, "autocast", fake_autocast)

    trainer.configure_amp()

    assert trainer.amp is True
    assert trainer.use_grad_scaler is False
    assert scaler_calls == [("cuda", False)]
    assert autocast_calls == [("cuda", torch.bfloat16)]
    assert trainer.amp_ctx == "autocast_ctx"
