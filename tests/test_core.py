"""Correctness tests runnable on CPU (no GPU / no mamba-ssm needed)."""
import os
import sys
import math
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mnemosyne import MnemosyneConfig, MnemosyneMAE, SSMBaseline
from mnemosyne.rcps import rc_input, rc_logits, COMPLEMENT_PERM
from mnemosyne.data.synthetic import make_mqar_batch, mqar_vocab_size, IGNORE


def small_cfg(**kw):
    base = dict(d_model=32, n_layers=2, d_state=8, expand=2, decoder_layers=1,
                mem_persistent=16, mem_registers=8, max_seq_len=64, dropout=0.0)
    base.update(kw)
    return MnemosyneConfig(**base)


def test_complement_perm_is_involution():
    p = COMPLEMENT_PERM
    assert torch.equal(p[p], torch.arange(len(p))), "complement must be its own inverse"


def test_forward_shapes_and_param_growth():
    cfg = small_cfg()
    model = MnemosyneMAE(cfg).eval()
    base = SSMBaseline(cfg).eval()
    ids = torch.randint(2, 6, (4, 48))
    assert model.logits(ids).shape == (4, 48, cfg.vocab_size)
    assert model.pooled_features(ids).shape == (4, cfg.d_model)
    # memory must add parameters over the pure-SSM baseline
    assert model.num_params() > base.num_params()
    print(f"params: mnemosyne={model.num_params():,}  ssm={base.num_params():,}")


def test_rc_equivariance_of_reconstruction():
    cfg = small_cfg(rc_equivariant=True)
    model = MnemosyneMAE(cfg).eval()
    ids = torch.randint(2, 6, (3, 40))
    lhs = model.logits(rc_input(ids))     # f_equiv(R x)
    rhs = rc_logits(model.logits(ids))    # S f_equiv(x)
    err = (lhs - rhs).abs().max().item()
    assert err < 1e-4, f"RC-equivariance broken: max err {err}"
    print(f"RC-equivariance max err = {err:.2e}")


def test_rc_invariance_of_pooled_features():
    cfg = small_cfg(rc_equivariant=True)
    model = MnemosyneMAE(cfg).eval()
    ids = torch.randint(2, 6, (3, 40))
    a = model.pooled_features(ids)
    b = model.pooled_features(rc_input(ids))
    err = (a - b).abs().max().item()
    assert err < 1e-4, f"RC-invariance broken: max err {err}"
    print(f"RC-invariance max err  = {err:.2e}")


def test_memory_receives_gradient():
    cfg = small_cfg()
    model = MnemosyneMAE(cfg).train()
    ids = torch.randint(2, 6, (4, 48))
    loss, acc = model.mae_loss(ids, mask_ratio=0.3)
    loss.backward()
    mem = model.encoder.layers[0].memory
    for name, p in [("M", mem.M), ("reg_q", mem.reg_q), ("log_beta", mem.log_beta),
                    ("W_q", mem.W_q.weight)]:
        assert p.grad is not None, f"no grad for {name}"
        assert p.grad.abs().sum().item() > 0, f"zero grad for {name} (memory inert)"
    print(f"mae loss={loss.item():.3f} acc={acc.item():.3f}; memory grads OK")


def test_energy_and_usage_run():
    cfg = small_cfg()
    model = MnemosyneMAE(cfg).eval()
    ids = torch.randint(2, 6, (2, 40))
    h = model.encode(ids)
    mem = model.encoder.layers[0].memory
    e = mem.energy(h)
    usage = mem.slot_usage(h)
    assert torch.isfinite(e) and usage.shape[0] == cfg.mem_slots_total
    assert abs(usage.sum().item() - 1.0) < 1e-3
    print(f"hopfield energy={e.item():.3f}; slots={usage.shape[0]}; usage sums to 1")


def test_mqar_generator():
    V = mqar_vocab_size(16)
    toks, tgts = make_mqar_batch(8, seq_len=32, n_pairs=4, n_queries=4,
                                 n_symbols=16, seed=0)
    assert toks.shape == (8, 32) and tgts.shape == (8, 32)
    assert (tgts != IGNORE).sum(1).float().mean().item() == 4.0  # 4 queries each
    assert toks.max().item() < V
    print(f"MQAR ok: vocab={V}, queries/seq=4")


if __name__ == "__main__":
    torch.manual_seed(0)
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}\n")
    print("ALL TESTS PASSED")
