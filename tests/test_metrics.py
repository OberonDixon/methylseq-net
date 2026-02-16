import pytest
import numpy as np
import torch

from methylseqnet.metrics import (
    PearsonAcrossPositions,
    PearsonAcrossTasks,
    CCCAcrossVariants,
)


def hand_pearson(x, y):
    """Pearson r between two 1D arrays."""
    xc = x - x.mean()
    yc = y - y.mean()
    return (xc * yc).sum() / (np.sqrt((xc**2).sum() * (yc**2).sum()) + 1e-8)


def test_pearson_across_positions_perfect():
    t = torch.tensor([[[10., 20., 30., 40., 50.]]])
    p = torch.tensor([[[12., 22., 32., 42., 52.]]])
    result = PearsonAcrossPositions()(t, p)
    assert abs(result.item() - 1.0) < 1e-6


def test_pearson_across_positions_negative():
    t = torch.tensor([[[10., 20., 30., 40., 50.]]])
    p = torch.tensor([[[50., 40., 30., 20., 10.]]])
    result = PearsonAcrossPositions()(t, p)
    assert abs(result.item() - (-1.0)) < 1e-6


def test_pearson_across_tasks_single_channel_nan():
    t = torch.tensor([[[10., 20., 30., 40., 50.]]])
    p = torch.tensor([[[12., 22., 32., 42., 52.]]])
    result = PearsonAcrossTasks()(t, p)
    assert torch.isnan(result)


def test_ccc_single_variant_nan():
    t = torch.tensor([[[10., 20., 30., 40., 50.]]])
    p = torch.tensor([[[12., 22., 32., 42., 52.]]])
    result = CCCAcrossVariants()(t, p)
    assert torch.isnan(result)


def test_all_below_threshold_nan():
    t = torch.ones(2, 3, 10) * 2.0
    p = torch.ones(2, 3, 10) * 3.0
    assert torch.isnan(PearsonAcrossPositions()(t, p))
    assert torch.isnan(PearsonAcrossTasks()(t, p))
    assert torch.isnan(CCCAcrossVariants()(t, p))


def test_hand_computable_2var_2ch_3pos():
    """All three metrics hand-computed for a small tensor."""
    t = torch.tensor([
        [[10., 20., 30.], [40., 50., 60.]],
        [[15., 25., 35.], [45., 55., 65.]],
    ])
    p = torch.tensor([
        [[11., 19., 31.], [39., 51., 59.]],
        [[16., 24., 36.], [44., 56., 64.]],
    ])
    tn = t.numpy()
    pn = p.numpy()

    # PearsonAcrossPositions: r per (variant, channel) row, then mean
    t_flat = tn.reshape(4, 3)
    p_flat = pn.reshape(4, 3)
    expected_pap = np.mean([hand_pearson(t_flat[i], p_flat[i]) for i in range(4)])
    assert abs(PearsonAcrossPositions()(t, p).item() - expected_pap) < 1e-5

    # PearsonAcrossTasks: r across channels at each (variant x position), variance-weighted
    t_task = tn.transpose(1, 0, 2).reshape(2, 6)
    p_task = pn.transpose(1, 0, 2).reshape(2, 6)
    rs = np.array([hand_pearson(t_task[:, j], p_task[:, j]) for j in range(6)])
    vs = np.array([t_task[:, j].var() for j in range(6)])
    expected_pat = (rs * vs).sum() / (vs.sum() + 1e-8)
    assert abs(PearsonAcrossTasks()(t, p).item() - expected_pat) < 1e-5

    # CCCAcrossVariants: CCC across variants per (channel x position) feature, then mean
    t_ccc = tn.reshape(2, 6)
    p_ccc = pn.reshape(2, 6)
    cccs = []
    for j in range(6):
        tj, pj = t_ccc[:, j], p_ccc[:, j]
        mx, my = tj.mean(), pj.mean()
        vx, vy = tj.var(), pj.var()
        sx, sy = np.sqrt(vx), np.sqrt(vy)
        rho = hand_pearson(tj, pj)
        cccs.append((2 * rho * sx * sy) / (vx + vy + (mx - my) ** 2 + 1e-8))
    expected_ccc = np.mean(cccs)
    assert abs(CCCAcrossVariants()(t, p).item() - expected_ccc) < 1e-5


def test_masking():
    """Positions below min_counts should be masked appropriately per metric."""
    t = torch.tensor([[[3., 20., 30.], [40., 2., 60.]]])
    p = torch.tensor([[[99., 19., 31.], [39., 99., 59.]]])

    # PearsonAcrossPositions: nan-masks individual positions per row
    # row0 (var0,ch0): active at pos 1,2
    r0 = hand_pearson(np.array([20., 30.]), np.array([19., 31.]))
    # row1 (var0,ch1): active at pos 0,2
    r1 = hand_pearson(np.array([40., 60.]), np.array([39., 59.]))
    expected_pap = np.mean([r0, r1])
    assert abs(PearsonAcrossPositions()(t, p).item() - expected_pap) < 1e-5

    # PearsonAcrossTasks: keeps position if *any* channel > threshold
    # All 3 positions active; correlates across 2 channels at each
    tn = t.numpy()
    pn = p.numpy()
    t_task = tn.transpose(1, 0, 2).reshape(2, 3)
    p_task = pn.transpose(1, 0, 2).reshape(2, 3)
    rs = np.array([hand_pearson(t_task[:, j], p_task[:, j]) for j in range(3)])
    vs = np.array([t_task[:, j].var() for j in range(3)])
    expected_pat = (rs * vs).sum() / (vs.sum() + 1e-8)
    assert abs(PearsonAcrossTasks()(t, p).item() - expected_pat) < 1e-5


def test_shuffled_tasks_sanity():
    """Shuffling channels should tank PearsonAcrossTasks but not necessarily PearsonAcrossPositions."""
    torch.manual_seed(789)
    t = torch.rand(2, 10, 50) * 50
    p_good = t * 0.9 + torch.randn_like(t) * 2
    idx = torch.randperm(10)
    p_shuffled = p_good[:, idx, :]

    pat_good = PearsonAcrossTasks()(t, p_good).item()
    pat_shuf = PearsonAcrossTasks()(t, p_shuffled).item()
    assert pat_good > 0.5, "Good predictions should have high cross-task correlation"
    assert pat_shuf < pat_good - 0.3, "Shuffled channels should substantially reduce cross-task correlation"


def test_random_smoke():
    """Smoke test at modest scale — no crashes and returns sensible values."""
    torch.manual_seed(123)
    t = torch.rand(2, 8, 40) * 50
    p = t * 0.7 + torch.randn_like(t) * 5

    for MetricCls in [PearsonAcrossPositions, PearsonAcrossTasks, CCCAcrossVariants]:
        result = MetricCls()(t, p)
        assert torch.isfinite(result), f"{MetricCls.__name__} returned non-finite value"
        assert result.item() > 0, f"{MetricCls.__name__} expected positive correlation on correlated data"


def test_genomic_scale_smoke():
    """Smoke test at realistic scale."""
    torch.manual_seed(456)
    t = torch.rand(2, 63, 896) * 50
    p = t * 0.8 + torch.randn_like(t) * 3

    for MetricCls in [PearsonAcrossPositions, PearsonAcrossTasks, CCCAcrossVariants]:
        result = MetricCls()(t, p)
        assert torch.isfinite(result), f"{MetricCls.__name__} returned non-finite at genomic scale"