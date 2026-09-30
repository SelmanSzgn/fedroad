import pandas as pd
import pytest

from fedroad.analysis import load, summarize
from fedroad.campaign import expand

SPEC = dict(seeds=[0, 1], grid=dict(a=[1, 2], b=[5, 6]))


def test_expand_grid_times_seeds():
    runs = list(expand(SPEC))
    assert len(runs) == 8
    tags = {t for t, _ in runs}
    assert tags == {"a=1_b=5", "a=1_b=6", "a=2_b=5", "a=2_b=6"}
    assert all(set(ov) == {"a", "b", "seed"} for _, ov in runs)


def test_expand_without_grid():
    assert list(expand(dict(seeds=[3]))) == [("base", {"seed": 3})]


def write(root, cfg, seed, drops):
    d = root / cfg / f"seed{seed}"
    d.mkdir(parents=True)
    pd.DataFrame(
        dict(
            round=[1, 2],
            acc=[10.0, 20.0],
            active=[4, 6],
            drop=drops,
            energy_j=[1.0, 2.0],
        )
    ).to_csv(d / "metrics.csv", index=False)


def test_load_and_summarize(tmp_path):
    write(tmp_path, "x", 0, [1, 1])  # 2 / 10 = 20 %
    write(tmp_path, "x", 1, [2, 3])  # 5 / 10 = 50 %
    df = load(tmp_path)
    assert len(df) == 2
    s = summarize(df)
    assert s.loc["x", ("drop", "mean")] == pytest.approx(35.0)
    assert s.loc["x", ("acc", "mean")] == pytest.approx(15.0)
    assert s.loc["x", ("energy", "mean")] == pytest.approx(3.0)
