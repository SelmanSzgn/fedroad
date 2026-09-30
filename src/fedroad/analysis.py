from pathlib import Path

import pandas as pd


def load(root):
    """One row per run: final acc, drop rate (%), total energy."""
    rows = []
    for p in sorted(Path(root).glob("*/seed*/metrics.csv")):
        d = pd.read_csv(p)
        rows.append(
            dict(
                cfg=p.parents[1].name,
                seed=p.parent.name,
                acc=d["acc"].tail(5).mean(),
                drop=100 * d["drop"].sum() / max(d["active"].sum(), 1),
                energy=d["energy_j"].sum(),
            )
        )
    return pd.DataFrame(rows)


def summarize(df):
    """Mean and std over seeds, per config."""
    return df.groupby("cfg")[["acc", "drop", "energy"]].agg(["mean", "std"])


def curves(root):
    """Accuracy per round: {cfg: DataFrame (rounds x seeds)}."""
    out = {}
    for p in sorted(Path(root).glob("*/seed*/metrics.csv")):
        s = pd.read_csv(p).set_index("round")["acc"]
        out.setdefault(p.parents[1].name, []).append(s)
    return {k: pd.concat(v, axis=1) for k, v in out.items()}


def plot(root, path):
    """Accuracy curves (mean +- std), drop rate and energy per config."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st = summarize(load(root))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for k, c in curves(root).items():
        m, s = c.mean(axis=1), c.std(axis=1)
        ax[0].plot(m.index, m, label=k)
        ax[0].fill_between(m.index, m - s, m + s, alpha=0.2)
    ax[0].set(xlabel="round", ylabel="test acc (%)", title="Accuracy")
    ax[0].legend()
    bars = [
        (ax[1], "drop", "Dropped clients (%)"),
        (ax[2], "energy", "Energy (J)"),
    ]
    for a, col, title in bars:
        a.bar(st.index, st[col]["mean"], yerr=st[col]["std"], capsize=4)
        a.set_title(title)
        a.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
