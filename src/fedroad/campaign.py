import itertools
import logging
from pathlib import Path

import typer
import yaml
from prefect import flow, task

from .analysis import load, plot, summarize
from .config import Cfg
from .sim import run
from .tracking import log_run


def expand(spec):
    """Yield (tag, overrides) for each grid point and seed."""
    grid = spec.get("grid", {})
    for vals in itertools.product(*grid.values()):
        ov = dict(zip(grid, vals, strict=True))
        tag = "_".join(f"{k}={v}" for k, v in ov.items()) or "base"
        for s in spec["seeds"]:
            yield tag, {**ov, "seed": s}


@task
def one(base, ov, tag, top, exp) -> None:
    """Run one simulation and log it to MLflow."""
    cfg = Cfg(**{**base, **ov})
    out = Path(top) / tag / f"seed{ov['seed']}"
    run(cfg, out)
    log_run(cfg, out, f"{tag}/seed{ov['seed']}", exp, {"cfg": tag})


@flow(name="campaign")
def campaign(spec_path: str, root: str = "runs/campaigns") -> str:
    """Run every (config, seed) of a campaign, then summarize."""
    spec = yaml.safe_load(Path(spec_path).read_text())
    base = yaml.safe_load(Path(spec["base"]).read_text())
    exp = spec["name"]
    top = Path(root) / exp
    # Sequential on purpose: random seeds are global state
    for tag, ov in expand(spec):
        one(base, ov, tag, str(top), exp)
    st = summarize(load(top))
    st.to_csv(top / "summary.csv")
    plot(top, top / "summary.png")
    return st.round(2).to_string()


def _main(
    spec: Path = Path("configs/campaign.yaml"),
    root: Path = Path("runs/campaigns"),
):
    """Run a campaign: a grid of configs x seeds."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    typer.echo(campaign(str(spec), str(root)))


def cli():
    typer.run(_main)
