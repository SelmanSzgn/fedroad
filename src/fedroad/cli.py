import logging
from pathlib import Path

import typer

from .config import load_cfg
from .sim import run

app = typer.Typer(add_completion=False)


@app.command()
def main(
    cfg: Path = Path("configs/default.yaml"),
    out: Path = Path("runs/last"),
):
    """Run a federated learning simulation."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    path = run(load_cfg(cfg), out)
    typer.echo(f"metrics saved to {path}")
