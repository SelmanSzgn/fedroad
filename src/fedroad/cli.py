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
    track: bool = True,
    name: str | None = None,
):
    """Run a federated learning simulation."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    c = load_cfg(cfg)
    path = run(c, out)
    typer.echo(f"metrics saved to {path}")
    if track:
        from .tracking import log_run  # lazy: mlflow import is slow

        log_run(c, out, name)
        typer.echo("run logged to MLflow")
