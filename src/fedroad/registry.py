import os
import shutil
import tempfile
from pathlib import Path

import mlflow
import torch
import typer
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from .model import Model
from .tracking import DEFAULT_URI

NAME = "fedroad-global"


def export_onnx(pt, dst):
    """Convert a saved state dict to ONNX (dynamic batch size)."""
    model = Model()
    model.load_state_dict(torch.load(pt, map_location="cpu"))
    model.eval()
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 32, 32),
        str(dst),
        input_names=["x"],
        output_names=["logits"],
        dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )


def promote(exp="fedroad", name=NAME, alias="champion", out="models"):
    """Register the best run's model and point `alias` to it.

    Also writes the ONNX file to `out` for serving.
    """
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_URI))
    runs = mlflow.search_runs(
        experiment_names=[exp],
        order_by=["metrics.acc DESC"],
        max_results=1,
        output_format="list",
    )
    if not runs:
        raise ValueError(f"no run in experiment {exp!r}")
    run = runs[0]
    rid, acc = run.info.run_id, run.data.metrics["acc"]
    cl = MlflowClient()
    with tempfile.TemporaryDirectory() as tmp:
        pt = mlflow.artifacts.download_artifacts(
            run_id=rid, artifact_path="model.pt", dst_path=tmp
        )
        onnx = Path(tmp) / "model.onnx"
        export_onnx(pt, onnx)
        cl.log_artifact(rid, str(onnx))
        Path(out).mkdir(parents=True, exist_ok=True)
        shutil.copy(onnx, Path(out) / "model.onnx")
    try:
        cl.create_registered_model(name)
    except MlflowException:
        pass  # already exists
    src = f"{run.info.artifact_uri}/model.onnx"
    mv = cl.create_model_version(name, src, run_id=rid)
    cl.set_model_version_tag(name, mv.version, "acc", f"{acc:.2f}")
    cl.set_registered_model_alias(name, alias, mv.version)
    return mv.version, acc


def _main(
    exp: str = "fedroad",
    out: Path = Path("models"),
):
    """Promote the best model of an experiment and export it as ONNX."""
    v, acc = promote(exp, out=out)
    typer.echo(f"{NAME} v{v} (acc {acc:.2f} %) is now champion")
    typer.echo(f"ONNX model written to {out / 'model.onnx'}")


def cli():
    typer.run(_main)
