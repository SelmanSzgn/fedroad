import csv
import os
from pathlib import Path

import mlflow

DEFAULT_URI = "sqlite:///mlflow.db"


def log_run(cfg, out, name=None, exp="fedroad", tags=None):
    """Log params, per-round metrics and files of a run to MLflow."""
    out = Path(out)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_URI))
    mlflow.set_experiment(exp)
    with mlflow.start_run(run_name=name):
        mlflow.set_tags(tags or {})
        mlflow.log_params(cfg.model_dump())
        with open(out / "metrics.csv") as f:
            for row in csv.DictReader(f):
                m = {k: float(v) for k, v in row.items() if v != ""}
                step = int(m.pop("round"))
                mlflow.log_metrics(m, step=step)
        mlflow.log_artifacts(str(out))
