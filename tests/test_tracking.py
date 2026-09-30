import mlflow

from fedroad.config import load_cfg
from fedroad.tracking import log_run
from test_config import CFG


def test_log_run(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path}/m.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "metrics.csv").write_text(
        "round,acc,cacc_mean\n1,10.0,\n2,20.0,15.0\n"
    )
    log_run(load_cfg(CFG), out, name="t")
    runs = mlflow.search_runs(
        experiment_names=["fedroad"], output_format="list"
    )
    assert runs[0].data.metrics["acc"] == 20.0
    assert runs[0].data.metrics["cacc_mean"] == 15.0
    assert runs[0].data.params["seed"] == "42"
