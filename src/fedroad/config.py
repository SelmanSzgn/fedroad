from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Cfg(BaseModel):
    """Simulation config, validated at load time."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    seed: int = 42
    simulation_time_s: float = Field(gt=0)
    poisson_rate: float = Field(gt=0)
    road_length_m: float = Field(gt=0)
    min_speed_kph: float = Field(gt=0)
    max_speed_kph: float = Field(gt=0)
    round_duration_s: int = Field(gt=0)
    min_n_data: int = Field(gt=0)
    max_n_data: int = Field(gt=0)
    n_sub_classes: int = Field(ge=1, le=10)
    min_cpu_hertz: float = Field(gt=0)
    max_cpu_hertz: float = Field(gt=0)
    batch: int = Field(gt=0)
    n_local_epochs: int = Field(gt=0)
    n_cpu_cycles_per_data: float = Field(gt=0)
    effective_capacitance: float = Field(gt=0)
    snr_db_min: float
    snr_db_max: float
    bandwidth_hz: float = Field(gt=0)
    tx_power_w: float = Field(gt=0)
    model_precision: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    momentum: float = Field(ge=0, lt=1)
    weight_decay: float = Field(ge=0)
    data_path: str = "data/cifar10"

    @model_validator(mode="after")
    def check_ranges(self):
        pairs = [
            (self.min_speed_kph, self.max_speed_kph),
            (self.min_n_data, self.max_n_data),
            (self.min_cpu_hertz, self.max_cpu_hertz),
            (self.snr_db_min, self.snr_db_max),
        ]
        if any(lo > hi for lo, hi in pairs):
            raise ValueError("a min value is above its max value")
        return self


def load_cfg(path: str | Path) -> Cfg:
    """Load and validate a YAML config."""
    with open(path) as f:
        return Cfg(**yaml.safe_load(f))
