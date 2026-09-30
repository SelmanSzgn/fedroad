from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from fedroad.config import Cfg, load_cfg

CFG = Path(__file__).parents[1] / "configs" / "default.yaml"


def base():
    with open(CFG) as f:
        return yaml.safe_load(f)


def test_default_cfg_loads():
    assert load_cfg(CFG).seed == 42


def test_unknown_key_is_rejected():
    d = base()
    d["typo_key"] = 1
    with pytest.raises(ValidationError):
        Cfg(**d)


def test_min_above_max_is_rejected():
    d = base()
    d["min_speed_kph"] = 100
    with pytest.raises(ValidationError):
        Cfg(**d)


def test_n_sub_classes_out_of_range():
    d = base()
    d["n_sub_classes"] = 11
    with pytest.raises(ValidationError):
        Cfg(**d)
