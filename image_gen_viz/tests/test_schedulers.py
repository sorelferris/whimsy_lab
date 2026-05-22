import pytest

from image_gen_viz.schedulers import SCHEDULER_NAMES, create_scheduler


class DummyConfig:
    prediction_type = "epsilon"


class DummyScheduler:
    config = {
        "prediction_type": "epsilon"
    }


def test_supported_scheduler_names_are_stable():
    assert SCHEDULER_NAMES == ["ddim", "euler", "euler_a", "dpmpp_2m"]


def test_create_scheduler_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unsupported scheduler"):
        create_scheduler("unknown", DummyScheduler())


def test_create_scheduler_returns_new_scheduler_instance():
    scheduler = create_scheduler("euler", DummyScheduler())

    assert scheduler.__class__.__name__ == "EulerDiscreteScheduler"
