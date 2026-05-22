import pytest

from image_gen_viz.schedulers import SCHEDULER_NAMES, create_scheduler, _scheduler_config


class DummyConfig:
    prediction_type = "epsilon"


class DummyScheduler:
    config = DummyConfig()


def test_supported_scheduler_names_are_stable():
    assert SCHEDULER_NAMES == ["ddim", "euler", "euler_a", "dpmpp_2m"]


def test_create_scheduler_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unsupported scheduler"):
        create_scheduler("unknown", DummyScheduler())


def test_create_scheduler_returns_new_scheduler_instance():
    scheduler = create_scheduler("euler", DummyScheduler())

    assert scheduler.__class__.__name__ == "EulerDiscreteScheduler"


def test_scheduler_config_preserves_instance_attributes():
    class InstanceConfig:
        prediction_type = "epsilon"

        def __init__(self):
            self.num_train_timesteps = 500
            self.prediction_type = "v_prediction"

    class InstanceScheduler:
        config = InstanceConfig()

    config = _scheduler_config(InstanceScheduler())

    assert config["num_train_timesteps"] == 500
    assert config["prediction_type"] == "v_prediction"
