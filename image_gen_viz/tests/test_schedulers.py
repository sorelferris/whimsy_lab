import pytest
from diffusers import EulerDiscreteScheduler

from image_gen_viz.schedulers import SCHEDULER_NAMES, create_scheduler


class DummyScheduler:
    def __init__(self):
        # Use a real scheduler's config dict
        real_scheduler = EulerDiscreteScheduler()
        self.config = real_scheduler.config


def test_supported_scheduler_names_are_stable():
    assert SCHEDULER_NAMES == ["ddim", "euler", "euler_a", "dpmpp_2m"]


def test_create_scheduler_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unsupported scheduler"):
        create_scheduler("unknown", DummyScheduler())


def test_create_scheduler_returns_new_scheduler_instance():
    scheduler = create_scheduler("euler", DummyScheduler())

    assert scheduler.__class__.__name__ == "EulerDiscreteScheduler"
