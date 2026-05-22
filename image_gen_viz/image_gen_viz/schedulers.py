from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler

SCHEDULER_NAMES = ["ddim", "euler", "euler_a", "dpmpp_2m"]

SCHEDULER_CLASSES = {
    "ddim": DDIMScheduler,
    "euler": EulerDiscreteScheduler,
    "euler_a": EulerAncestralDiscreteScheduler,
    "dpmpp_2m": DPMSolverMultistepScheduler,
}


def _scheduler_config(current_scheduler: object) -> object:
    config = current_scheduler.config
    if isinstance(config, dict):
        return config
    # Start with class attributes
    class_attrs = {name: value for name, value in vars(config.__class__).items() if not name.startswith("_") and not callable(value)}
    # Update with instance attributes
    instance_attrs = {name: value for name, value in vars(config).items() if not name.startswith("_") and not callable(value)}
    class_attrs.update(instance_attrs)
    return class_attrs


def create_scheduler(name: str, current_scheduler: object) -> object:
    scheduler_class = SCHEDULER_CLASSES.get(name)
    if scheduler_class is None:
        raise ValueError(f"Unsupported scheduler: {name}")
    return scheduler_class.from_config(_scheduler_config(current_scheduler))
