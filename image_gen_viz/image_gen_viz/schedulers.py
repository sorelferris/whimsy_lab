from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler

SCHEDULER_NAMES = ["ddim", "euler", "euler_a", "dpmpp_2m"]

SCHEDULER_CLASSES = {
    "ddim": DDIMScheduler,
    "euler": EulerDiscreteScheduler,
    "euler_a": EulerAncestralDiscreteScheduler,
    "dpmpp_2m": DPMSolverMultistepScheduler,
}


def create_scheduler(name: str, current_scheduler: object) -> object:
    scheduler_class = SCHEDULER_CLASSES.get(name)
    if scheduler_class is None:
        raise ValueError(f"Unsupported scheduler: {name}")
    return scheduler_class.from_config(current_scheduler.config)
