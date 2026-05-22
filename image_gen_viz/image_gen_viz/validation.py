from typing import Literal

from pydantic import BaseModel, Field, model_validator

SchedulerName = Literal["ddim", "euler", "euler_a", "dpmpp_2m"]


class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    negative_prompt: str = Field(default="", max_length=1000)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    steps: int = Field(default=20, ge=1, le=150)
    guidance_scale: float = Field(default=7.5, ge=0.0, le=30.0)
    width: int = Field(default=512, ge=128, le=1024)
    height: int = Field(default=512, ge=128, le=1024)
    scheduler: SchedulerName = "euler"
    decode_interval: int = Field(default=4, ge=1)
    model_id: str = Field(default="runwayml/stable-diffusion-v1-5", min_length=1, max_length=300)

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "GenerationRequest":
        if not self.prompt.strip():
            raise ValueError("prompt cannot be blank")
        if self.width % 8 != 0 or self.height % 8 != 0:
            raise ValueError("width and height must be a multiple of 8")
        if self.decode_interval > self.steps:
            raise ValueError("decode_interval cannot exceed steps")
        return self
