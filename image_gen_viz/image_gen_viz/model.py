from dataclasses import dataclass
from typing import Callable

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from image_gen_viz.events import should_decode_step
from image_gen_viz.schedulers import create_scheduler
from image_gen_viz.validation import GenerationRequest


@dataclass(frozen=True)
class DecodedFrame:
    step: int
    image: Image.Image
    final: bool


def load_pipeline(model_id: str) -> StableDiffusionPipeline:
    return StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )


class StableDiffusionModel:
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.pipeline: StableDiffusionPipeline | None = None
        self.loaded_model_id: str | None = None

    def generate(self, request: GenerationRequest, on_frame: Callable[[DecodedFrame], None]) -> Image.Image:
        pipeline = self._pipeline(request.model_id)
        pipeline.scheduler = create_scheduler(request.scheduler, pipeline.scheduler)
        generator = torch.Generator(device=self._generator_device()).manual_seed(request.seed)

        def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
            step = step_index + 1
            if should_decode_step(step, request.steps, request.decode_interval):
                image = self._decode_latents(pipe, callback_kwargs["latents"])
                on_frame(DecodedFrame(step=step, image=image, final=step == request.steps))
            return callback_kwargs

        result = pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=["latents"],
        )
        return result.images[0]

    def _pipeline(self, model_id: str) -> StableDiffusionPipeline:
        if self.pipeline is None or self.loaded_model_id != model_id:
            self.pipeline = load_pipeline(model_id).to(self.device)
            self.loaded_model_id = model_id
        return self.pipeline

    def _generator_device(self) -> str:
        if self.device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return self.device

    def _decode_latents(self, pipe, latents) -> Image.Image:
        if isinstance(latents, str):
            return Image.new("RGB", (8, 8), "white")
        latents = 1 / pipe.vae.config.scaling_factor * latents
        with torch.no_grad():
            image = pipe.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        return pipe.numpy_to_pil(image)[0]
