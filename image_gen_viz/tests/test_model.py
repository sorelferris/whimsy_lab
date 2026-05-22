from PIL import Image

from image_gen_viz.model import DecodedFrame, StableDiffusionModel
from image_gen_viz.validation import GenerationRequest


class FakePipeline:
    def __init__(self):
        self.scheduler = object()
        self.calls = []

    def to(self, device):
        self.device = device
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        callback = kwargs["callback_on_step_end"]
        total_steps = kwargs["num_inference_steps"]
        for zero_based_step in range(total_steps):
            callback(self, zero_based_step, None, {"latents": f"latent-{zero_based_step + 1}"})
        return type("Result", (), {"images": [Image.new("RGB", (8, 8), "green")]})()


def test_model_emits_decoded_frames_at_interval(monkeypatch):
    fake_pipeline = FakePipeline()
    monkeypatch.setattr("image_gen_viz.model.load_pipeline", lambda model_id: fake_pipeline)
    monkeypatch.setattr("image_gen_viz.model.create_scheduler", lambda name, current: current)
    model = StableDiffusionModel(device="cuda")
    request = GenerationRequest(prompt="a fox", steps=5, decode_interval=2, width=512, height=512)
    frames = []

    final_image = model.generate(request, on_frame=frames.append)

    assert isinstance(final_image, Image.Image)
    assert [frame.step for frame in frames] == [2, 4, 5]
    assert all(isinstance(frame.image, Image.Image) for frame in frames)
    assert fake_pipeline.calls[0]["prompt"] == "a fox"
    assert fake_pipeline.calls[0]["num_inference_steps"] == 5


def test_decoded_frame_records_step_and_image():
    image = Image.new("RGB", (8, 8), "white")
    frame = DecodedFrame(step=3, image=image, final=False)

    assert frame.step == 3
    assert frame.image is image
    assert frame.final is False
