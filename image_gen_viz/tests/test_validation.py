import pytest
from pydantic import ValidationError

from image_gen_viz.validation import GenerationRequest


def test_generation_request_accepts_valid_advanced_parameters():
    request = GenerationRequest(
        prompt="a luminous castle in the clouds",
        negative_prompt="blurry",
        seed=123,
        steps=20,
        guidance_scale=7.5,
        width=512,
        height=512,
        scheduler="euler",
        decode_interval=4,
    )

    assert request.prompt == "a luminous castle in the clouds"
    assert request.negative_prompt == "blurry"
    assert request.seed == 123
    assert request.decode_interval == 4


def test_decode_interval_must_be_positive():
    with pytest.raises(ValidationError, match="decode_interval"):
        GenerationRequest(prompt="x", steps=20, decode_interval=0)


def test_decode_interval_cannot_exceed_steps():
    with pytest.raises(ValidationError, match="decode_interval"):
        GenerationRequest(prompt="x", steps=20, decode_interval=21)


def test_resolution_must_be_multiple_of_eight():
    with pytest.raises(ValidationError, match="multiple of 8"):
        GenerationRequest(prompt="x", width=510, height=512)


def test_prompt_cannot_be_blank():
    with pytest.raises(ValidationError, match="prompt"):
        GenerationRequest(prompt="   ")
