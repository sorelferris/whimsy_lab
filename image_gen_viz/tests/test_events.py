import json

from image_gen_viz.events import GenerationEvent, format_sse, should_decode_step


def test_should_decode_step_uses_interval_and_always_includes_final():
    decoded = [
        step
        for step in range(1, 11)
        if should_decode_step(step=step, total_steps=10, decode_interval=4)
    ]

    assert decoded == [4, 8, 10]


def test_progress_event_serializes_to_json_payload():
    event = GenerationEvent.progress(run_id="abc", step=4, total_steps=10)

    assert event.type == "progress"
    assert event.payload == {"run_id": "abc", "step": 4, "total_steps": 10}


def test_frame_event_serializes_to_sse():
    event = GenerationEvent.frame(run_id="abc", step=4, url="/runs/abc/frames/step_0004.png", final=False)

    sse = format_sse(event)

    assert sse.startswith("event: frame\n")
    data = json.loads(sse.split("data: ", 1)[1])
    assert data == {"run_id": "abc", "step": 4, "url": "/runs/abc/frames/step_0004.png", "final": False}
