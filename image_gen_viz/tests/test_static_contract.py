from pathlib import Path

STATIC_DIR = Path("image_gen_viz/static")


def test_index_contains_required_generation_fields():
    html = (STATIC_DIR / "index.html").read_text()

    for field in [
        "prompt",
        "negative_prompt",
        "seed",
        "steps",
        "guidance_scale",
        "width",
        "height",
        "scheduler",
        "decode_interval",
        "model_id",
    ]:
        assert f'name="{field}"' in html


def test_frontend_handles_sse_and_timeline_playback():
    script = (STATIC_DIR / "app.js").read_text()

    assert "new EventSource" in script
    assert "addFrame" in script
    assert "playTimeline" in script
    assert "loadRun" in script


def test_timeline_frames_are_rendered_without_html_injection():
    script = (STATIC_DIR / "app.js").read_text()

    assert "innerHTML" not in script
    assert 'document.createElement("img")' in script
    assert "textContent" in script


def test_frontend_handles_failed_requests_and_stale_streams():
    script = (STATIC_DIR / "app.js").read_text()

    assert "try {" in script
    assert "catch (error)" in script
    assert "finally" in script
    assert "closeEventSource" in script
    assert "encodeURIComponent" in script
    assert "Failed to connect to generation stream" in script
