from fastapi import FastAPI

from image_gen_viz.web import create_app


def test_create_app_returns_fastapi_app():
    app = create_app()
    assert isinstance(app, FastAPI)
