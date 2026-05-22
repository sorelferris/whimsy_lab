from fastapi import FastAPI


def create_app() -> FastAPI:
    return FastAPI(title="Image Gen Viz")
