import uvicorn

from image_gen_viz.web import create_app


app = create_app()


def main() -> None:
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
