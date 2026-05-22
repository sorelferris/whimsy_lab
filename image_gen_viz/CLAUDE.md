# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`image-gen-viz` is a Python 3.12 local web app for visualizing Stable Diffusion denoising. A FastAPI backend validates generation parameters, runs one local GPU generation task at a time, stores per-run frames and metadata, and streams progress with Server-Sent Events. A static HTML/CSS/JavaScript frontend provides an advanced parameter console, live preview, and timeline playback.

## Commands

- Install dependencies:
  ```bash
  uv sync --group dev
  ```

- Run the application:
  ```bash
  uv run python main.py
  ```

- Run all tests:
  ```bash
  uv run pytest
  ```

- Run a focused test file:
  ```bash
  uv run pytest tests/test_tasks.py -v
  ```

- Confirm the configured Python version:
  ```bash
  uv run python --version
  ```

There are no configured lint, format, type-check, or build commands. Do not assume `ruff`, `mypy`, or packaging commands are available unless they are added first.

## Architecture

- `main.py` exposes the Uvicorn entrypoint and imports `image_gen_viz.web:create_app()`.
- `image_gen_viz/web.py` builds the FastAPI app, mounts static assets and run artifacts, exposes run APIs, and serves the SSE event stream.
- `image_gen_viz/validation.py` defines the Pydantic generation request model and parameter bounds.
- `image_gen_viz/storage.py` owns run directories, metadata, frame URLs, final image state, and run-id validation.
- `image_gen_viz/events.py` defines generation event payloads and Server-Sent Events formatting.
- `image_gen_viz/model.py` wraps `diffusers` Stable Diffusion pipeline loading, scheduler selection, step progress callbacks, and latent decoding.
- `image_gen_viz/tasks.py` manages the single active generation task, event queues, frame persistence, and user-facing generation errors.
- `image_gen_viz/static/` contains the frontend console, timeline, playback, and run reload UI.
- `tests/` contains pytest coverage for validation, storage, scheduler mapping, model callbacks, task events, API behavior, and static frontend contracts.

Generated run artifacts are written under `runs/` and should not be committed.
