# Diffusion Visualization Design

## Goal

Build a local interactive web application that visualizes how noise becomes an image during real Stable Diffusion 1.5 text-to-image generation.

The first version targets a local GPU environment. It should expose advanced generation parameters, run a real SD1.5-compatible model, decode intermediate latents at a user-configurable interval, and let the user inspect the resulting frame sequence.

## Product scope

The application opens in a browser. The user fills an advanced parameter panel, clicks Generate, watches progress and newly decoded frames appear, then scrubs or plays a timeline after generation completes.

In scope for the first version:

- Local browser UI.
- Real Stable Diffusion 1.5-compatible text-to-image inference.
- Local GPU execution as the primary target.
- Advanced controls for prompt, negative prompt, seed, steps, guidance scale, resolution, scheduler, and decode interval.
- Intermediate frame capture based on decode interval, always including the final image.
- Current-frame preview, progress display, timeline thumbnails, timeline scrubbing, and playback.
- Local run storage with parameters, intermediate frames, final image, and metadata.
- Reloading a previous run by run id.
- Clear UI errors for invalid parameters, model load failure, out-of-memory failures, and interrupted generation.

Out of scope for the first version:

- Accounts, authentication, cloud sharing, or remote hosting.
- Multi-user or multi-GPU task scheduling.
- Model training or fine-tuning.
- Complex image editing workflows.
- Automatic CPU fallback after GPU failure.

## Recommended interface

Use a console-and-timeline layout:

- Left panel: advanced generation controls.
- Right panel: current decoded frame, progress/status, timeline thumbnails, and playback controls.

This layout keeps parameter control visible while making the denoising sequence the main visual object.

## Architecture

Use a small Python web application with separated UI, task, and model boundaries.

Recommended stack:

- FastAPI backend for parameter submission, run lookup, static frame serving, and progress streaming.
- A simple custom frontend for the control panel and timeline UI.
- `diffusers` model layer for Stable Diffusion 1.5-compatible inference.
- Server-Sent Events for generation progress because the first version only needs server-to-client updates.

Core units:

- Web/API layer: validates request shape, starts generation, streams run events, and serves saved run artifacts.
- Generation task layer: owns one active GPU generation task, run lifecycle, frame persistence, and progress event emission.
- Model layer: loads the SD1.5 pipeline, applies scheduler/seed/parameter choices, performs denoising, and exposes intermediate latent checkpoints for decoding.
- Storage layer: creates per-run directories and writes parameters, metadata, intermediate frames, and final image.
- Frontend layer: renders parameters, progress, current preview, thumbnails, scrubbing, playback, and error states.

The first version should allow only one active generation task to avoid GPU memory contention.

## Data flow

1. The frontend submits generation parameters and creates a run.
2. The backend validates parameter bounds such as steps, decode interval, resolution, seed, and scheduler name.
3. The generation task initializes the model pipeline, seed, and scheduler.
4. After each denoising step, the task checks whether the step should be decoded based on `decode_interval`, and also decodes the final step.
5. Selected latents are decoded through the VAE into preview images and saved under the run directory.
6. The task emits progress events containing status, step index, frame path when present, and error information when relevant.
7. The frontend updates the progress indicator, current image, and timeline as events arrive.
8. On completion, metadata and final output are available for reload by run id.

## Decode interval behavior

`decode_interval` is a user-facing advanced parameter. It controls how frequently intermediate latents are decoded into visible frames.

Expected behavior:

- A lower interval saves more frames and gives smoother visualization at higher cost.
- A higher interval saves fewer frames and runs faster.
- The final image is always saved even when the final step does not align with the interval.
- Invalid values such as zero, negative numbers, or values larger than the number of steps are rejected or normalized through explicit validation rules chosen during implementation planning.

## Error handling

The backend should return structured errors for:

- Invalid generation parameters.
- Model loading failure.
- CUDA or GPU out-of-memory failure.
- Generation interruption or unexpected task failure.

The frontend should display errors above the parameter panel and keep already generated frames browsable when a partial run exists. GPU out-of-memory errors should not trigger automatic CPU fallback; the UI should suggest lowering resolution, step count, or closing other GPU workloads.

## Testing strategy

Unit tests should cover logic that does not require loading a full model:

- Parameter validation.
- Scheduler name mapping.
- Run directory and metadata creation.
- Decode interval frame-selection logic.
- Event payload shape for progress, frame, complete, and error states.

Manual or integration validation should run on a GPU machine with a small SD1.5 generation, such as 512x512, 10-20 steps, and decode interval 2-5. The validation should confirm that progress streams to the UI, intermediate frames appear during generation, the final image is saved, and the completed timeline can be scrubbed and played.

## Acceptance criteria

- The app starts locally and opens in a browser.
- A Stable Diffusion 1.5-compatible model can complete a text-to-image generation on a local GPU.
- The advanced parameter panel includes `decode_interval`.
- Changing `decode_interval` changes how many intermediate frames are saved.
- During generation, the UI shows progress and newly decoded frames.
- After generation, the UI supports timeline scrubbing and playback.
- Invalid parameters, model loading failure, GPU out-of-memory, and interrupted generation show understandable errors.
- Run artifacts are saved locally and can be viewed again by run id.
