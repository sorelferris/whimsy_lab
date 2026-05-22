const form = document.querySelector("#generate-form");
const loadForm = document.querySelector("#load-form");
const errorBox = document.querySelector("#error");
const statusText = document.querySelector("#status");
const progress = document.querySelector("#progress");
const preview = document.querySelector("#preview");
const timeline = document.querySelector("#timeline");
const scrubber = document.querySelector("#scrubber");
const frameLabel = document.querySelector("#frame-label");
const playButton = document.querySelector("#play-button");
const generateButton = document.querySelector("#generate-button");

let frames = [];
let currentIndex = 0;
let eventSource = null;
let playbackTimer = null;

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function clearError() {
  errorBox.textContent = "";
  errorBox.hidden = true;
}

function collectRequest() {
  const data = new FormData(form);
  return {
    prompt: data.get("prompt"),
    negative_prompt: data.get("negative_prompt"),
    seed: Number(data.get("seed")),
    steps: Number(data.get("steps")),
    guidance_scale: Number(data.get("guidance_scale")),
    width: Number(data.get("width")),
    height: Number(data.get("height")),
    scheduler: data.get("scheduler"),
    decode_interval: Number(data.get("decode_interval")),
    model_id: data.get("model_id"),
  };
}

function closeEventSource() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
}

function resetTimeline() {
  frames = [];
  currentIndex = 0;
  timeline.replaceChildren();
  preview.removeAttribute("src");
  scrubber.value = "0";
  scrubber.max = "0";
  frameLabel.textContent = "No frames";
}

function addFrame(frame) {
  frames.push(frame);
  const index = frames.length - 1;
  const button = document.createElement("button");
  button.type = "button";
  button.className = "frame-thumb";
  const image = document.createElement("img");
  image.src = frame.url;
  image.alt = `Step ${frame.step}`;
  const label = document.createElement("span");
  label.textContent = `step ${frame.step}${frame.final ? " final" : ""}`;
  button.append(image, label);
  button.addEventListener("click", () => selectFrame(index));
  timeline.appendChild(button);
  scrubber.max = String(frames.length - 1);
  selectFrame(index);
}

function selectFrame(index) {
  if (frames.length === 0) return;
  currentIndex = Math.max(0, Math.min(index, frames.length - 1));
  preview.src = frames[currentIndex].url;
  scrubber.value = String(currentIndex);
  frameLabel.textContent = `step ${frames[currentIndex].step}`;
  [...timeline.children].forEach((child, childIndex) => {
    child.classList.toggle("active", childIndex === currentIndex);
  });
}

function subscribe(runId, totalSteps) {
  closeEventSource();
  eventSource = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`);
  eventSource.addEventListener("started", () => {
    statusText.textContent = `Running ${runId}`;
    progress.value = 0;
  });
  eventSource.addEventListener("progress", (message) => {
    const event = JSON.parse(message.data);
    progress.value = Math.round((event.step / event.total_steps) * 100);
  });
  eventSource.addEventListener("frame", (message) => {
    addFrame(JSON.parse(message.data));
  });
  eventSource.addEventListener("complete", () => {
    progress.value = 100;
    statusText.textContent = `Complete ${runId}`;
    generateButton.disabled = false;
    closeEventSource();
  });
  eventSource.addEventListener("error", (message) => {
    let errorMessage = "Failed to connect to generation stream";
    if (message.data) {
      errorMessage = JSON.parse(message.data).message;
    }
    showError(errorMessage);
    statusText.textContent = "Error";
    generateButton.disabled = false;
    closeEventSource();
  });
}

async function loadRun(runId) {
  closeEventSource();
  const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
  if (!response.ok) throw new Error(`Run not found: ${runId}`);
  const metadata = await response.json();
  resetTimeline();
  metadata.frames.forEach(addFrame);
  statusText.textContent = `${metadata.status} ${runId}`;
  progress.value = metadata.status === "completed" ? 100 : 0;
  if (metadata.error) showError(metadata.error);
}

function playTimeline() {
  if (playbackTimer) {
    clearInterval(playbackTimer);
    playbackTimer = null;
    playButton.textContent = "Play";
    return;
  }
  playButton.textContent = "Pause";
  playbackTimer = setInterval(() => {
    if (frames.length === 0 || currentIndex >= frames.length - 1) {
      clearInterval(playbackTimer);
      playbackTimer = null;
      playButton.textContent = "Play";
      return;
    }
    selectFrame(currentIndex + 1);
  }, 350);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  resetTimeline();
  generateButton.disabled = true;
  const request = collectRequest();
  try {
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const payload = await response.json();
      showError(payload.detail || "Generation failed");
      return;
    }
    const payload = await response.json();
    subscribe(payload.run_id, request.steps);
  } catch (error) {
    showError(error.message);
  } finally {
    if (!eventSource) {
      generateButton.disabled = false;
    }
  }
});

loadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  const runId = new FormData(loadForm).get("run_id");
  try {
    await loadRun(runId);
  } catch (error) {
    showError(error.message);
  }
});

scrubber.addEventListener("input", () => selectFrame(Number(scrubber.value)));
playButton.addEventListener("click", playTimeline);
