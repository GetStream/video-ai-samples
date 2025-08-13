### AI Security Cam (Stream + YOLO + Roboflow Supervision)

This example turns a Stream video call into a simple “AI security cam” which tries to detect if delivered packages are possibly missing. It subscribes to an incoming video track, runs YOLO object detection and ByteTrack tracking, annotates the frames, and publishes the processed video back into the same call.

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or `pip`
- YOLO model custom trained to detect "packages"

### Stream environment

Create a `.env` in `security_cam/` with at least:

```bash
STREAM_API_KEY=your_stream_api_key
STREAM_API_SECRET=your_stream_api_secret

# URL of a simple web client that can join a Stream call and publish a camera track.
# You can point this to your own example app. The script opens: ${EXAMPLE_BASE_URL}/join/<call_id>?...
EXAMPLE_BASE_URL=https://your-web-client.example.com
```

### Model weights

The script loads YOLO weights from `yolo_custom_weights.pt`:

```python
YOLO("yolo_custom_weights.pt", task="detect")
```

If your weights file is named `weights.pt` (included here), either:

- Rename the file to `yolo_custom_weights.pt`, or
- Change the filename in `main.py` (line near the YOLO constructor) to `weights.pt`.

### Install dependencies

Using uv (recommended):

```bash
cd security_cam
uv sync
```

Or using pip (in a virtualenv):

```bash
cd security_cam
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Run

Without an input file (publish from your browser/client as the camera user):

```bash
uv run python main.py
```

This will:
- Create a Stream call and an AI bot that publishes the processed video.
- Open your browser to `EXAMPLE_BASE_URL` with a token for the `cam-*` user.
- From the web client, allow camera access to start sending a video track.

With an input video file (the script publishes the file as the camera):

```bash
uv run python main.py -i /absolute/path/to/video.mp4
```

This will:
- Publish the specified file as the camera track using `aiortc.MediaPlayer`.
- Open the browser with a token for a `viewer-*` user so you can watch the processed stream.

Enable debug frame dumps to `security_cam/debug/`:

```bash
uv run python main.py -d
```

### How it works (high level)

1. Creates a Stream call and joins two peers programmatically: `cam-*` (publisher) and `ai-*` (processor). Optionally a `viewer-*` joins when `-i` is used.
2. When a video track is detected for the camera user, frames are sent to an analyzer coroutine.
3. YOLO runs detection; `supervision`'s ByteTrack provides tracking IDs and annotators draw boxes/labels.
4. A polygon zone (bottom strip of a 1280×720 frame) is defined; its annotation and a missing-items banner are rendered when applicable.
5. Annotated frames are republished by a custom `aiortc.VideoStreamTrack` back into the call.

Notes:
- The ROI polygon in `main.py` is hard-coded for 1280×720. For other resolutions, adjust the `roi` coordinates.
- `-d/--debug` dumps input and output frames to `security_cam/debug/` for inspection.
