# Live Sports Coach AI

A real-time AI-powered sports coaching app that uses video analysis and voice feedback to provide personalized coaching for mini-golf (with potential for other sports by modifying the prompt). The application leverages Google's Gemini Live AI model and Stream's WebRTC platform to deliver interactive coaching sessions.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Google AI API key
- Stream API credentials

### Installation

1. Clone the repository and navigate to the project directory:
```bash
cd live_sports_coach
```

2. Install dependencies:
```bash
uv sync
```

### Environment Setup

Create a `.env` file in the project root with your API credentials:

```env
GOOGLE_API_KEY=your_google_ai_api_key_here
STREAM_API_KEY=your_stream_api_key_here
STREAM_API_SECRET=your_stream_api_secret_here
```

### Configuration

The AI coach is currently configured for **mini-golf coaching** with the following characteristics:
- Analyzes player posture, stance, swing, and overall form
- Provides proactive coaching feedback
- Uses a veteran coach personality with sarcastic elements
- Focuses on technique improvement

To adapt for other sports, modify the `PROMPT` variable in `main.py`.

## 🎮 Usage

### Basic Usage

Run the application to start a live coaching session:

```bash
uv run main.py
```

This will:
1. Create a unique video call session
2. Generate user tokens and open a browser for the player to join
3. Start the AI coach that analyzes incoming video streams
4. Provide real-time coaching feedback

### Using Input Files

To analyze pre-recorded video files:

```bash
uv run main.py -i path/to/your/video/file.mp4
```

### Debug Mode

Enable debug mode to save video frames and AI responses:

```bash
uv run main.py -d
```

This creates a `debug/` directory with:
- Individual video frames (`image_*.png`)
- AI analysis transcripts (`analysis.txt`)

### Command Line Options

- `-i, --input-file`: Path to input video file (optional)
- `-d, --debug`: Enable debug mode for development

### Dependencies

- **getstream[webrtc]** (2.3.0a3): WebRTC platform for real-time communication
- **google-genai** (1.20.0): Google's Generative AI SDK for Gemini Live

### Coaching Prompts

Modify the coaching behavior by editing the `PROMPT` variable in `main.py`. The current prompt focuses on mini-golf but can be adapted for other sports.

## 🛠️ Development

### Project Structure

```
live_sports_coach/
├── main.py              # Main application entry point
├── utils.py             # Utility functions for user management and browser integration
├── pyproject.toml       # Project configuration and dependencies
├── .env                 # Environment variables (not tracked)
├── debug/               # Debug output directory (created when -d flag is used)
└── README.md            # This readme file
```

