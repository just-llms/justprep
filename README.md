# justprep
=======
# AI-Powered Interview System

A real-time, voice-based AI mock interview engine that conducts technical interviews with natural conversation flow, interruption handling (barge-in), and intelligent follow-up questions.

## Overview

This system implements a controlled real-time interview platform that:
- Conducts synchronous interviews using voice input/output
- Adapts questions in real time based on candidate responses
- Handles natural interruptions (barge-in)
- Manages interview phases with deterministic flow control
- Provides low-latency, real-time voice interaction

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- API keys for:
  - Deepgram (for Speech-to-Text)
  - Cartesia AI (for Text-to-Speech)
  - OpenRouter (for LLM access)

## Setup Instructions

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API keys:
   ```env
   DEEPGRAM_API_KEY=your_actual_deepgram_api_key
   CARTESIA_API_KEY=your_actual_cartesia_api_key
   OPENROUTER_API_KEY=your_actual_openrouter_api_key
   ```

3. Adjust other configuration values as needed (audio settings, VAD thresholds, LLM model, etc.)

### 4. Verify Installation

```bash
python -c "import fastapi; print('FastAPI installed successfully')"
```

## Running the Application

### Development Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The application will be available at:
- API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Production Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Project Structure

```
interview/
├── core/                    # Core business logic and services
│   ├── audio/              # Audio processing components
│   ├── stt/                # Speech-to-text engine
│   ├── tts/                # Text-to-speech engine
│   ├── ai/                 # AI engine components
│   └── memory/            # Memory system
├── models/                 # Data models and exceptions (ALL Pydantic models)
│   ├── constants.py        # Enums and constants
│   ├── session_models.py   # Session state models
│   ├── request_models.py   # Request models
│   ├── response_models.py  # Response models
│   └── exceptions.py       # Custom exceptions
├── routes/                 # FastAPI route handlers
├── util/                   # Utility functions and helpers
│   ├── logger.py          # Logging configuration
│   └── audio_config.py    # Audio format constants
├── dependencies/          # Dependency injection functions
├── main.py                # FastAPI application entry point
├── requirements.txt       # Python dependencies
└── .env.example           # Environment variables template
```

## Key Components

- **Audio Processing**: VAD (Voice Activity Detection), turn control, audio format handling
- **STT Engine**: Streaming speech-to-text using Deepgram
- **TTS Engine**: Streaming text-to-speech using Cartesia AI
- **AI Engine**: FSM controller, context builder, LLM integration, response planner
- **Memory System**: Short-term sliding window and long-term async memory
- **Session Management**: In-memory session state management

## Development Guidelines

This project follows strict coding standards defined in `.cursor/rules.md`:

- **Type Hints**: All functions must include type hints
- **Naming Conventions**: PascalCase for classes, snake_case for functions
- **File Organization**: One class per file when possible
- **Pydantic Models**: All data models must be in `models/` directory
- **Logging**: Use `logging.getLogger(__name__)` pattern
- **Error Handling**: Always handle errors gracefully with proper logging

## Environment Variables

Key environment variables (see `.env.example` for complete list):

- `DEEPGRAM_API_KEY`: Deepgram API key for STT
- `CARTESIA_API_KEY`: Cartesia AI API key for TTS
- `OPENROUTER_API_KEY`: OpenRouter API key for LLM
- `AUDIO_SAMPLE_RATE`: Audio sample rate (default: 16000)
- `LOG_LEVEL`: Logging level (default: INFO)

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Troubleshooting

### Import Errors
Ensure you've activated the virtual environment and installed all dependencies:
```bash
pip install -r requirements.txt
```

### API Key Errors
Verify your `.env` file exists and contains valid API keys.

### Audio Issues
Check that audio format constants in `util/audio_config.py` match your client configuration.

## Testing

### Test Client

A test client is provided to test the audio processor functionality:

1. **HTML Test Client** (`test_client.html`):
   - Open `test_client.html` in a web browser
   - Or serve it via HTTP: `python -m http.server 8080` then open `http://localhost:8080/test_client.html`
   - Click "Connect" to establish WebSocket connection
   - Click "Start Audio" to begin capturing microphone audio
   - Speak into microphone to test VAD detection
   - Watch server logs for VAD events and turn controller state changes

2. **Python Test Script** (`test_audio_processor.py`):
   ```bash
   pip install websockets numpy
   python test_audio_processor.py
   ```
   - Sends synthetic audio data to test the pipeline
   - Useful for testing without microphone access

### What to Test

- **WebSocket Connection**: Verify connection is established
- **Audio Capture**: Check that audio chunks are being sent
- **VAD Detection**: Watch for `speech_start` and `speech_end` events in logs
- **Turn Controller**: Verify state transitions (IDLE → USER → IDLE)
- **Audio Processor**: Check orchestration of all components

### Expected Log Output

When testing, you should see logs like:
```
INFO - WebSocket connected for session test-session-123
INFO - Created audio processor for session test-session-123
INFO - Audio processing started for session test-session-123
DEBUG - Processed audio chunk for session test-session-123: 640 bytes
INFO - VAD event received for session test-session-123: speech_start
INFO - User starting speech. Transitioning idle -> user
INFO - VAD event received for session test-session-123: speech_end
INFO - User finished speaking. Transitioning user -> idle
```


