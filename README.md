# Solar Host

A process manager for llama-server instances with REST API and WebSocket log streaming.

## Features

- Launch and manage llama-server child processes
- Auto-assign ports starting from 3500
- Persistent configuration with auto-restart on boot
- Real-time log streaming via WebSocket
- REST API for instance management
- API key authentication

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file or set environment variables:

```bash
API_KEY=your-secret-key-here
HOST=0.0.0.0
PORT=8001
```

## Running

```bash
# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

## API Endpoints

### Instance Management

- `POST /instances` - Create new instance
- `GET /instances` - List all instances
- `GET /instances/{id}` - Get instance details
- `PUT /instances/{id}` - Update instance config
- `DELETE /instances/{id}` - Remove instance
- `POST /instances/{id}/start` - Start instance
- `POST /instances/{id}/stop` - Stop instance
- `POST /instances/{id}/restart` - Restart instance

### WebSocket

- `WS /instances/{id}/logs` - Stream logs with sequence numbers

## Authentication

All requests require an `X-API-Key` header with your configured API key.

## Example Instance Configuration

```json
{
  "model": "/path/to/model.gguf",
  "alias": "gpt-oss:120b",
  "threads": 1,
  "n_gpu_layers": 999,
  "temp": 1.0,
  "top_p": 1.0,
  "top_k": 0,
  "min_p": 0.0,
  "ctx_size": 131072,
  "chat_template_file": "/path/to/template.jinja",
  "host": "0.0.0.0",
  "api_key": "instance-api-key"
}
```

