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

## Setup

### 1. Create .env file

Create a `.env` file in the `solar-host/` directory:

```bash
API_KEY=your-secret-key-here
HOST=0.0.0.0
PORT=8001
```

**Note:** You can also copy from the example:
```bash
cp .env.example .env
# Then edit .env with your values
```

### 2. Start the server

```bash
# Start the server (from solar-host directory)
uvicorn app.main:app --host 0.0.0.0 --port 8001

# Or with auto-reload for development
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

The server will:
- Create `config.json` automatically (if it doesn't exist)
- Create `logs/` directory for instance logs
- Auto-restart any instances that were running before shutdown

### 3. Verify it's running

```bash
curl http://localhost:8001/health
# Should return: {"status":"healthy","service":"solar-host","version":"1.0.0"}
```

### 4. Access Swagger UI (Optional)

Open your browser to: **http://localhost:8001/docs**

1. Click the **"Authorize"** button (green lock icon at the top right)
2. Enter your API key from `.env` file
3. Click **"Authorize"** and then **"Close"**
4. Now you can use the interactive API documentation to create and manage instances!

This is much easier than using `curl` commands! ✨

## Managing Instances

### Important: config.json is AUTO-GENERATED

**You do NOT need to manually create `config.json`!**

- The file is automatically created in the `solar-host/` directory
- All instances are stored in this single file
- The file persists instance configurations and their running state
- It's managed entirely through the REST API

### Creating Your First Instance

Use the REST API to create instances. Here's a complete example:

```bash
curl -X POST http://localhost:8001/instances \
  -H "X-API-Key: your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "model": "/Users/administrator/models/llama-3-8b.gguf",
      "alias": "llama-3:8b",
      "threads": 4,
      "n_gpu_layers": 999,
      "temp": 0.7,
      "top_p": 0.9,
      "top_k": 40,
      "min_p": 0.05,
      "ctx_size": 8192,
      "chat_template_file": "/Users/administrator/models/templates/llama3.jinja",
      "host": "0.0.0.0",
      "api_key": "llama3-instance-key"
    }
  }'
```

**Response will include the instance ID:**
```json
{
  "instance": {
    "id": "abc123...",
    "config": { ... },
    "status": "stopped",
    ...
  },
  "message": "Instance abc123... created successfully"
}
```

### Adding Multiple Instances

Simply call the create endpoint multiple times with different configurations:

```bash
# Instance 1: Small model
curl -X POST http://localhost:8001/instances \
  -H "X-API-Key: your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "model": "/path/to/small-model.gguf",
      "alias": "small-model:7b",
      "threads": 2,
      "n_gpu_layers": 999,
      "temp": 0.7,
      "top_p": 0.9,
      "top_k": 40,
      "min_p": 0.05,
      "ctx_size": 4096,
      "host": "0.0.0.0",
      "api_key": "instance1-key"
    }
  }'

# Instance 2: Large model
curl -X POST http://localhost:8001/instances \
  -H "X-API-Key: your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "model": "/path/to/large-model.gguf",
      "alias": "large-model:70b",
      "threads": 8,
      "n_gpu_layers": 999,
      "temp": 0.8,
      "top_p": 0.95,
      "top_k": 40,
      "min_p": 0.05,
      "ctx_size": 32768,
      "host": "0.0.0.0",
      "api_key": "instance2-key"
    }
  }'
```

All instances are stored in the same `config.json` file.

### Starting an Instance

```bash
# Get the instance ID from the create response or list endpoint
curl -X POST http://localhost:8001/instances/{instance-id}/start \
  -H "X-API-Key: your-secret-key-here"
```

The instance will:
- Get assigned an available port (starting from 3500)
- Start the llama-server process
- Begin logging to `logs/{alias}_{timestamp}.log`

### Viewing All Instances

```bash
curl http://localhost:8001/instances \
  -H "X-API-Key: your-secret-key-here"
```

### Complete Workflow Example

```bash
# 1. Create instance
RESPONSE=$(curl -s -X POST http://localhost:8001/instances \
  -H "X-API-Key: your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "model": "/path/to/model.gguf",
      "alias": "my-model:8b",
      "threads": 4,
      "n_gpu_layers": 999,
      "temp": 0.7,
      "top_p": 0.9,
      "top_k": 40,
      "min_p": 0.05,
      "ctx_size": 8192,
      "host": "0.0.0.0",
      "api_key": "instance-key"
    }
  }')

# 2. Extract instance ID
INSTANCE_ID=$(echo $RESPONSE | jq -r '.instance.id')
echo "Created instance: $INSTANCE_ID"

# 3. Start the instance
curl -X POST http://localhost:8001/instances/$INSTANCE_ID/start \
  -H "X-API-Key: your-secret-key-here"

# 4. Check status
curl http://localhost:8001/instances/$INSTANCE_ID \
  -H "X-API-Key: your-secret-key-here"

# 5. View logs (WebSocket - use a WebSocket client)
# ws://localhost:8001/instances/$INSTANCE_ID/logs

# 6. Stop when done
curl -X POST http://localhost:8001/instances/$INSTANCE_ID/stop \
  -H "X-API-Key: your-secret-key-here"
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

All requests require an `X-API-Key` header with your configured API key from the `.env` file.

## Configuration Reference

### Instance Config Parameters

When creating an instance, you can configure these parameters:

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `model` | ✅ Yes | - | Full path to the GGUF model file |
| `alias` | ✅ Yes | - | Model alias (e.g., "llama-3:8b") used for routing |
| `threads` | No | 1 | Number of CPU threads to use |
| `n_gpu_layers` | No | 999 | Number of layers to offload to GPU (999 = all) |
| `temp` | No | 1.0 | Sampling temperature (0.0-2.0) |
| `top_p` | No | 1.0 | Top-p sampling (0.0-1.0) |
| `top_k` | No | 0 | Top-k sampling (0 = disabled) |
| `min_p` | No | 0.0 | Min-p sampling (0.0-1.0) |
| `ctx_size` | No | 131072 | Context window size |
| `chat_template_file` | No | - | Path to Jinja chat template file |
| `special` | No | false | When true, starts llama-server with the `--special` flag |
| `host` | No | "0.0.0.0" | Host to bind llama-server to |
| `port` | No | auto | Port (auto-assigned if not specified) |
| `api_key` | ✅ Yes | - | API key for this specific llama-server instance |

### Example Configurations

**Small 7B model:**
```json
{
  "model": "/Users/admin/models/llama-3-7b.gguf",
  "alias": "llama-3:7b",
  "threads": 4,
  "n_gpu_layers": 999,
  "temp": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "min_p": 0.05,
  "ctx_size": 8192,
  "host": "0.0.0.0",
  "api_key": "llama3-7b-key"
}
```

**Large 120B model with custom template:**
```json
{
  "model": "/Users/admin/models/gpt-oss-120b-F16.gguf",
  "alias": "gpt-oss:120b",
  "threads": 1,
  "n_gpu_layers": 999,
  "temp": 1.0,
  "top_p": 1.0,
  "top_k": 0,
  "min_p": 0.0,
  "ctx_size": 131072,
  "chat_template_file": "/Users/admin/models/templates/harmony.jinja",
  "special": false,
  "host": "0.0.0.0",
  "api_key": "gpt-oss-120b-key"
}
```

## File Structure

After running solar-host, your directory will look like:

```
solar-host/
├── .env                    # Your configuration (not in git)
├── config.json             # Auto-generated instance storage (not in git)
├── logs/                   # Auto-generated log directory (not in git)
│   ├── llama-3-7b_1234567890.log
│   └── gpt-oss-120b_1234567891.log
├── app/                    # Application code
├── requirements.txt
└── README.md
```

## Troubleshooting

### Solar-host won't start

**Error: "Address already in use"**
- Another service is using port 8001
- Solution: Change `PORT` in `.env` or stop the other service

**Error: "No module named 'app'"**
- You're not in the correct directory
- Solution: `cd solar-host` and run from there

### Instance fails to start

**Check the following:**

1. **Model path is correct:**
   ```bash
   ls -lh /path/to/your/model.gguf
   ```

2. **llama-server is in PATH:**
   ```bash
   which llama-server
   # Should return the path to llama-server
   ```

3. **Check instance logs:**
   - Look in `logs/` directory for the instance log file
   - File name includes the model alias and timestamp

4. **Port already in use:**
   - Solar-host will auto-assign the next available port
   - Check with: `lsof -i :3500` (replace with your port)

5. **Model format issues:**
   - Ensure the model is a valid GGUF file
   - Try running llama-server manually first

### Instance keeps retrying and failing

- Solar-host will retry starting an instance up to 2 times
- After 2 failures, it gives up and marks the instance as "failed"
- Check the `error_message` field in the instance details:
  ```bash
  curl http://localhost:8001/instances/{instance-id} \
    -H "X-API-Key: your-key" | jq '.error_message'
  ```

### Logs not showing up

- Logs are stored in `logs/` directory
- File name format: `{alias-with-dashes}_{timestamp}.log`
- For real-time logs, use the WebSocket endpoint

### Auto-restart not working

- Make sure instances were in "running" state before shutdown
- Check `config.json` - running instances have their status saved
- The status is restored on solar-host startup

## Integration with Solar Control

Once you have solar-host running with instances:

1. Register this host with solar-control:
   ```bash
   curl -X POST http://your-control-server:8000/hosts \
     -H "X-API-Key: gateway-key" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "Mac Studio 1",
       "url": "http://192.168.1.100:8001",
       "api_key": "your-solar-host-api-key"
     }'
   ```

2. Your instances will be accessible through the unified OpenAI gateway!

## Support

For issues and questions:
- Check logs in `logs/` directory
- View instance status via API
- Check llama-server is installed and accessible
- Verify model paths are correct

