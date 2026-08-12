# Grok-MCP
MCP server for xAI’s Grok API with Web/X search, vision, image/video generation and file support.

<a href="https://glama.ai/mcp/servers/@merterbak/Grok-MCP">
  <img width="380" height="200" src="https://glama.ai/mcp/servers/@merterbak/Grok-MCP/badge" />
</a>

## Features

- **Agentic Tool Calling**: Web search, X search, and code execution with multi-step reasoning
- **Multiple Grok Models**: Access to latest models such as grok-4.6, grok-4.5, grok-build-0.1 and more
- **Image and Video Generation**: Create images and videos using Grok Imagine
- **Vision Capabilities**: Analyze images with Grok's vision models
- **Files API**: Upload, manage, and chat with documents 
- **Stateful Conversations**: Maintain conversation context as id across multiple requests
- **Local Chat History**: Option to save persistent client side chat history as JSON files in chats/

## Prerequisites

- Python 3.11 or higher
- xAI API key ([Get one here](https://console.x.ai))
- [Astral UV](https://docs.astral.sh/uv/getting-started/installation/)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/merterbak/Grok-MCP.git
cd Grok-MCP
```

2. Create a venv environment:
```bash
uv venv
source .venv/bin/activate # macOS/Linux or .venv\Scripts\activate on Windows
```

3. Install dependencies:

```bash
uv sync
```


## Configuration

### Claude Desktop Integration

Add this to your Claude Desktop configuration file:

```json
{
  "mcpServers": {
    "grok": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/Grok-MCP",
        "run",
        "python",
        "main.py"
      ],
      "env": {
        "XAI_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

### Claude Code Integration

Run this command from inside the project directory:

```bash
claude mcp add grok-mcp -e XAI_API_KEY=your_api_key_here -- uv run --directory /path/to/Grok-MCP python main.py
```

Or if you have a `.env` file with your key:

```bash
 claude mcp add grok-mcp -- uv run --directory /path/to/Grok-MCP python main.py
```

Verify it's registered:

```bash
claude mcp list
```

### Filesystem MCP (Optional)

Claude Desktop can't send uploaded images in the chat to an MCP tool.
The easiest way to give access to files directly from your computer is official Filesystem MCP server.
After setting it up you’ll be able to just write the image’s file path (such as /Users/mert/Desktop/image.png) in chat and Claude can use it with any vision chat tool.

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/<your-username>/Desktop",
        "/Users/<your-username>/Downloads"
      ]
    }
  }
}

```

---

For stdio:

```bash
uv run python main.py
```
Docker:

```bash
docker compose up --build
```
Mcp Inspector:

```bash
mcp dev main.py
```


# Available Tools

Each tool has a full docstring in [src/server.py](src/server.py) with its arguments and return format. MCP client surfaces those directly, so this list is just a quick map of what's available.

Note: For using images and files, you must provide paths to chat. See [Filesystem MCP (Optional)](#filesystem-mcp-optional) for setup.

### Chat and reasoning
- `chat` — standard chat completion with optional persistent history and multi-agent support.
- `chat_with_vision` — analyze local or remote images with a Grok vision model.
- `chat_with_files` — chat grounded on previously uploaded documents.
- `stateful_chat` — continue a server-side stored conversation via `response_id`.
- `retrieve_stateful_response` — fetch a stored response by ID.
- `delete_stateful_response` — delete a stored response by ID.

### Agentic tools
- `web_search` — autonomous web research with domain filters and citations.
- `x_search` — autonomous search over X (Twitter) posts, with handle and date filters.
- `code_executor` — solve tasks by running Python in a sandbox.
- `grok_agent` — unified agent that mixes files, images, web search, X search, and code execution.

### Image and video
- `generate_image` — create or edit images with Grok Imagine (multi-reference editing supported).
- `generate_video` — text-to-video, image-to-video, or video editing with Grok Imagine.
- `extend_video` — extend an existing generated video with a follow-up prompt.

### Files
- `upload_file` — upload a local document.
- `list_files` — list uploaded files with sorting.
- `get_file` — fetch file metadata by ID.
- `get_file_content` — download file content as text.
- `delete_file` — delete a file by ID.

### Local chat history
- `list_chat_sessions` — list saved sessions in `chats/`.
- `get_chat_history` — get a session's full transcript.
- `clear_chat_history` — delete a session's local history file.

### Models
- `list_models` — list all Grok language and image models with live pricing.

  
## License

This project is open source and available under the MIT License.
