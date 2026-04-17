# Grok-MCP

MCP server for xAI's Grok API with agentic tool calling, image and video generation, vision, and file support.


<a href="https://glama.ai/mcp/servers/@merterbak/Grok-MCP">
  <img width="380" height="200" src="https://glama.ai/mcp/servers/@merterbak/Grok-MCP/badge" />
</a>

## Features

- **Agentic Tool Calling**: Web search, X search, and code execution with multi-step reasoning
- **Multiple Grok Models**: Access to latest models such as grok-4.20-0309-reasoning, grok-4-1-fast-reasoning and more
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

Note: For using images and files, you must provide paths to chat. See [Filesystem MCP (Optional)](#filesystem-mcp-optional) for setup.

### `list_models`
List all available Grok models with pricing.
---

### `chat`
Standard chat completion with optional persistent history.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | required | Your message |
| `session` | str | None | Session name to save/load history |
| `model` | str | grok-4-1-fast-reasoning | Model to use |
| `system_prompt` | str | None | System instruction |

---

### `chat_with_vision`
Analyze images with text.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | required | Question about the image |
| `session` | str | None | Session name to save/load history |
| `model` | str | grok-4-1-fast-reasoning | Vision model |
| `image_paths` | List[str] | None | Local image file paths |
| `image_urls` | List[str] | None | Image URLs |
| `detail` | str | auto | auto, low, or high |

**Returns:** Content + usage with `prompt_image_tokens`

---

### `generate_image`
Create or edit images from text.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | required | Image description or edit instruction |
| `model` | str | grok-imagine-image | Image model |
| `image_path` | str | None | Local image path to edit |
| `image_url` | str | None | Image URL to edit |
| `n` | int | 1 | Number of images (1-10) |
| `aspect_ratio` | str | None | like "16:9", "1:1" |
| `resolution` | str | None | "1k" or "2k" |

---

### `generate_video`
Create or edit videos from text, images, or existing videos.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | required | Video description or edit instruction |
| `model` | str | grok-imagine-video | Video model |
| `image_path` | str | None | Local image path to animate |
| `image_url` | str | None | Image URL to animate |
| `video_path` | str | None | Local video path to edit (max 20MB) |
| `video_url` | str | None | Video URL to edit |
| `duration` | int | None | Duration in seconds (1-15) |
| `aspect_ratio` | str | None | like "16:9", "4:3" |
| `resolution` | str | None | "720p" or "480p" |
| `reference_image_urls` | List[str] | None | Reference image URLs for conditioning |

---

### `web_search`
Agentic web search with autonomous research.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | required | Search query |
| `model` | str | grok-4-1-fast-reasoning | Model |
| `allowed_domains` | List[str] | None | Restrict to domains (max 5) |
| `excluded_domains` | List[str] | None | Exclude domains (max 5) |
| `enable_image_understanding` | bool | False | Analyze images in results |
| `include_inline_citations` | bool | False | Embed citations in text |
| `max_turns` | int | None | Limit reasoning turns |

**Returns:** Content, citations, tool_calls, usage

---

### `x_search`
Agentic X (Twitter) search.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | required | Search query |
| `model` | str | grok-4-1-fast-reasoning | Model |
| `allowed_x_handles` | List[str] | None | Only these handles (max 10) |
| `excluded_x_handles` | List[str] | None | Exclude handles (max 10) |
| `from_date` | str | None | Start date (DD-MM-YYYY) |
| `to_date` | str | None | End date (DD-MM-YYYY) |
| `enable_image_understanding` | bool | False | Analyze images |
| `enable_video_understanding` | bool | False | Analyze videos |
| `include_inline_citations` | bool | False | Embed citations |
| `max_turns` | int | None | Limit turns |

**Returns:** Content, citations, tool_calls, usage

---

### `grok_agent`
Unified agent combining files, images, and all agentic tools (web search, X search, code execution).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | required | Your query |
| `session` | str | None | Session name to save/load history |
| `model` | str | grok-4-1-fast-reasoning | Model |
| `file_ids` | List[str] | None | Uploaded file IDs to search |
| `image_urls` | List[str] | None | Image URLs to analyze |
| `image_paths` | List[str] | None | Local image paths |
| `use_web_search` | bool | False | Enable web search |
| `use_x_search` | bool | False | Enable X search |
| `use_code_execution` | bool | False | Enable code execution |
| + all web_search and x_search params | | | |

**Returns:** Content, citations, tool_calls, code_outputs, uploaded_file_ids, usage

---

### `code_executor`
Execute Python code for calculations and analysis.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | required | Task description |
| `model` | str | grok-4-1-fast-reasoning | Model |
| `max_turns` | int | None | Limit turns |

**Returns:** Content, tool_calls, code_outputs, usage

---

### `stateful_chat`
Maintain conversation state across requests.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | required | Your message |
| `model` | str | grok-4-1-fast-reasoning | Model |
| `response_id` | str | None | Previous response ID to continue a conversation |
| `system_prompt` | str | None | System instruction |

**Returns:** Content, response_id, usage

---

### `retrieve_stateful_response`
Retrieve a stored conversation.

---

### `delete_stateful_response`
Delete a stored conversation.


### `upload_file`
Upload a document (max 48 MB).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | str | required | Local file path |

**Supported formats:** .txt, .md, .py, .js, .csv, .json, .pdf, and more

---

### `list_files`
List uploaded files with sorting.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 100 | Max files to return |
| `order` | str | desc | asc or desc |
| `sort_by` | str | created_at | created_at, filename, or size |

---

### `get_file`
Get file metadata by ID.

---

### `get_file_content`
Download file content by ID.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_id` | str | required | File ID |
| `max_bytes` | int | 500000 | Max bytes to return |

---

### `delete_file`
Delete a file by ID.

---

### `chat_with_files`
Chat with uploaded documents using agentic document search.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | required | Question about docs |
| `session` | str | None | Session name to save/load history |
| `model` | str | grok-4-1-fast-reasoning | Model |
| `file_ids` | List[str] | None | File IDs to search |
| `system_prompt` | str | None | System instruction |

Returns: Content, citations, usage

---

### `list_chat_sessions`
List all saved chat sessions in `chats/`.

---

### `get_chat_history`
Get the full message history for a session.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session` | str | default | Session name |

---

### `clear_chat_history`
Delete the history file for a session.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session` | str | default | Session name |



---

  
## License

This project is open source and available under the MIT License.
