# Grok-MCP

A MCP server for xAI's Grok API, providing access to capabilities including image understanding, image generation, live web search, and reasoning models.

## 🚀 Features

- **Multiple Grok Models**: Access to Grok-4, Grok-4-Fast, Grok-3-Mini, and more
- **Image Generation**: Create images using Grok's image generation models
- **Vision Capabilities**: Analyze images with Grok's vision models
- **Live Web Search**: Real-time web search with citations from news, web, X (Twitter), and RSS feeds
- **Reasoning Models**: Advanced reasoning with extended thinking models (Grok-3-Mini, Grok-4)
- **Stateful Conversations**: Maintain conversation context across multiple requests
- **Conversation History**: Built-in support for multi-turn conversations

## 📋 Prerequisites

- Python 3.11 or higher
- xAI API key ([Get one here](https://console.x.ai))
- `uv` package manager 

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/merterbak/Grok-MCP.git
cd Grok-MCP
```

2. Install dependencies using `uv`:
```bash
uv sync
```

## 🔧 Configuration

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
## Usage

For stdio:

```bash
uv run python main.py
```

## 📚 Available Tools

### 1. `list_models`
List all available Grok models with creation dates and ownership information.

### 2. `chat`
Standard chat completion with extensive customization options.

**Parameters:**
- `prompt` (required): Your message
- `model`: Model to use (default: "grok-4-fast")
- `system_prompt`: Optional system instruction
- `use_conversation_history`: Enable multi-turn conversations
- `temperature`, `max_tokens`, `top_p`: Generation parameters
- `presence_penalty`, `frequency_penalty`, `stop`: Advanced control
- `reasoning_effort`: For reasoning models ("low" or "high")

### 3. `chat_with_reasoning`
Get detailed reasoning along with the response.

**Parameters:**
- `prompt` (required): Your question or task
- `model`: "grok-4", "grok-3-mini", or "grok-3-mini-fast"
- `reasoning_effort`: "low" or "high" (not for grok-4)
- `system_prompt`, `temperature`, `max_tokens`, `top_p`

**Returns:** Content, reasoning content, and usage statistics

### 4. `chat_with_vision`
Analyze images with natural language queries.

**Parameters:**
- `prompt` (required): Your question about the image(s)
- `image_paths`: List of local image file paths
- `image_urls`: List of image URLs
- `detail`: "auto", "low", or "high"
- `model`: Vision-capable model (default: "grok-4-0709")

**Supported formats:** JPG, JPEG, PNG

### 5. `generate_image`
Create images from text descriptions.

**Parameters:**
- `prompt` (required): Image description
- `n`: Number of images to generate (default: 1)
- `response_format`: "url" or "b64_json"
- `model`: Image generation model (default: "grok-2-image-1212")

**Returns:** Generated images and revised prompt

### 6. `live_search`
Search the web in real-time with source citations.

**Parameters:**
- `prompt` (required): Your search query
- `model`: Model to use (default: "grok-4")
- `mode`: "on" or "off"
- `return_citations`: Include source citations (default: true)
- `from_date`, `to_date`: Date range (YYYY-MM-DD)
- `max_search_results`: Max results to fetch (default: 20)
- `country`: Country code for localized search
- `rss_links`: List of RSS feed URLs to search
- `sources`: Custom source configuration

**Returns:** Content, citations, usage stats, and number of sources used

### 7. `stateful_chat`
Maintain conversation state across multiple requests on xAI servers.

**Parameters:**
- `prompt` (required): Your message
- `response_id`: Previous response ID to continue conversation
- `model`: Model to use (default: "grok-4")
- `system_prompt`: System instruction (only for new conversations)
- `include_reasoning`: Include reasoning summary
- `temperature`, `max_tokens`

**Returns:** Response with ID for continuing the conversation (stored for 30 days)

### 8. `retrieve_stateful_response`
Retrieve a previously stored conversation response.

**Parameters:**
- `response_id` (required): The response ID to retrieve

### 9. `delete_stateful_response`
Delete a stored conversation from xAI servers.

**Parameters:**
- `response_id` (required): The response ID to delete
  
## Roadmap

- add docker support
- fix chat vision model tool

## 📄 License

This project is open source and available under the MIT License.
