import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Optional, Any
import httpx
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession

mcp = FastMCP("Grok MCP")

XAI_API_KEY = os.getenv("XAI_API_KEY", "")

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def create_client(timeout: float = 120.0):
    return httpx.AsyncClient(
        base_url=XAI_API_BASE,
        headers={"Authorization": f"Bearer {XAI_API_KEY}"},
        timeout=timeout
    )


@mcp.tool()
async def chat(prompt: str,  conversation_history: Optional[List[Dict[str, str]]] = None,model: str = "grok-4-fast",system_prompt: Optional[str] = None) -> str:

    messages = []
    
    if conversation_history:
        messages = conversation_history.copy()
    elif system_prompt:
            system_message = {
                "role": "system",
                "content": system_prompt
            }
            messages.append(system_message)
    
    user_message = {
        "role": "user",
        "content": prompt
    }
    messages.append(user_message)
    
    request_data = {
        "model": model,
        "messages": messages
    }

    client = get_client()
    response = await client.post("/chat/completions", json=request_data)
    response.raise_for_status()
    data = response.json()
    answer = data["choices"][0]["message"]["content"]
    await client.aclose()
    
    return answer

def main():

    mcp.run()