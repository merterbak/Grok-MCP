import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Optional, Any
import httpx
from datetime import datetime
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
async def list_models() -> str:
    client = create_client()
    response = await client.get("/models")
    response.raise_for_status()
    data = response.json()
    
    models_info = []
    for model in data.get("data", []):
        model_id = model.get("id", "")
        created_timestamp = model.get("created", 0)
        owned_by = model.get("owned_by", "")
        
        # Convert Unix timestamp to date
        if created_timestamp:
            created_date = datetime.fromtimestamp(created_timestamp).strftime("%Y-%m-%d")
        else:
            created_date = ""
        
        models_info.append(f"- {model_id} (Owner: {owned_by}, Created: {created_date})")
    
    await client.aclose()
    
    return "Available Grok Models:\n" + "\n".join(models_info)

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

    client = create_client()
    response = await client.post("/chat/completions", json=request_data)
    response.raise_for_status()
    data = response.json()
    answer = data["choices"][0]["message"]["content"]
    await client.aclose()
    
    return answer
    

    
@mcp.tool()
async def generate_image(prompt: str, n: int = 1, response_format: str = "url", model: str = "grok-2-image-1212") -> dict:
    
    client = create_client()
    request_data = {
        "model": model,
        "prompt": prompt,
        "n": n,
        "response_format": response_format
    }
    response = await client.post("/images/generations", json=request_data)
    response.raise_for_status()
    data = response.json()
    await client.aclose()
      
    images = data.get("data", [])
    revised_prompt = images[0].get("revised_prompt", "")
    
    return {
        "images": images,
        "revised_prompt": revised_prompt
    }

def main():
    mcp.run()

if __name__ == "__main__":
    main()
