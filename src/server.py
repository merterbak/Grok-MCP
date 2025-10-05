import os
from dotenv import load_dotenv
import json
import base64
from pathlib import Path
from typing import List, Dict, Optional, Any
import httpx
from datetime import datetime, timedelta
from mcp.server.fastmcp import FastMCP, Image
from .utils import (
    encode_image_to_base64,
    create_client,
    is_reasoning_model,
    is_vision_model,
    get_model_timeout,
    XAI_API_KEY,
)


mcp = FastMCP(
    name="Grok MCP Server",
)

conversation_history: List[Dict[str, str]] = []

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
async def generate_image(
    prompt: str,
    n: int = 1,
    response_format: str = "url",
    model: str = "grok-2-image-1212"
) -> dict:
    
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

@mcp.tool()
async def chat_with_vision(
    prompt: str,
    image_paths: Optional[List[str]] = None,
    image_urls: Optional[List[str]] = None,
    detail: str = "auto",
    model: str = "grok-4-0709"
) -> str:
    content_items = []
    if image_paths:
        for path in image_paths:
            ext = Path(path).suffix.lower().replace('.', '')
            if ext not in ["jpg", "jpeg", "png"]:
                raise ValueError(f"Unsupported image type: {ext}")
            base64_img = encode_image_to_base64(path)
            content_items.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{ext};base64,{base64_img}",
                    "detail": detail
                }
            })
    if image_urls:
        for url in image_urls:
            content_items.append({
                "type": "image_url",
                "image_url": {
                    "url": url,
                    "detail": detail
                }
            })
    if prompt:
        content_items.append({
            "type": "text",
            "text": prompt
        })
    messages = [
        {
            "role": "user",
            "content": content_items
        }
    ]
    request_data = {
        "model": model,
        "messages": messages
    }
    client = create_client()
    response = await client.post("/chat/completions", json=request_data)
    response.raise_for_status()
    data = response.json()
    await client.aclose()
    return data["choices"][0]["message"]["content"]

@mcp.tool()
async def chat(
    prompt: str,
    model: str = "grok-4-fast",
    system_prompt: Optional[str] = None,
    use_conversation_history: bool = False,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    stop: Optional[List[str]] = None,
    reasoning_effort: Optional[str] = None
) -> str:

    global conversation_history
    if not use_conversation_history:
        conversation_history = []
    
    messages = []
    
    if system_prompt and len(conversation_history) == 0:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    messages.extend(conversation_history)
    
    messages.append({
        "role": "user",
        "content": prompt
    })
    
    request_data = {
        "model": model,
        "messages": messages
    }
    
    if temperature is not None:
        request_data["temperature"] = temperature
    if max_tokens is not None:
        request_data["max_tokens"] = max_tokens
    if top_p is not None:
        request_data["top_p"] = top_p
    
    is_reasoning = is_reasoning_model(model)
    
    if is_reasoning:
        # reasoning_effort only for grok-3-mini
        if reasoning_effort and model != "grok-4":
            if reasoning_effort not in ["low", "high"]:
                raise ValueError("reasoning_effort must be 'low' or 'high'")
            request_data["reasoning_effort"] = reasoning_effort
    else:
        # These parameters only work with non reasoning models
        if presence_penalty is not None:
            request_data["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            request_data["frequency_penalty"] = frequency_penalty
        if stop is not None:
            request_data["stop"] = stop
    
    timeout = get_model_timeout(model)
    client = create_client(timeout=timeout)
    response = await client.post("/chat/completions", json=request_data)
    response.raise_for_status()
    data = response.json()
    await client.aclose()
    
    assistant_response = data["choices"][0]["message"]["content"]
    
    if use_conversation_history:
        conversation_history.append({
            "role": "user",
            "content": prompt
        })
        conversation_history.append({
            "role": "assistant",
            "content": assistant_response
        })
    
    return assistant_response

@mcp.tool()
async def chat_with_reasoning(
    prompt: str,
    model: str = "grok-3-mini",
    system_prompt: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None
) -> Dict[str, Any]:
    
    if model not in ["grok-4", "grok-3-mini", "grok-3-mini-fast"]:
        raise ValueError(f"Model {model} isn't a reasoning model. Use 'grok-4', 'grok-3-mini', or 'grok-3-mini-fast'.")
    
    messages = []
    
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    messages.append({
        "role": "user",
        "content": prompt
    })
    
    request_data = {
        "model": model,
        "messages": messages
    }
    
    # optional parameters
    if temperature is not None:
        request_data["temperature"] = temperature
    if max_tokens is not None:
        request_data["max_tokens"] = max_tokens
    if top_p is not None:
        request_data["top_p"] = top_p
    
    if reasoning_effort and model != "grok-4":
        if reasoning_effort not in ["low", "high"]:
            raise ValueError("reasoning_effort must be 'low' or 'high'")
        request_data["reasoning_effort"] = reasoning_effort
    
    client = create_client(timeout=3600.0)
    response = await client.post("/chat/completions", json=request_data)
    response.raise_for_status()
    data = response.json()
    await client.aclose()
    
    choice = data["choices"][0]
    message = choice["message"]
    
    result = {
        "content": message.get("content", ""),
        "reasoning_content": message.get("reasoning_content", ""),
        "usage": data.get("usage", {})
    }
    
    return result
    
@mcp.tool()
async def live_search(
    prompt: str,
    model: str = "grok-4",
    mode: str = "on",
    return_citations: bool = True,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    max_search_results: int = 20,
    country: Optional[str] = None,
    rss_links: Optional[List[str]] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
    system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    
    messages = []
    
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    messages.append({
        "role": "user",
        "content": prompt
    })
    
    search_params: Dict[str, Any] = {
        "mode": mode,
        "return_citations": return_citations
    }
    
    if from_date:
        search_params["from_date"] = from_date
    if to_date:
        search_params["to_date"] = to_date
    if max_search_results != 20:
        search_params["max_search_results"] = max_search_results
    
    if sources:
        search_params["sources"] = sources
    elif country or rss_links:
        default_sources = []
        
        if country:
            default_sources.extend([
                {"type": "web", "country": country},
                {"type": "news", "country": country},
                {"type": "x"}
            ])
        else:
            default_sources.extend([
                {"type": "web"},
                {"type": "news"},
                {"type": "x"}
            ])
        
        if rss_links:
            for link in rss_links:
                default_sources.append({"type": "rss", "links": [link]})
        
        search_params["sources"] = default_sources
    
    request_data = {
        "model": model,
        "messages": messages,
        "search_parameters": search_params
    }
    
    client = create_client()
    response = await client.post("/chat/completions", json=request_data)
    response.raise_for_status()
    data = response.json()
    
    choice = data["choices"][0]
    result = {
        "content": choice["message"]["content"],
        "usage": data.get("usage", {}),
    }
    
    if return_citations and "citations" in choice["message"]:
        result["citations"] = choice["message"]["citations"]
    
    if "num_sources_used" in data.get("usage", {}):
        result["num_sources_used"] = data["usage"]["num_sources_used"]
    
    await client.aclose()
    return result



@mcp.tool()
async def stateful_chat(
    prompt: str,
    response_id: Optional[str] = None,
    model: str = "grok-4",
    system_prompt: Optional[str] = None,
    include_reasoning: bool = False,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> Dict[str, Any]:
    
    input_messages = []
    
    # System prompt only for new conversations (not when continuing)
    if system_prompt and not response_id:
        input_messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    input_messages.append({
        "role": "user",
        "content": prompt
    })
    
    request_data: Dict[str, Any] = {
        "model": model,
        "input": input_messages,
        "store": True 
    }
    
    if response_id:
        request_data["previous_response_id"] = response_id
    
    # Optional parameters
    if temperature is not None:
        request_data["temperature"] = temperature
    if max_tokens is not None:
        request_data["max_output_tokens"] = max_tokens
    
    if include_reasoning:
        request_data["reasoning"] = {
            "include": ["encrypted_content"]
        }
    
    timeout = get_model_timeout(model)
    
    client = create_client(timeout=timeout, use_state=True)
    
    
    response = await client.post("/v1/responses", json=request_data)
    response.raise_for_status()
    data = response.json()
    await client.aclose()
    
    output = data.get("output", [])
    content = ""
    reasoning_summary = None
    
    for item in output:
        if item.get("type") == "message" and item.get("role") == "assistant":
            for content_item in item.get("content", []):
                if content_item.get("type") == "output_text":
                    content = content_item.get("text", "")
                    break
        elif item.get("type") == "reasoning":
            for summary_item in item.get("summary", []):
                if summary_item.get("type") == "summary_text":
                    reasoning_summary = summary_item.get("text", "")
                    break
    
    expiration = datetime.now() + timedelta(days=30)
    
    result = {
        "content": content,
        "response_id": data.get("id"),
        "status": data.get("status"),
        "model": data.get("model"),
        "usage": data.get("usage", {}),
        "stored_until": expiration.strftime("%Y-%m-%d"),
        "continued_from": response_id if response_id else None
    }
    
    if reasoning_summary:
        result["reasoning"] = reasoning_summary
    
    return result


@mcp.tool()
async def retrieve_stateful_response(response_id: str) -> Dict[str, Any]:
    
    client = create_client(use_state=True)
    
    response = await client.get(f"/v1/responses/{response_id}")
    response.raise_for_status()
    data = response.json()
    
    output = data.get("output", [])
    content = ""
    reasoning = None
    
    #for getting text of reasoning or pass reasoning 
    for item in output:
        if item.get("type") == "message" and item.get("role") == "assistant":
            for content_item in item.get("content", []):
                if content_item.get("type") == "output_text":
                    content = content_item.get("text", "")
                    break
        elif item.get("type") == "reasoning":
            for summary_item in item.get("summary", []):
                if summary_item.get("type") == "summary_text":
                    reasoning = summary_item.get("text", "")
                    break
    
    await client.aclose()
    
    return {
        "response_id": data.get("id"),
        "model": data.get("model"),
        "created_at": datetime.fromtimestamp(data.get("created_at", 0)).isoformat(),
        "status": data.get("status"),
        "content": content,
        "reasoning": reasoning,
        "usage": data.get("usage", {}),
        "previous_response_id": data.get("previous_response_id"),
        "store": data.get("store", False)
    }


@mcp.tool()
async def delete_stateful_response(response_id: str):
    
    client = create_client(use_state=True)
    
    response = await client.delete(f"/v1/responses/{response_id}")
    response.raise_for_status()
    data = response.json()
    
    await client.aclose()
    
    return {
        "response_id": data.get("id"),
        "deleted": data.get("deleted", False),
        "message": f"Response {response_id} deleted from xAI servers"
    }


def main():
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()