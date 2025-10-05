import os
import base64
from pathlib import Path
from typing import Optional
import httpx
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY", "")
if XAI_API_KEY:
    os.environ["XAI_API_KEY"] = XAI_API_KEY

XAI_URL = "https://api.x.ai/v1"
XAI_STATE_URL = "https://api.x.ai"

REASONING_MODELS = ["grok-4", "grok-3-mini", "grok-3-mini-fast"]
VISION_MODELS = ["grok-4-0709", "grok-4", "grok-4-fast"]
IMAGE_GENERATION_MODELS = ["grok-2-image-1212"]


def get_api_key() -> str:
    api_key = os.getenv("XAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "XAI_API_KEY not found. Please set it in .env file or environment variables."
        )
    return api_key


def encode_image_to_base64(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    if path.suffix.lower() not in valid_extensions:
        raise ValueError(
            f"Unsupported image type: {path.suffix}. "
            f"Supported types: {', '.join(valid_extensions)}"
        )
    
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def create_client(
    timeout: float = 120.0,
    use_state: bool = False,
    api_key: Optional[str] = None
) -> httpx.AsyncClient:

    base_url = XAI_STATE_URL if use_state else XAI_URL
    key = api_key or get_api_key()
    
    return httpx.AsyncClient(
        base_url=base_url,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        },
        timeout=timeout
    )


def is_reasoning_model(model: str) -> bool:
    return model in REASONING_MODELS


def is_vision_model(model: str) -> bool:
    return model in VISION_MODELS


def validate_image_url(url: str) -> bool:
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp')
    return url.lower().endswith(valid_extensions) or url.startswith('data:image/')


def get_model_timeout(model: str) -> float:
    if is_reasoning_model(model):
        return 600.0  
    return 120.0  

def format_timestamp(timestamp: int) -> str:
    if timestamp:
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    return ""