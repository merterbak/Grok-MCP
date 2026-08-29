import os
import json
import base64
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv("example.env")

XAI_API_KEY = os.getenv("XAI_API_KEY", "")


def encode_image_to_base64(image_path: str):
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_video_to_base64(video_path: str):
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")


def usage_footer(*responses):
    prompt_tokens = completion_tokens = reasoning_tokens = 0
    cost = 0.0
    has_cost = False
    for response in responses:
        usage = response.usage
        if usage:
            prompt_tokens += usage.prompt_tokens
            completion_tokens += usage.completion_tokens
            reasoning_tokens += usage.reasoning_tokens
        if response.cost_usd is not None:
            cost += response.cost_usd
            has_cost = True

    parts = []
    if prompt_tokens or completion_tokens:
        tokens = f"**Tokens:** {prompt_tokens:,} in / {completion_tokens:,} out"
        if reasoning_tokens:
            tokens += f" ({reasoning_tokens:,} reasoning)"
        parts.append(tokens)
    if has_cost:
        parts.append(f"**Cost:** ${cost:.4f}")
    if not parts:
        return ""
    return "\n\n---\n" + " · ".join(parts)

def load_history(session: str):
    path = Path("chats") / f"{session}.json"
    if path.exists():
        return json.loads(path.read_text())
    return []


def save_history(session: str, history: list):
    Path("chats").mkdir(exist_ok=True)
    (Path("chats") / f"{session}.json").write_text(json.dumps(history, indent=2, ensure_ascii=False))


def build_params(**kwargs):
    result = {}
    for key, value in kwargs.items():
        if value:
            result[key] = value
    return result

def save_generated_image(data: bytes, mime_type: str, index: int):
    Path("images").mkdir(exist_ok=True)
    ext = mime_type.removeprefix("image/") or "jpg"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path("images") / f"grok-{stamp}-{index}.{ext}"
    suffix = 2
    while path.exists():
        path = Path("images") / f"grok-{stamp}-{index}-{suffix}.{ext}"
        suffix += 1
    path.write_bytes(data)
    return path
