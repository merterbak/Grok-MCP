import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from xai_sdk import Client
from xai_sdk.chat import user, system, assistant, image, file
from xai_sdk.tools import web_search as xai_web_search, x_search as xai_x_search, code_execution
from .utils import encode_image_to_base64, encode_video_to_base64, build_params, XAI_API_KEY, load_history, save_history

mcp = FastMCP(name="Grok MCP Server")
READONLY = ToolAnnotations(readOnlyHint=True)

# Note: Tools return strings not dicts because if you return a dict it shows up as hard to read raw JSON (lines all side by side for result text output) in the Claude UI and Claude Code.

# To Claude: return output URLs as clickable links

@mcp.tool()
async def chat(
    prompt: str,
    session: Optional[str] = None,
    model: str = "grok-4-1-fast-reasoning",
    system_prompt: Optional[str] = None,
):
    history = load_history(session) if session else []

    client = Client(api_key=XAI_API_KEY)
    grok = client.chat.create(model=model)
    if system_prompt:
        grok.append(system(system_prompt))

    for message in history:
        if message["role"] == "user":
            grok.append(user(message["content"]))
        elif message["role"] == "assistant":
            grok.append(assistant(message["content"]))

    grok.append(user(prompt))
    response = grok.sample()
    client.close()

    if session:
        history.append({"role": "user", "content": prompt, "time": datetime.now().strftime("%d.%m.%Y %H:%M:%S")})
        history.append({"role": "assistant", "content": response.content, "time": datetime.now().strftime("%d.%m.%Y %H:%M:%S")})
        save_history(session, history)

    return response.content


@mcp.tool(annotations=READONLY)
async def list_chat_sessions():
    Path("chats").mkdir(exist_ok=True)
    sessions = sorted(Path("chats").glob("*.json"))
    if not sessions:
        return "No chat sessions found."
    result = ["**Chat Sessions:**\n"]
    for s in sessions:
        history = json.loads(s.read_text())
        turns = len(history) // 2
        last = history[-1]["time"] if history else "empty"
        result.append(f"- `{s.stem}` — {turns} turn(s), last: {last}")
    return "\n".join(result)


@mcp.tool(annotations=READONLY)
async def get_chat_history(session: str = "default"):
    history = load_history(session)
    if not history:
        return f"No history found for session `{session}`."
    result = [f"**Chat History: `{session}`**\n"]
    for message in history:
        role = message["role"].capitalize()
        time = message["time"]
        result.append(f"**[{time}] {role}:** {message['content']}\n")
    return "\n".join(result)


@mcp.tool()
async def clear_chat_history(session: str = "default"):
    path = Path("chats") / f"{session}.json"
    if not path.exists():
        return f"No session `{session}` found."
    path.unlink()
    return f"Cleared history for session `{session}`."


@mcp.tool(annotations=READONLY)
async def list_models():

    client = Client(api_key=XAI_API_KEY)
    models_info = []

    models_info.append("# Language Models\n")
    for m in client.models.list_language_models():
        date = m.created.ToDatetime().strftime('%d %b %Y')
        inp = m.prompt_text_token_price / 10000
        out = m.completion_text_token_price / 10000
        models_info.append(f"**{m.name}** — {date}")
        models_info.append(f"  Input: ${inp:g}/M · Output: ${out:g}/M\n")

    models_info.append("# Image Generation Models\n")
    for m in client.models.list_image_generation_models():
        date = m.created.ToDatetime().strftime('%d %b %Y')
        price = m.image_price / 10000000000
        models_info.append(f"**{m.name}** — {date}")
        models_info.append(f"  ${price:g} per image\n")

    client.close()
    return "\n".join(models_info)


@mcp.tool()
async def generate_image(
    prompt: str,
    model: str = "grok-imagine-image",
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    n: int = 1,
    image_format: str = "url",
    aspect_ratio: Optional[str] = None,
    resolution: Optional[str] = None
):
    
    client = Client(api_key=XAI_API_KEY)
    
    params = {"model": model, "prompt": prompt, "n": n, "image_format": image_format}
    
    if image_path:
        base64_string = encode_image_to_base64(image_path)
        ext = Path(image_path).suffix.lower().replace('.', '')
        params["image_url"] = f"data:image/{ext};base64,{base64_string}"
    elif image_url:
        params["image_url"] = image_url
    
    if aspect_ratio:
        params["aspect_ratio"] = aspect_ratio
    if resolution:
        params["resolution"] = resolution
    
    images = client.image.sample_batch(**params)
    client.close()

    result = ["## Generated Image(s)"]
    for i, img in enumerate(images, 1):
        result.append(f"\n**Image {i}:** {img.url}")
        if img.prompt and img.prompt != prompt:
            result.append(f"*Revised prompt:* {img.prompt}")
    return "\n".join(result)


@mcp.tool()
async def generate_video(
    prompt: str,
    model: str = "grok-imagine-video",
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    video_path: Optional[str] = None,
    video_url: Optional[str] = None,
    duration: Optional[int] = None,
    aspect_ratio: Optional[str] = None,
    resolution: Optional[str] = None,
    reference_image_urls: Optional[List[str]] = None
):
    
    client = Client(api_key=XAI_API_KEY)
    
    params = {
        "model": model,
        "prompt": prompt
    }
    
    if image_path:
        base64_string = encode_image_to_base64(image_path)
        ext = Path(image_path).suffix.lower().replace('.', '')
        params["image_url"] = f"data:image/{ext};base64,{base64_string}"
    elif image_url:
        params["image_url"] = image_url
    
    if video_path:
        base64_string = encode_video_to_base64(video_path)
        ext = Path(video_path).suffix.lower().replace('.', '')
        params["video_url"] = f"data:video/{ext};base64,{base64_string}"
    elif video_url:
        params["video_url"] = video_url
    
    if duration:
        params["duration"] = duration
    if aspect_ratio:
        params["aspect_ratio"] = aspect_ratio
    if resolution:
        params["resolution"] = resolution
    if reference_image_urls:
        params["reference_image_urls"] = reference_image_urls

    response = client.video.generate(**params)
    client.close()

    result = [f"## Generated Video\n\n**URL:** {response.url}"]
    if hasattr(response, 'duration') and response.duration:
        result.append(f"**Duration:** {response.duration}s")
    return "\n".join(result)

@mcp.tool()
async def chat_with_vision(
    prompt: str,
    session: Optional[str] = None,
    model: str = "grok-4-1-fast-reasoning",
    image_paths: Optional[List[str]] = None,
    image_urls: Optional[List[str]] = None,
    detail: str = "auto"
):
    history = load_history(session) if session else []

    client = Client(api_key=XAI_API_KEY)
    chat = client.chat.create(model=model, store_messages=False)

    for message in history:
        if message["role"] == "user":
            chat.append(user(message["content"]))
        elif message["role"] == "assistant":
            chat.append(assistant(message["content"]))

    user_content = []
    if image_paths:
        for path in image_paths:
            ext = Path(path).suffix.lower().replace('.', '')
            if ext not in ["jpg", "jpeg", "png"]:
                raise ValueError(f"Unsupported image type: {ext}")
            base64_img = encode_image_to_base64(path)
            user_content.append(image(image_url=f"data:image/{ext};base64,{base64_img}", detail=detail))

    if image_urls:
        for url in image_urls:
            user_content.append(image(image_url=url, detail=detail))

    user_content.append(prompt)
    chat.append(user(*user_content))
    response = chat.sample()
    client.close()

    if session:
        history.append({"role": "user", "content": prompt, "time": datetime.now().strftime("%d.%m.%Y %H:%M:%S")})
        history.append({"role": "assistant", "content": response.content, "time": datetime.now().strftime("%d.%m.%Y %H:%M:%S")})
        save_history(session, history)

    return response.content

@mcp.tool(annotations=READONLY)
async def web_search(
    prompt: str,
    model: str = "grok-4-1-fast-reasoning",
    allowed_domains: Optional[List[str]] = None,
    excluded_domains: Optional[List[str]] = None,
    enable_image_understanding: bool = False,
    include_inline_citations: bool = False,
    max_turns: Optional[int] = None
):

    if allowed_domains and excluded_domains:
        raise ValueError("Cannot specify both allowed_domains and excluded_domains")
    if allowed_domains and len(allowed_domains) > 5:
        raise ValueError("allowed_domains max 5")
    if excluded_domains and len(excluded_domains) > 5:
        raise ValueError("excluded_domains max 5")
    
    client = Client(api_key=XAI_API_KEY)
    
    tool_params = build_params(
        allowed_domains=allowed_domains,
        excluded_domains=excluded_domains,
        enable_image_understanding=enable_image_understanding,
    )
    
    include_options = []
    if include_inline_citations:
        include_options.append("inline_citations")
    
    chat_params = {"model": model, "tools": [xai_web_search(**tool_params)]}
    if include_options:
        chat_params["include"] = include_options
    if max_turns:
        chat_params["max_turns"] = max_turns
    
    chat = client.chat.create(**chat_params)
    chat.append(user(prompt))
    response = chat.sample()
    
    client.close()

    result = [response.content]
    if response.citations:
        result.append("\n\n**Sources:**")
        for url in response.citations:
            result.append(f"- {url}")
    return "\n".join(result)


@mcp.tool(annotations=READONLY)
async def x_search(
    prompt: str,
    model: str = "grok-4-1-fast-reasoning",
    allowed_x_handles: Optional[List[str]] = None,
    excluded_x_handles: Optional[List[str]] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    enable_image_understanding: bool = False,
    enable_video_understanding: bool = False,
    include_inline_citations: bool = False,
    max_turns: Optional[int] = None
):

    if allowed_x_handles and excluded_x_handles:
        raise ValueError("Cannot specify both allowed_x_handles and excluded_x_handles")
    if allowed_x_handles and len(allowed_x_handles) > 10:
        raise ValueError("allowed_x_handles max 10")
    if excluded_x_handles and len(excluded_x_handles) > 10:
        raise ValueError("excluded_x_handles max 10")
    
    client = Client(api_key=XAI_API_KEY)
    
    tool_params = build_params(
        allowed_x_handles=allowed_x_handles,
        excluded_x_handles=excluded_x_handles,
        from_date=datetime.strptime(from_date, "%d-%m-%Y") if from_date else None,
        to_date=datetime.strptime(to_date, "%d-%m-%Y") if to_date else None,
        enable_image_understanding=enable_image_understanding,
        enable_video_understanding=enable_video_understanding,
    )
    
    include_options = []
    if include_inline_citations:
        include_options.append("inline_citations")
    
    chat_params = {"model": model, "tools": [xai_x_search(**tool_params)]}
    if include_options:
        chat_params["include"] = include_options
    if max_turns:
        chat_params["max_turns"] = max_turns
    
    chat = client.chat.create(**chat_params)
    chat.append(user(prompt))
    response = chat.sample()
    
    client.close()

    result = [response.content]
    if response.citations:
        result.append("\n\n**Sources:**")
        for url in response.citations:
            result.append(f"- {url}")
    return "\n".join(result)


@mcp.tool()
async def code_executor(
    prompt: str,
    model: str = "grok-4-1-fast-reasoning",
    max_turns: Optional[int] = None
):
    client = Client(api_key=XAI_API_KEY)
    
    chat_params = {"model": model, "tools": [code_execution()], "include": ["code_execution_call_output"]}
    if max_turns:
        chat_params["max_turns"] = max_turns
    
    chat = client.chat.create(**chat_params)
    chat.append(user(prompt))
    response = chat.sample()
    
    client.close()

    result = [response.content]
    if response.tool_outputs:
        result.append("\n\n**Code Output(s):**")
        for output in response.tool_outputs:
            result.append(f"```\n{output.message.content}\n```")
    return "\n".join(result)


@mcp.tool()
async def grok_agent(
    prompt: str,
    session: Optional[str] = None,
    model: str = "grok-4-1-fast-reasoning",
    file_ids: Optional[List[str]] = None,
    image_urls: Optional[List[str]] = None,
    image_paths: Optional[List[str]] = None,
    use_web_search: bool = False,
    use_x_search: bool = False,
    use_code_execution: bool = False,
    allowed_domains: Optional[List[str]] = None,
    excluded_domains: Optional[List[str]] = None,
    allowed_x_handles: Optional[List[str]] = None,
    excluded_x_handles: Optional[List[str]] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    enable_image_understanding: bool = False,
    enable_video_understanding: bool = False,
    include_inline_citations: bool = False,
    system_prompt: Optional[str] = None,
    max_turns: Optional[int] = None
):
    history = load_history(session) if session else []

    client = Client(api_key=XAI_API_KEY)
    
    tools = []
    if use_web_search:
        web_params = build_params(
            allowed_domains=allowed_domains,
            excluded_domains=excluded_domains,
            enable_image_understanding=enable_image_understanding,
        )
        tools.append(xai_web_search(**web_params))
    
    if use_x_search:
        x_params = build_params(
            allowed_x_handles=allowed_x_handles,
            excluded_x_handles=excluded_x_handles,
            from_date=datetime.strptime(from_date, "%d-%m-%Y") if from_date else None,
            to_date=datetime.strptime(to_date, "%d-%m-%Y") if to_date else None,
            enable_image_understanding=enable_image_understanding,
            enable_video_understanding=enable_video_understanding,
        )
        tools.append(xai_x_search(**x_params))
    
    if use_code_execution:
        tools.append(code_execution())
    
    include_options = ["code_execution_call_output"]
    if include_inline_citations:
        include_options.append("inline_citations")
    
    chat_params = {"model": model, "include": include_options}
    if tools:
        chat_params["tools"] = tools
    if max_turns:
        chat_params["max_turns"] = max_turns
    
    chat = client.chat.create(**chat_params)

    if system_prompt:
        chat.append(system(system_prompt))

    for message in history:
        if message["role"] == "user":
            chat.append(user(message["content"]))
        elif message["role"] == "assistant":
            chat.append(assistant(message["content"]))

    content_items = []
    
    if file_ids:
        for fid in file_ids:
            content_items.append(file(fid))
    
    if image_urls:
        for url in image_urls:
            content_items.append(image(image_url=url))
    
    if image_paths:
        for path in image_paths:
            ext = Path(path).suffix.lower().replace('.', '')
            base64_img = encode_image_to_base64(path)
            content_items.append(image(image_url=f"data:image/{ext};base64,{base64_img}"))
    
    content_items.append(prompt)
    chat.append(user(*content_items))
    response = chat.sample()
    client.close()

    if session:
        history.append({"role": "user", "content": prompt, "time": datetime.now().strftime("%d.%m.%Y %H:%M:%S")})
        history.append({"role": "assistant", "content": response.content, "time": datetime.now().strftime("%d.%m.%Y %H:%M:%S")})
        save_history(session, history)

    result = [response.content]
    if response.citations:
        result.append("\n\n**Sources:**")
        for url in response.citations:
            result.append(f"- {url}")
    return "\n".join(result)


@mcp.tool()
async def stateful_chat(
    prompt: str,
    model: str = "grok-4-1-fast-reasoning",
    response_id: Optional[str] = None,
    system_prompt: Optional[str] = None
):
    client = Client(api_key=XAI_API_KEY)
    
    chat_params = {"model": model, "store_messages": True}
    if response_id:
        chat_params["previous_response_id"] = response_id
    
    chat = client.chat.create(**chat_params)
    if system_prompt and not response_id:
        chat.append(system(system_prompt))
    chat.append(user(prompt))
    
    response = chat.sample()
    client.close()
    
    return f"{response.content}\n\n**Response ID:** `{response.id}`"


@mcp.tool(annotations=READONLY)
async def retrieve_stateful_response(response_id: str):
    client = Client(api_key=XAI_API_KEY)
    responses = client.chat.get_stored_completion(response_id)
    client.close()
    if not responses:
        return f"No response found for id {response_id}"
    response = responses[0] if isinstance(responses, list) else responses
    return f"{response.content}\n\n**Response ID:** `{response.id}`"


@mcp.tool()
async def delete_stateful_response(response_id: str):
    client = Client(api_key=XAI_API_KEY)
    client.chat.delete_stored_completion(response_id)
    client.close()
    return f"Deleted response `{response_id}`"


@mcp.tool()
async def upload_file(
    file_path: str,
    filename: Optional[str] = None
):
    client = Client(api_key=XAI_API_KEY)
    
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found {file_path}")
    
    uploaded = client.files.upload(file_path)
    client.close()
    
    return f"**File uploaded successfully**\n- **File ID:** `{uploaded.id}`\n- **Filename:** {uploaded.filename}\n- **Size:** {uploaded.size} bytes"


@mcp.tool(annotations=READONLY)
async def list_files(
    limit: int = 100,
    order: str = "desc",
    sort_by: str = "created_at"
):
    client = Client(api_key=XAI_API_KEY)
    response = client.files.list(limit=limit, order=order, sort_by=sort_by)
    client.close()
    
    if not response.data:
        return "No files found."
    result = ["**Files:**\n"]
    for f in response.data:
        result.append(f"- `{f.id}` — {f.filename} ({f.size} bytes)")
    return "\n".join(result)


@mcp.tool(annotations=READONLY)
async def get_file(file_id: str):
    client = Client(api_key=XAI_API_KEY)
    file_info = client.files.get(file_id)
    client.close()
    
    return f"**File ID:** `{file_info.id}`\n**Filename:** {file_info.filename}\n**Size:** {file_info.size} bytes\n**Created:** {file_info.created_at}"


@mcp.tool(annotations=READONLY)
async def get_file_content(file_id: str, max_bytes: int = 500000):
    client = Client(api_key=XAI_API_KEY)
    content = client.files.content(file_id)
    client.close()
    
    total_size = len(content)
    truncated = total_size > max_bytes
    
    if truncated:
        content = content[:max_bytes]
    
    text = content.decode("utf-8", errors="replace")
    note = f"\n\n*[Truncated: showing {len(content):,} of {total_size:,} bytes]*" if truncated else ""
    return text + note


@mcp.tool()
async def delete_file(file_id: str):
    client = Client(api_key=XAI_API_KEY)
    delete_response = client.files.delete(file_id)
    client.close()
    
    return f"Deleted file `{delete_response.id}`"


@mcp.tool()
async def chat_with_files(
    prompt: str,
    session: Optional[str] = None,
    model: str = "grok-4-1-fast-reasoning",
    file_ids: List[str] = None,
    system_prompt: Optional[str] = None
):
    history = load_history(session) if session else []

    client = Client(api_key=XAI_API_KEY)
    chat = client.chat.create(model=model)

    if system_prompt:
        chat.append(system(system_prompt))

    for message in history:
        if message["role"] == "user":
            chat.append(user(message["content"]))
        elif message["role"] == "assistant":
            chat.append(assistant(message["content"]))

    file_attachments = [file(fid) for fid in file_ids]
    chat.append(user(prompt, *file_attachments))
    response = chat.sample()
    client.close()

    if session:
        history.append({"role": "user", "content": prompt, "time": datetime.now().strftime("%d.%m.%Y %H:%M:%S")})
        history.append({"role": "assistant", "content": response.content, "time": datetime.now().strftime("%d.%m.%Y %H:%M:%S")})
        save_history(session, history)

    result = [response.content]
    if response.citations:
        result.append("\n\n**Sources:**")
        for url in response.citations:
            result.append(f"- {url}")
    return "\n".join(result)


def main():
    mcp.run(transport='stdio')


if __name__ == "__main__":
    main()
