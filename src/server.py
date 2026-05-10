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
# Every tool has a docstring so Claude understands what it does better.
# To Claude: return output URLs as clickable links

@mcp.tool()
async def chat(
    prompt: str,
    session: Optional[str] = None,
    model: str = "grok-4.3",
    system_prompt: Optional[str] = None,
    agent_count: Optional[int] = None,
):
    """Send a text prompt to a Grok model and return its reply.

    Replays prior turns from `chats/{session}.json` when a session is given,
    then appends the new user message and saves the round trip.

    Args:
        prompt: User message to send to the model.
        session: Optional session name. Loads and appends history to `chats/{session}.json`.
        model: Grok model id (default `grok-4.3`).
        system_prompt: Optional system instruction prepended to the conversation.
        agent_count: 4 or 16. Only valid with `grok-4.20-multi-agent` for multi-agent research.

    Returns:
        The assistant's reply text.
    """
    history = load_history(session) if session else []

    client = Client(api_key=XAI_API_KEY)
    chat_params = {"model": model}
    if agent_count:
        chat_params["agent_count"] = agent_count
    grok = client.chat.create(**chat_params)
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
    """List all local chat sessions stored under `chats/`.

    Returns:
        Markdown list of session names with turn counts and last-message timestamps,
        or a placeholder message when no sessions exist.
    """
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
    """Return the full message history for a local chat session.

    Args:
        session: Session name to load from `chats/{session}.json` (default `default`).

    Returns:
        Formatted transcript with timestamps and roles, or a not-found message.
    """
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
    """Delete the local history file for a chat session.

    Only removes the client-side JSON file. Server-side stored responses are untouched.

    Args:
        session: Session name whose `chats/{session}.json` file should be deleted.

    Returns:
        Confirmation string or a not-found message.
    """
    path = Path("chats") / f"{session}.json"
    if not path.exists():
        return f"No session `{session}` found."
    path.unlink()
    return f"Cleared history for session `{session}`."


@mcp.tool(annotations=READONLY)
async def list_models():
    """List all Grok language and image models with live pricing from the xAI API.

    Returns:
        Markdown sections for language models (input/output $/M tokens) and
        image generation models ($/image), each with the model's release date.
    """
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
    image_paths: Optional[List[str]] = None,
    image_urls: Optional[List[str]] = None,
    n: int = 1,
    image_format: str = "url",
    aspect_ratio: Optional[str] = None,
    resolution: Optional[str] = None,
):
    """Generate new images or edit existing ones with Grok Imagine.

    Pass `image_paths` and/or `image_urls` to edit images or use them as
    visual references. Multiple references are combined in a single call.

    Args:
        prompt: Image description, or the edit instruction when references are provided.
        model: Image model (`grok-imagine-image` or `grok-imagine-image-pro`).
        image_paths: Local image files (JPG/PNG) used as edit sources or references.
        image_urls: Public image URLs used as edit sources or references.
        n: Number of images to generate (1–10).
        image_format: `"url"` (default) or `"base64"`.
        aspect_ratio: Aspect ratio like `"16:9"`, `"1:1"`, or `"9:16"`.
        resolution: `"1k"` or `"2k"`.

    Returns:
        Markdown block with each generated image URL and any revised prompt.
    """
    client = Client(api_key=XAI_API_KEY)

    params = {"model": model, "prompt": prompt, "n": n, "image_format": image_format}

    refs = []
    if image_paths:
        for path in image_paths:
            base64_string = encode_image_to_base64(path)
            ext = Path(path).suffix.lower().replace('.', '')
            refs.append(f"data:image/{ext};base64,{base64_string}")
    if image_urls:
        refs.extend(image_urls)

    if refs:
        params["image_urls"] = refs

    if aspect_ratio:
        params["aspect_ratio"] = aspect_ratio
    if resolution:
        params["resolution"] = resolution

    images = client.image.sample_batch(**params)
    client.close()

    result = ["## Generated Image(s)\n\n"]
    for i, img in enumerate(images, 1):
        result.append(f"\n**Image {i}:** {img.url}\n\n")
        if img.prompt and img.prompt != prompt:
            result.append(f"*Revised prompt:* {img.prompt}\n\n")
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
    resolution: Optional[str] = None
):
    """Generate or edit videos with Grok Imagine.

    Text-to-video by default. Provide an image to animate (image-to-video), or
    a source video to edit. Only one mode per call. Generation polls
    synchronously (xAI's default timeout is 10 minutes).

    Args:
        prompt: Video description, or the edit instruction for video editing.
        model: Video model (default `grok-imagine-video`).
        image_path: Local image to use as the starting frame.
        image_url: Public image URL to use as the starting frame.
        video_path: Local video to edit (max 20 MB, .mp4, ≤ 8.7s).
        video_url: Public video URL to edit (.mp4, ≤ 8.7s).
        duration: Video length in seconds (1–15, ignored when editing).
        aspect_ratio: Aspect ratio like `"16:9"` or `"9:16"` (ignored when editing).
        resolution: `"480p"` or `"720p"` (ignored when editing).

    Returns:
        Markdown block with the generated video URL and actual duration.
    """
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

    response = client.video.generate(**params)
    client.close()

    return f"## Generated Video\n\n\n**URL:** {response.url}\n\n\n**Duration:** {response.duration}s\n\n"


@mcp.tool()
async def extend_video(
    prompt: str,
    video_url: str,
    model: str = "grok-imagine-video",
    duration: Optional[int] = None,
):
    """Extend an existing video with a follow-up prompt.

    Continues the source video seamlessly from its last frame. `duration` sets
    the length of the extension only, not the total output. For example, a
    10 second input plus `duration=5` yields a 15 second final video.

    Args:
        prompt: What should happen in the extended segment.
        video_url: Public URL of the source video (.mp4, 2–15 s).
        model: Video model (default `grok-imagine-video`).
        duration: Length of the extension in seconds (2–10, default 6).

    Returns:
        Markdown block with the extended video URL and total duration.
    """
    client = Client(api_key=XAI_API_KEY)

    params = {"model": model, "prompt": prompt, "video_url": video_url}
    if duration:
        params["duration"] = duration

    response = client.video.extend(**params)
    client.close()

    return f"## Extended Video\n\n\n**URL:** {response.url}\n\n\n**Duration:** {response.duration}s\n\n"


@mcp.tool()
async def chat_with_vision(
    prompt: str,
    session: Optional[str] = None,
    model: str = "grok-4.3",
    image_paths: Optional[List[str]] = None,
    image_urls: Optional[List[str]] = None,
    detail: str = "auto"
):
    """Analyze one or more images with a Grok vision model.

    Accepts local image paths and/or public URLs in the same call. Local images
    are sent as base64 data URIs (JPG/JPEG/PNG only, max 20 MiB each).

    Args:
        prompt: Question or instruction about the image(s).
        session: Optional session name for persistent history in `chats/{session}.json`.
        model: Vision-capable Grok model (default `grok-4.3`).
        image_paths: Local image file paths to analyze.
        image_urls: Public image URLs to analyze.
        detail: Image detail level. One of `"auto"`, `"low"`, or `"high"`.

    Returns:
        The model's textual answer about the image(s).
    """
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
    model: str = "grok-4.3",
    allowed_domains: Optional[List[str]] = None,
    excluded_domains: Optional[List[str]] = None,
    enable_image_understanding: bool = False,
    include_inline_citations: bool = False,
    max_turns: Optional[int] = None
):
    """Answer a query using agentic real-time web search.

    Grok browses the web across multiple turns, optionally inspecting images on
    pages, then synthesizes an answer with citations.

    Args:
        prompt: Search query or research question.
        model: Grok model used to drive the agent (default `grok-4.3`).
        allowed_domains: Restrict search to these domains (max 5, mutually exclusive with excluded).
        excluded_domains: Exclude these domains from search (max 5).
        enable_image_understanding: Let the agent analyze images it encounters.
        include_inline_citations: Embed `[1]`-style citation markers into the answer text.
        max_turns: Cap the agent's reasoning/tool turns.

    Returns:
        Markdown with the answer body followed by a `**Sources:**` list of cited URLs.
    """
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
    model: str = "grok-4.3",
    allowed_x_handles: Optional[List[str]] = None,
    excluded_x_handles: Optional[List[str]] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    enable_image_understanding: bool = False,
    enable_video_understanding: bool = False,
    include_inline_citations: bool = False,
    max_turns: Optional[int] = None
):
    """Answer a query using agentic search over X (Twitter).

    Searches posts, threads, and users on X. Can filter by handle allow/deny
    lists and a date range, and optionally analyze images/videos attached to posts.

    Args:
        prompt: Search query or question about X content.
        model: Grok model driving the agent (default `grok-4.3`).
        allowed_x_handles: Restrict search to these handles (max 10, mutually exclusive with excluded).
        excluded_x_handles: Exclude these handles (max 10).
        from_date: Inclusive start date as `DD-MM-YYYY`.
        to_date: Inclusive end date as `DD-MM-YYYY`.
        enable_image_understanding: Let the agent analyze images in posts.
        enable_video_understanding: Let the agent analyze videos in posts (X Search only).
        include_inline_citations: Embed `[1]`-style citation markers into the answer.
        max_turns: Cap the agent's reasoning/tool turns.

    Returns:
        Markdown with the answer body followed by a `**Sources:**` list of cited posts.
    """
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
    model: str = "grok-4.3",
    max_turns: Optional[int] = None
):
    """Solve a task by letting Grok run Python in a stateful sandbox.

    The agent iteratively writes and executes Python (with common scientific
    libraries) to arrive at a numeric, data, or analysis answer.

    Args:
        prompt: Task or question requiring computation.
        model: Grok model driving the agent (default `grok-4.3`).
        max_turns: Cap the number of reasoning/execution turns.

    Returns:
        Markdown with the final answer followed by each code execution block's stdout.
    """
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
    model: str = "grok-4.3",
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
    max_turns: Optional[int] = None,
    agent_count: Optional[int] = None,
):
    """All-in-one Grok agent combining files, vision, web/X search, and code execution.

    Enable any subset of tools and attach any mix of uploaded files and images.
    The agent decides which tools to use per turn. Supports optional local
    session history and multi-agent research via `agent_count`.

    Args:
        prompt: Task or question for the agent.
        session: Optional session name for persistent history in `chats/{session}.json`.
        model: Grok model driving the agent (default `grok-4.3`).
        file_ids: IDs of previously uploaded files to attach as context.
        image_urls: Public image URLs to attach.
        image_paths: Local image files to attach (sent as base64 data URIs).
        use_web_search: Enable the agentic web search tool.
        use_x_search: Enable the agentic X (Twitter) search tool.
        use_code_execution: Enable the Python code execution tool.
        allowed_domains: Web search allow-list (max 5, mutually exclusive with excluded).
        excluded_domains: Web search deny-list (max 5).
        allowed_x_handles: X search handle allow-list (max 10, mutually exclusive with excluded).
        excluded_x_handles: X search handle deny-list (max 10).
        from_date: X search inclusive start date as `DD-MM-YYYY`.
        to_date: X search inclusive end date as `DD-MM-YYYY`.
        enable_image_understanding: Let search tools analyze images they encounter.
        enable_video_understanding: Let X search analyze videos in posts.
        include_inline_citations: Embed `[1]`-style citation markers into the answer.
        system_prompt: Optional system instruction prepended to the conversation.
        max_turns: Cap the agent's reasoning/tool turns.
        agent_count: 4 or 16. Only valid with `grok-4.20-multi-agent`.

    Returns:
        Markdown with the answer body followed by a `**Sources:**` list when citations exist.
    """
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
    if agent_count:
        chat_params["agent_count"] = agent_count

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
    model: str = "grok-4.3",
    response_id: Optional[str] = None,
    system_prompt: Optional[str] = None
):
    """Continue a server-side stored conversation using xAI's deferred/stateful chat.

    The xAI API stores every turn so the client only needs to send the latest
    prompt plus `previous_response_id`. Omit `response_id` to start a new thread.

    Args:
        prompt: User message to append.
        model: Grok model id (default `grok-4.3`).
        response_id: ID of the previous response to continue from (omit to start fresh).
        system_prompt: Optional system instruction. Applied only on the first turn.

    Returns:
        Assistant reply followed by the new `**Response ID:**` to pass back next turn.
    """
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
    """Fetch a stored chat completion from xAI by its response ID.

    Args:
        response_id: ID returned by a prior `stateful_chat` call.

    Returns:
        The stored assistant reply and its `**Response ID:**`, or a not-found message.
    """
    client = Client(api_key=XAI_API_KEY)
    responses = client.chat.get_stored_completion(response_id)
    client.close()
    if not responses:
        return f"No response found for id {response_id}"
    response = responses[0] if isinstance(responses, list) else responses
    return f"{response.content}\n\n**Response ID:** `{response.id}`"


@mcp.tool()
async def delete_stateful_response(response_id: str):
    """Delete a stored chat completion from xAI's servers.

    Args:
        response_id: ID of the stored response to remove.

    Returns:
        Confirmation string with the deleted response ID.
    """
    client = Client(api_key=XAI_API_KEY)
    client.chat.delete_stored_completion(response_id)
    client.close()
    return f"Deleted response `{response_id}`"


@mcp.tool()
async def upload_file(file_path: str):
    """Upload a local file to xAI so it can be attached to later chats.

    Supported types include PDFs and text documents (see xAI file docs). The
    returned file ID can be passed to `chat_with_files` or `grok_agent`.

    Args:
        file_path: Absolute or relative path to the local file.

    Returns:
        Markdown block with the assigned file ID, filename, and size.
    """
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
    """List files previously uploaded to xAI.

    Args:
        limit: Maximum number of files to return (default 100).
        order: `"asc"` or `"desc"` sort order (default `"desc"`).
        sort_by: Field to sort by, such as `"created_at"`.

    Returns:
        Markdown list of files with ID, filename, and size, or a placeholder when empty.
    """
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
    """Fetch metadata for a single uploaded file.

    Args:
        file_id: ID returned by `upload_file`.

    Returns:
        Markdown block with the file's ID, filename, size, and creation time.
    """
    client = Client(api_key=XAI_API_KEY)
    file_info = client.files.get(file_id)
    client.close()
    
    return f"**File ID:** `{file_info.id}`\n**Filename:** {file_info.filename}\n**Size:** {file_info.size} bytes\n**Created:** {file_info.created_at}"


@mcp.tool(annotations=READONLY)
async def get_file_content(file_id: str, max_bytes: int = 500000):
    """Download the raw content of an uploaded file as text.

    Bytes are decoded as UTF-8 with replacement for invalid sequences. Output
    is truncated to `max_bytes` to avoid overwhelming the response.

    Args:
        file_id: ID of the uploaded file.
        max_bytes: Maximum bytes to return (default 500 000).

    Returns:
        File text, with a truncation note appended when the content exceeds `max_bytes`.
    """
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
    """Permanently delete an uploaded file from xAI.

    Args:
        file_id: ID of the file to remove.

    Returns:
        Confirmation string with the deleted file ID.
    """
    client = Client(api_key=XAI_API_KEY)
    delete_response = client.files.delete(file_id)
    client.close()
    
    return f"Deleted file `{delete_response.id}`"


@mcp.tool()
async def chat_with_files(
    prompt: str,
    file_ids: List[str],
    session: Optional[str] = None,
    model: str = "grok-4.3",
    system_prompt: Optional[str] = None
):
    """Chat with Grok using one or more previously uploaded files as context.

    Attaches the given `file_ids` to the user turn so Grok can read/quote their
    contents. Optional `session` persists local chat history across calls.

    Args:
        prompt: Question or instruction about the attached files.
        file_ids: IDs of files previously returned by `upload_file`.
        session: Optional session name for persistent history in `chats/{session}.json`.
        model: Grok model id (default `grok-4.3`).
        system_prompt: Optional system instruction prepended to the conversation.

    Returns:
        Assistant reply, followed by a `**Sources:**` list when the model cites URLs.
    """
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
