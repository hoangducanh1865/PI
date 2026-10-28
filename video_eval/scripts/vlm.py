"""
Ask a vision-language model one question about one image through any OpenAI-compatible chat endpoint and parse its
JSON answer. Shared by check_ee_points.py and eval_vlm.py. The API key comes from an environment variable or the repo's
.env and is never printed.

Author: Duc Anh Hoang
"""

import base64
import io
import json
import os
import re
import urllib.request

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, holds .env


def api_key(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    with open(os.path.join(REPO, ".env")) as f:
        for line in f:
            key, _, value = line.strip().partition("=")
            if key == name:
                return value.strip("\"'")
    raise SystemExit(f"{name} is not set (environment or .env)")


def png(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def ask(base_url: str, key: str, model: str, question: str, image_b64: str) -> dict:
    """The model's JSON answer, or {"error": ...} after three failed tries."""
    picture = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
    content = [{"type": "text", "text": question}, picture]
    body = {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0, "stream": False}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    request = urllib.request.Request(f"{base_url}/chat/completions", data=json.dumps(body).encode(), headers=headers)
    error = ""
    for _ in range(3):
        try:
            raw = urllib.request.urlopen(request, timeout=180).read().decode()
            if raw.startswith("data:"):  # some providers stream (SSE) regardless: join the content deltas
                lines = [line[5:] for line in raw.splitlines() if line.startswith("data:") and "[DONE]" not in line]
                chunks = [json.loads(line) for line in lines]
                text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c.get("choices"))
            else:
                text = json.loads(raw)["choices"][0]["message"]["content"]
            return json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        except Exception as e:  # network errors and unparsable answers: retry, then record
            error = str(e)
    return {"error": error}
