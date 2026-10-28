"""
Check the automatically chosen end-effector points with vision-language judges: for each image drawn by
scripts/eval_motion.py --draw (frames 0 and 15, the point in red), ask whether the point lies on the robot's gripper or
wrist in both frames. A clip is verified when every judge says yes.

Usage (on the machine that reaches the endpoint):
    python scripts/check_ee_points.py --dir <DRAW_DIR> --models <MODEL> <MODEL> [--base-url URL] [--key-env NAME]
    e.g. python scripts/check_ee_points.py --dir <DRAW_DIR> --models ag/gemini-3.8-flash-low cx/gpt-6.1-sol \\
             --base-url http://localhost:20128/v1 --key-env GEMINI_API_KEY

Any OpenAI-compatible chat endpoint; the API key is read from the environment variable --key-env, or from the repo's
.env, and is never printed. Writes <DRAW_DIR>/vlm_check.json (every answer) and <DRAW_DIR>/verified_clips.json, and
prints each judge's yes count, the agreement and the clips not verified.

Author: Duc Anh Hoang
"""

import argparse
import base64
import glob
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, holds .env
QUESTION = (
    "The image shows two frames of a robot video side by side (first frame left, last frame right). A red dot marks "
    "one tracked point in each frame and a red line its path. Is the red dot on the robot's gripper or wrist (end "
    'effector) in BOTH frames? Answer JSON only: {"left": "<what the dot is on in the left frame>", '
    '"right": "<the same for the right frame>", "on_gripper_both": true or false}'
)


def api_key(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    with open(os.path.join(REPO, ".env")) as f:
        for line in f:
            key, _, value = line.strip().partition("=")
            if key == name:
                return value.strip("\"'")
    raise SystemExit(f"{name} is not set (environment or .env)")


def ask(base_url: str, key: str, model: str, path: str) -> dict:
    with open(path, "rb") as f:
        image = base64.b64encode(f.read()).decode()
    picture = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}
    content = [{"type": "text", "text": QUESTION}, picture]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--base-url", default="http://localhost:20128/v1")
    parser.add_argument("--key-env", default="GEMINI_API_KEY")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    key = api_key(args.key_env)

    paths = sorted(glob.glob(os.path.join(args.dir, "*.png")))
    answers = {}
    with ThreadPoolExecutor(args.workers) as pool:
        jobs = {(m, p): pool.submit(ask, args.base_url, key, m, p) for m in args.models for p in paths}
        for (model, path), job in jobs.items():
            answers.setdefault(os.path.basename(path)[:-4], {})[model] = job.result()
    verified = sorted(c for c, a in answers.items() if all(v.get("on_gripper_both") is True for v in a.values()))
    with open(os.path.join(args.dir, "vlm_check.json"), "w") as f:
        json.dump(answers, f, indent=1)
    with open(os.path.join(args.dir, "verified_clips.json"), "w") as f:
        json.dump(verified, f)

    for model in args.models:
        votes = [a[model].get("on_gripper_both") for a in answers.values()]
        print(f"{model}: yes {sum(v is True for v in votes)} / {len(votes)}, errors {sum(v is None for v in votes)}")
    agree = sum(len({a[m].get("on_gripper_both") for m in args.models}) == 1 for a in answers.values())
    print(f"agreement {agree} / {len(answers)}; verified {len(verified)}")
    for clip in sorted(set(answers) - set(verified)):
        print(clip, {m: answers[clip][m].get("on_gripper_both") for m in args.models})


if __name__ == "__main__":
    main()
