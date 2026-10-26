"""
Ask a vision-language model, through an OpenAI-compatible endpoint, where each exported failure lands.

Usage (on the Mac or wherever the endpoint is reachable; the images come from analyze_errors.py --export):
    python scripts/judge_failures.py --dir <EXPORT_DIR> --model <MODEL> [--base-url URL] [--key-env NAME] [--workers 8]
    e.g. python scripts/judge_failures.py --dir export --model ag/gemini-3.8-flash-medium \\
             --base-url http://localhost:20128/v1 --key-env GEMINI_API_KEY

Each image shows a scene with one predicted grasp rectangle and its centre in yellow; the judge gets the image and the
prompt and answers in JSON: the object the prompt asks for, the object under the yellow centre (or "none"), whether it
is the requested object, and whether the rectangle is on the requested part. The API key is read from the environment
variable --key-env, or from the repo's .env; it is never printed. Writes <EXPORT_DIR>/judge_<model>.csv and prints the
shares per failure type.

Author: Duc Anh Hoang
"""

import argparse
import base64
import collections
import csv
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
QUESTION = """The image shows a scene and ONE predicted robot grasp: a yellow rectangle with a yellow dot at its centre.
The robot was told: "{prompt}"
Look at what lies under the yellow dot. Answer with JSON only, no other text:
{{"requested_object": "<the object the instruction asks for>",
  "object_under_dot": "<the object under the yellow dot, or \\"none\\" if it is table, floor or background>",
  "on_requested_object": <true if the dot is on the requested object, else false>,
  "on_requested_part": <true if it is on the requested part, false if not, null if no part is named>}}"""


def api_key(name):
    if os.environ.get(name):
        return os.environ[name]
    with open(os.path.join(ROOT, ".env")) as f:
        for line in f:
            key, _, value = line.strip().partition("=")
            if key == name:
                return value.strip("\"'")
    raise SystemExit(f"{name} is not set (environment or .env)")


def ask(base_url, key, model, image_path, prompt):
    with open(image_path, "rb") as f:
        image = base64.b64encode(f.read()).decode()
    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": QUESTION.format(prompt=prompt)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
                ],
            }
        ],
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    text = json.load(urllib.request.urlopen(request, timeout=180))["choices"][0]["message"]["content"]
    return json.loads(re.search(r"\{.*\}", text, re.S).group(0))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, help="export folder of analyze_errors.py (manifest.csv + images)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://localhost:20128/v1")
    parser.add_argument("--key-env", default="GEMINI_API_KEY")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    key = api_key(args.key_env)
    with open(os.path.join(args.dir, "manifest.csv")) as f:
        rows = list(csv.DictReader(f))

    def judge(row):
        try:
            return row, ask(args.base_url, key, args.model, os.path.join(args.dir, row["file"]), row["prompt"])
        except Exception as e:  # one bad answer must not stop the batch
            return row, {"error": str(e)[:200]}

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(judge, rows))
    fields = ["requested_object", "object_under_dot", "on_requested_object", "on_requested_part", "error"]
    out = os.path.join(args.dir, f"judge_{args.model.replace('/', '_')}.csv")
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([*rows[0].keys(), *fields])
        for row, answer in results:
            writer.writerow([*row.values(), *(answer.get(k) for k in fields)])
    by_type = collections.defaultdict(collections.Counter)
    for row, answer in results:
        if "error" in answer:
            verdict = "error"
        elif answer.get("on_requested_object"):
            verdict = "on requested object"
        elif str(answer.get("object_under_dot", "")).lower() in ("none", ""):
            verdict = "on background"
        else:
            verdict = "on another object"
        by_type[row["type"]][verdict] += 1
    for kind, counts in by_type.items():
        total = sum(counts.values())
        print(f"{kind} ({total}): " + ", ".join(f"{v} {n / total:.0%}" for v, n in counts.most_common()))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
