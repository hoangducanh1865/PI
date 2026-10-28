"""
Move the working data between machines and accounts like code moves through GitHub: push <BASE>/data to a private
Hugging Face dataset repo, pull it anywhere.

Usage (any platform, from the repo root):
    python scripts/data_sync.py pull [--base <BASE>] [--only <PATTERN> ...]
    python scripts/data_sync.py push [--base <BASE>] [--only <PATTERN> ...] [--message "<what changed>"]

<BASE> holds data/ (default per platform: Kaggle /kaggle/working, Colab /content/drive/MyDrive/it/fpt, otherwise the
folder two levels above this repo). Synced: data/<task>/ of every task and data/datasets/processed/ (packed datasets);
not synced: data/datasets/raw/ (public downloads, 76 GB, fetched again when needed). Push uploads only new or changed
files, as one commit; pull downloads only what is missing or changed. --only limits either, e.g. --only 'video_eval/**'.
Tokens: HF_WRITE_TOKEN (push) and HF_TOKEN (pull), from the environment or the repo's .env; never printed.

Author: Duc Anh Hoang
"""

import argparse
import importlib.util
import os

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "hoangducanh1865/fpt-data"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root, holds .env
SYNCED = ["lang_grasp/**", "sim2real_worldmodel/**", "video_eval/**", "datasets/processed/**"]


def default_base() -> str:
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "/kaggle/working"
    try:
        if importlib.util.find_spec("google.colab") is not None:
            return "/content/drive/MyDrive/it/fpt"
    except ModuleNotFoundError:
        pass
    return os.path.dirname(os.path.dirname(ROOT))


def token(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    with open(os.path.join(ROOT, ".env")) as f:
        for line in f:
            key, _, value = line.strip().partition("=")
            if key == name:
                return value.strip("\"'")
    raise SystemExit(f"{name} is not set (environment or .env)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("pull", "push"))
    parser.add_argument("--base", default=default_base())
    parser.add_argument("--only", nargs="+", help="patterns relative to data/, default everything synced")
    parser.add_argument("--message", default="sync data", help="push: commit message")
    args = parser.parse_args()
    data = os.path.join(args.base, "data")

    if args.action == "push":
        api = HfApi(token=token("HF_WRITE_TOKEN"))
        api.create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)
        commit = api.upload_folder(
            repo_id=REPO_ID, repo_type="dataset", folder_path=data, allow_patterns=args.only or SYNCED,
            commit_message=args.message,
        )  # fmt: skip
        print(f"pushed {data} -> {REPO_ID}: {commit.commit_url}")
    else:
        snapshot_download(
            REPO_ID, repo_type="dataset", local_dir=data, allow_patterns=args.only or SYNCED, token=token("HF_TOKEN"),
        )  # fmt: skip
        print(f"pulled {REPO_ID} -> {data}")


if __name__ == "__main__":
    main()
