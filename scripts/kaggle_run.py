"""
Run one command on a Kaggle GPU from your terminal, in batch mode (no browser; the session ends with the command).

Usage (on the Mac or the PC, from any folder; needs the Kaggle CLI and its API token, see terminal_commands.md):
    python scripts/kaggle_run.py "<COMMAND>" --task <TASK> [--kernel <SLUG>] [--dataset <OWNER/SLUG>] [--gpu <SHAPE>]
        [--dry-run]
    e.g. python scripts/kaggle_run.py "python src/train.py experiment=grconvnet_clip" --task lang_grasp --kernel base

It fills the parameters cell of notebooks/kaggle.ipynb with the commit to run (git HEAD, which must be pushed), the
task folder (<TASK>/: its requirements.txt is installed and <COMMAND> runs in it) and <COMMAND>, then pushes it as a
new version of the private kernel <USER>/<SLUG> with a GPU and internet on. Use one slug per run to run several at
once. --dataset mounts a Kaggle Dataset under /kaggle/input/ (repeat it for more);
--dry-run writes the kernel folder and prints its path without pushing.

Author: Duc Anh Hoang
"""

import argparse
import json
import os
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
NOTEBOOK = os.path.join(ROOT, "notebooks", "kaggle.ipynb")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def kaggle_user() -> str:
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    for path in ("~/.kaggle/kaggle.json", "~/.config/kaggle/kaggle.json"):
        path = os.path.expanduser(path)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)["username"]
    raise SystemExit("Kaggle user unknown: set KAGGLE_USERNAME or put kaggle.json in ~/.kaggle/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", help="command run in the task folder on Kaggle")
    parser.add_argument("--task", required=True, help="task folder of the repo, e.g. lang_grasp")
    parser.add_argument("--kernel", default="fpt-entrance-test", help="kernel slug, one per concurrent run")
    parser.add_argument("--dataset", action="append", default=[], help="Kaggle Dataset OWNER/SLUG to mount")
    parser.add_argument("--gpu", default="NvidiaTeslaT4", help="Kaggle machine_shape (NvidiaTeslaT4 = 2x T4)")
    parser.add_argument("--dry-run", action="store_true", help="write the kernel folder, do not push")
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(ROOT, args.task, "requirements.txt")):
        raise SystemExit(f"{args.task}/requirements.txt not found: --task must be a task folder of the repo.")
    commit = git("rev-parse", "HEAD")
    if not git("branch", "-r", "--contains", commit):
        raise SystemExit(f"Commit {commit[:7]} is not pushed: Kaggle clones the code from GitHub, push it first.")
    if git("status", "--porcelain", "--", args.task, "scripts"):
        print(f"Warning: uncommitted changes are not sent, Kaggle runs commit {commit[:7]}.")

    with open(NOTEBOOK) as f:
        notebook = json.load(f)
    cell = next(c for c in notebook["cells"] if "parameters" in c["metadata"].get("tags", []))
    cell["source"] = [f"COMMIT = {commit!r}\n", f"TASK = {args.task!r}\n", f"COMMAND = {args.command!r}"]

    user = kaggle_user()
    folder = tempfile.mkdtemp(prefix=f"kaggle_{args.kernel}_")
    with open(os.path.join(folder, "kaggle.ipynb"), "w") as f:
        json.dump(notebook, f, indent=1)
    metadata = {
        "id": f"{user}/{args.kernel}",
        "title": args.kernel,
        "code_file": "kaggle.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": args.gpu,
        "dataset_sources": args.dataset,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    with open(os.path.join(folder, "kernel-metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    if args.dry_run:
        print(folder)
        return
    subprocess.run(["kaggle", "kernels", "push", "-p", folder], check=True)
    print(f"Pushed commit {commit[:7]}. Follow it with:  kaggle kernels status {user}/{args.kernel}")


if __name__ == "__main__":
    main()
