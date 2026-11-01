"""
Run one command on a Kaggle GPU from your terminal, in batch mode (no browser; the session ends with the command).

Usage (on the Mac or the PC, from any folder; needs the Kaggle CLI and its API token, see terminal_commands.md):
    python scripts/kaggle_run.py "<COMMAND>" --task <TASK> [--kernel <SLUG>] [--dataset <OWNER/SLUG>] [--gpu <SHAPE>]
        [--dry-run]
    e.g. python scripts/kaggle_run.py "python src/train.py experiment=grconvnet_clip" --task lang_grasp --kernel base

It writes a one-command notebook for the commit to run (git HEAD, which must be pushed): it clones the repo at that
commit, links the mounted packed data to the layout of configs/paths/kaggle.yaml, installs <TASK>/requirements.txt and
runs <COMMAND> in <TASK>/; then it pushes the notebook as a new version of the private kernel <USER>/<SLUG> with a GPU
and internet on. Use one slug per run to run several at
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
REPO = "https://github.com/hoangducanh1865/PI.git"
CODE = "/kaggle/working/code/fpt_entrance_test"  # the layout of configs/paths/kaggle.yaml


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


def notebook(commit: str, task: str, command: str) -> dict:
    """A notebook whose cells set up the code and data, then run the command; a failing cell stops the run."""
    setup = (
        f"!mkdir -p /kaggle/working/data/datasets/processed /kaggle/working/data/{task}/logs"
        ' && ln -sfn "$(dirname "$(find /kaggle/input -name train.h5 | head -1)")"'
        " /kaggle/working/data/datasets/processed/Grasp-Anything-pp\n"
        f"!rm -rf {CODE} && git clone -q {REPO} {CODE} && git -C {CODE} checkout -q {commit}"
        f" && pip install -q -r {CODE}/{task}/requirements.txt\n"
        'assert _exit_code == 0, "setup failed"\n'
        f"!git -C {CODE} log --oneline -1 && nvidia-smi --query-gpu=name,memory.total --format=csv"
    )
    run = f'!cd {CODE}/{task} && {command}\nassert _exit_code == 0, f"the command failed (exit code {{_exit_code}})"'
    cells = [
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}
        for source in (setup, run)
    ]
    kernel = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return {"cells": cells, "metadata": {"kernelspec": kernel}, "nbformat": 4, "nbformat_minor": 5}


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

    user = kaggle_user()
    folder = tempfile.mkdtemp(prefix=f"kaggle_{args.kernel}_")
    with open(os.path.join(folder, "kaggle.ipynb"), "w") as f:
        json.dump(notebook(commit, args.task, args.command), f, indent=1)
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
    pushed = subprocess.run(["kaggle", "kernels", "push", "-p", folder], check=True, capture_output=True, text=True)
    print(pushed.stdout.strip())
    if "error" in pushed.stdout.lower():  # the CLI exits 0 on refusals such as "Maximum weekly GPU quota ... reached"
        raise SystemExit("Kaggle refused the push (above).")
    print(f"Pushed commit {commit[:7]}. Follow it with:  kaggle kernels status {user}/{args.kernel}")


if __name__ == "__main__":
    main()
