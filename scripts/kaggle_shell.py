"""
Run shell commands on a running Kaggle notebook session from your terminal.

Kaggle's notebook editor gives outside clients a Jupyter Server URL (Run > Kaggle Jupyter Server > VS Code compatible
URL). This script starts its own Python kernel there, runs each command in bash through it and streams the output; the
notebook open in the browser is not touched. The URL is read from KAGGLE_SESSION_URL (environment, or the repo's .env,
which is gitignored). It holds an access token and changes with every session: never print it or commit it.

Usage (on the Mac or the PC, from any folder):
    python scripts/kaggle_shell.py "<COMMAND>"     # run one command, exit with its exit code
    python scripts/kaggle_shell.py                 # line-by-line shell (cd is kept, no vim / htop); Ctrl-D quits
    e.g. python scripts/kaggle_shell.py "nvidia-smi"

Ctrl-C stops the running command. Quitting deletes the kernel and whatever still runs in it: start a long run detached
with tmux (terminal_commands.md, section Kaggle), or use scripts/kaggle_run.py (batch mode, no browser needed).

Author: Duc Anh Hoang
"""

import ast
import json
import os
import readline  # noqa: F401  (line editing and history for input())
import sys
import urllib.request
import uuid

import websocket  # websocket-client

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
CWD_FILE = "/tmp/.kaggle_shell_cwd"
# Runs inside the kernel: bash runs the command in its own process group (Ctrl-C stops all of it) and saves its last
# working directory, so a cd is kept for the next command.
RUNNER = """\
import os, signal, subprocess, sys
_p = subprocess.Popen({script!r}, shell=True, executable="/bin/bash", start_new_session=True,
                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
try:
    while _chunk := os.read(_p.stdout.fileno(), 4096):
        sys.stdout.write(_chunk.decode(errors="replace"))
        sys.stdout.flush()
except KeyboardInterrupt:
    os.killpg(_p.pid, signal.SIGTERM)
_rc = _p.wait()
if os.path.exists({cwd_file!r}):
    os.chdir(open({cwd_file!r}).read().strip())
"""


def session_url() -> str:
    url = os.environ.get("KAGGLE_SESSION_URL", "")
    env_file = os.path.join(ROOT, ".env")
    if not url and os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                key, _, value = line.strip().partition("=")
                if key == "KAGGLE_SESSION_URL":
                    url = value.strip("\"'")
    if not url:
        raise SystemExit("KAGGLE_SESSION_URL is not set (environment or .env)")
    return url.rstrip("/")


def api(url: str, method: str, path: str, body: dict | None = None) -> bytes:
    data = json.dumps(body).encode() if body is not None else b""
    request = urllib.request.Request(
        f"{url}/{path}", data=data, method=method, headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(request).read()


class Kernel:
    """A fresh Python kernel on the session, driven over the Jupyter messaging protocol."""

    def __init__(self, url: str):
        self.url = url
        self.id = json.loads(api(url, "POST", "api/kernels", {"name": "python3"}))["id"]
        self.ws = websocket.create_connection(f"wss://{url.split('://', 1)[1]}/api/kernels/{self.id}/channels")
        self.session = uuid.uuid4().hex
        self.cwd = "~"

    def run(self, command: str) -> int:
        """Run a bash command, stream its output, return its exit code."""
        script = f"{command}\n_rc=$?; pwd > {CWD_FILE}; exit $_rc"
        msg_id = uuid.uuid4().hex
        header = {"msg_id": msg_id, "session": self.session, "msg_type": "execute_request", "version": "5.3"}
        content = {
            "code": RUNNER.format(script=script, cwd_file=CWD_FILE),
            "silent": False,
            "store_history": False,
            "user_expressions": {"rc": "_rc", "cwd": "os.getcwd()"},
            "allow_stdin": False,
            "stop_on_error": True,
        }
        request = {"header": header, "parent_header": {}, "metadata": {}, "content": content, "channel": "shell"}
        self.ws.send(json.dumps(request))
        while True:
            try:
                message = json.loads(self.ws.recv())
            except KeyboardInterrupt:
                api(self.url, "POST", f"api/kernels/{self.id}/interrupt")
                continue
            if message.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            content = message["content"]
            if message["msg_type"] == "stream":
                sys.stdout.write(content["text"])
                sys.stdout.flush()
            elif message["msg_type"] == "error":
                print(f"{content['ename']}: {content['evalue']}", file=sys.stderr)
            elif message["msg_type"] == "execute_reply":
                results = content.get("user_expressions", {})
                if "cwd" in results:
                    self.cwd = ast.literal_eval(results["cwd"]["data"]["text/plain"])
                return int(results["rc"]["data"]["text/plain"]) if "rc" in results else 1

    def close(self) -> None:
        self.ws.close()
        api(self.url, "DELETE", f"api/kernels/{self.id}")


def main() -> None:
    kernel = Kernel(session_url())
    try:
        if len(sys.argv) > 1:
            sys.exit(kernel.run(" ".join(sys.argv[1:])))
        print("Connected to the Kaggle session (Ctrl-D quits).")
        kernel.run("cd ~")
        while True:
            try:
                line = input(f"kaggle:{kernel.cwd}$ ").strip()
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                continue
            if line in ("exit", "logout"):
                break
            if line:
                kernel.run(line)
    finally:
        kernel.close()


if __name__ == "__main__":
    main()
