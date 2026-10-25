# Terminal commands

One codebase, three platforms. Every command below runs from the repo root and is the same everywhere: the code
detects the platform (`src/utils/platform.py`) and reads its paths from `configs/paths/<platform>.yaml`.

| Platform | GPU | Role | How you reach it |
|---|---|---|---|
| `local` (Mac, PC) | PC: RTX 3050 | edit code, light tests on fake data, data packing | your terminal |
| `kaggle` | 2x T4, 30 h per week | interactive sessions and background runs up to 12 h | your terminal: `kaggle_shell.py`, `kaggle_run.py` (section 2) |
| `colab` | T4 (or a CPU runtime for downloads) | interactive work, data download to Drive | your terminal: `ssh` through a tunnel opened by `notebooks/colab.ipynb` (section 3) |

Use the GPUs, not the TPUs (Kaggle TPU v5e-8, Colab TPU v5e-1): PyTorch on a TPU needs `torch_xla` and code changes.

## 0. One-time setup on your machine

```bash
pip install -r requirements.txt
pip install kaggle                       # Kaggle CLI, for background runs
cp .env.example .env                     # then paste the Kaggle session URL into .env (gitignored)
```

Kaggle API token (background runs): kaggle.com > Settings > API > Create New Token, save it as
`~/.kaggle/kaggle.json`, then `chmod 600 ~/.kaggle/kaggle.json`. The account needs a verified phone number for GPU and
internet.

## 1. Local

```bash
python src/train.py experiment=baseline --cfg job    # print the composed config of a run
pytest                                               # configs compose on every platform
```

## Data: pack Grasp-Anything++ (once, on Colab)

Raw files (downloaded from Hugging Face) in `MyDrive/it/fpt/datasets/raw/`; output in `.../processed/`. CPU runtime is
enough; about 30-40 min (one sequential pass over the 65 GB image zip). Run it inside tmux; the log goes to `/content`,
never into the code folder:

```bash
tmux new -d -s prep "python scripts/prepare_data.py --raw /content/drive/MyDrive/it/fpt/datasets/raw --out /content/drive/MyDrive/it/fpt/datasets/processed 2>&1 | tee /content/prepare_data.log"
tail -n 5 /content/prepare_data.log
```

Expected end: `done in ... min` and `processed/{train,test_seen,test_unseen}.h5` + `stats.json`.

## 2. Kaggle

### Interactive session, from your terminal

1. kaggle.com > Create > New Notebook; Settings: Accelerator GPU T4 x2, Internet on.
2. Run > Kaggle Jupyter Server > copy the VS Code compatible URL into `.env` as `KAGGLE_SESSION_URL=...`
   (it changes with every session). Keep the browser tab open.
3. In your terminal:

```bash
python scripts/kaggle_shell.py "nvidia-smi"          # one command; exits with its exit code
python scripts/kaggle_shell.py                       # line-by-line shell (cd is kept); Ctrl-D quits
```

Then, inside the shell:

```bash
git clone -q https://github.com/hoangducanh1865/fpt_entrance_test.git /tmp/fpt_entrance_test   # new code: git -C /tmp/fpt_entrance_test pull
cd /tmp/fpt_entrance_test && pip install -q -r requirements.txt
python src/train.py experiment=<EXP>                 # short runs
tmux new -d -s <EXP> "python src/train.py experiment=<EXP> 2>&1 | tee /kaggle/working/<EXP>.log"   # long runs
tail -n 20 /kaggle/working/<EXP>.log
```

The line-by-line shell has no full-screen programs (`vim`, `htop`, `tmux attach`): follow a run with `tail`, stop it
with `tmux kill-session -t <EXP>`. Anything outside `/kaggle/working` disappears when the session ends.
If the script fails with HTTP 403 / 404 or a connection error, the session ended or its URL changed: copy the new URL
into `.env`.

### Background run (batch mode: no browser, up to 12 h)

Push the code to GitHub first (Kaggle clones the pushed commit), then:

```bash
python scripts/kaggle_run.py "python src/train.py experiment=<EXP>" --kernel <SLUG> [--dataset <OWNER>/<DATASET>]
kaggle kernels status <USER>/<SLUG>                  # queued, running, complete or error
kaggle kernels output <USER>/<SLUG> -p logs/kaggle/<SLUG>    # the log and /kaggle/working (logs/, checkpoints)
```

Example: `python scripts/kaggle_run.py "python src/train.py experiment=baseline" --kernel baseline`.
Add `--dry-run` to only write the kernel folder. The kernel also shows on kaggle.com (Your Work > Code).

## 3. Colab

The notebook `notebooks/colab.ipynb` is local only (gitignored, it holds `HF_TOKEN`). Open it in VS Code with the Google
Colab extension (Select Kernel > Colab > New Colab Server > GPU T4, or CPU for downloads).

Rules: the Colab FAQ forbids SSH on free runtimes (allowed with a paid plan and compute units left). On the free account
Colab may end the runtime without warning: keep important results on Drive.

### One-time setup on your machine

```bash
brew install cloudflared
```

Add to `~/.ssh/config` (the tunnel host name changes every session, so no host key is kept):

```text
Host *.trycloudflare.com
    User root
    ProxyCommand cloudflared access ssh --hostname %h
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR
    ServerAliveInterval 30
```

### Every session

1. In the notebook run cell 1 (mount Drive), cell 2 (clone or `git pull` the code, install) and cell 4 (SSH tunnel).
   Keep cell 4 running: stopping it closes the tunnel. Cell 4 prints `SSH ready. On your machine:  ssh <HOST>`.
2. **Wait about 30 seconds** (the new host name needs time to appear in DNS), then in your terminal:

```bash
ssh <HOST>.trycloudflare.com                         # lands in the code folder on Drive
tmux new -s main                                     # anything long runs inside tmux: detach Ctrl-b d
tmux attach -t main                                  # back to it after an SSH drop
nvidia-smi
python src/train.py experiment=<EXP>
```

Code: `/content/drive/MyDrive/it/fpt/fpt_entrance_test`, kept between sessions. Update it only by: edit on the Mac,
`git push`, then `git pull` there (cell 2 does it). Never edit code on Colab. Data and logs: Drive.
Changing the runtime type (GPU to CPU or back) gives a new machine: run cells 1, 2, 4 again and use the new host.

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `dial tcp: lookup <HOST>: no such host` | you connected before the host was in DNS, and macOS cached "no such host" | `sudo dscacheutil -flushcache; sudo killall -HUP mDNSResponder`, then ssh again (or wait a few minutes) |
| `websocket: bad handshake` | the tunnel is down (cell 4 stopped, runtime restarted) | run cell 4 again, use the new host |
| cell 4 prints `sshd is not listening on 127.0.0.1:2022` and a log | the SSH server did not start | paste the log to Claude |
| Colab Secrets error about `HF_TOKEN` | Secrets only work in the Colab web UI, not in VS Code | define `HF_TOKEN = "..."` in the local notebook |
