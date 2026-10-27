# Terminal commands

One repo, one folder per task (`lang_grasp/`, ...), three platforms. A task's commands run from its folder
(`cd lang_grasp`) and are the same everywhere: the code detects the platform (`src/utils/platform.py` of the task) and
reads its paths from the task's `configs/paths/<platform>.yaml`. The platform tools shared by every task
(`scripts/kaggle_shell.py`, `scripts/kaggle_run.py`, `scripts/colab_ssh.sh`) run from the repo root.

| Platform | GPU | Role | How you reach it |
|---|---|---|---|
| `local` (Mac, PC) | PC: RTX 3050 | edit code, light tests on fake data, data packing | your terminal |
| `kaggle` | 2x T4, 30 h per week | interactive sessions and background runs up to 12 h | your terminal: `kaggle_shell.py`, `kaggle_run.py` (section 2) |
| `colab` | T4 (or a CPU runtime for downloads) | interactive work, data download to Drive | your terminal: `ssh` through a tunnel opened by `notebooks/colab.ipynb` (section 3) |

Use the GPUs, not the TPUs (Kaggle TPU v5e-8, Colab TPU v5e-1): PyTorch on a TPU needs `torch_xla` and code changes.

## 0. One-time setup on your machine

```bash
pip install -r lang_grasp/requirements.txt   # a task's dependencies (one venv per task is fine)
pip install kaggle                       # Kaggle CLI, for background runs
cp .env.example .env                     # then paste the Kaggle session URL into .env (gitignored)
```

Kaggle API token (background runs): kaggle.com > Settings > API > Create New Token, save it as
`~/.kaggle/kaggle.json`, then `chmod 600 ~/.kaggle/kaggle.json`. The account needs a verified phone number for GPU and
internet.

## 1. Local

```bash
cd lang_grasp
python src/train.py experiment=grconvnet_clip --cfg job    # print the composed config of a run
pytest                                               # configs compose on every platform
```

## lang_grasp data: pack Grasp-Anything++ (once, on Colab, from `lang_grasp/`)

Raw files (downloaded from Hugging Face) in `<BASE>/data/datasets/raw/`; output in `.../processed/Grasp-Anything-pp/`. CPU runtime is
enough; about 40 min (one sequential pass over the 65 GB image zip). Run it inside tmux; the log goes to Drive
(`logs/scripts/`), never into the code folder:

```bash
mkdir -p /content/drive/MyDrive/it/fpt/data/lang_grasp/logs/scripts
tmux new -d -s prep "python scripts/prepare_data.py --raw /content/drive/MyDrive/it/fpt/data/datasets/raw --out /content/drive/MyDrive/it/fpt/data/datasets/processed/Grasp-Anything-pp 2>&1 | tee /content/drive/MyDrive/it/fpt/data/lang_grasp/logs/scripts/prepare_data.log"
tail -n 5 /content/drive/MyDrive/it/fpt/data/lang_grasp/logs/scripts/prepare_data.log
python scripts/cache_text_emb.py --processed /content/drive/MyDrive/it/fpt/data/datasets/processed/Grasp-Anything-pp   # then: CLIP text features (~2 min)
```

Expected end: `done in ... min` and `processed/{train,test_seen,test_unseen}.h5` + `stats.json`.

## lang_grasp: train and evaluate (any platform, from `lang_grasp/`; on Colab inside tmux, over SSH)

```bash
python -u src/train.py experiment=<EXP> +trainer.fast_dev_run=true      # smoke test: 1 train + 1 val batch
tmux new -d -s train "python -u src/train.py experiment=<EXP> 2>&1 | tee <LOG_DIR>/<EXP>/console_$(date +%F_%H-%M-%S).log"
tail -n 3 <LOG_DIR>/<EXP>/<RUN>/metrics.csv                             # epoch, step, losses, val/success
python -u src/train.py experiment=<EXP> ckpt_path=<LOG_DIR>/<EXP>/<RUN>/checkpoints/last.ckpt   # resume after a cut
python -u src/eval.py ckpt_path=<LOG_DIR>/<EXP>/<RUN>/checkpoints/<BEST>.ckpt   # Seen, Unseen, H -> <LOG_DIR>/eval/<RUN>/
```

`<EXP>`: `grconvnet` (image only) or `grconvnet_clip` (the paper's baseline). `<LOG_DIR>` on Colab:
`/content/drive/MyDrive/it/fpt/data/lang_grasp/logs` (Kaggle: `/kaggle/working/data/lang_grasp/logs`). One epoch on a Colab T4 takes about 5 min, a full run (50 epochs) about 4 h.
`-u` keeps Python from buffering the output, so the log shows the progress bar.

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
mkdir -p /kaggle/working/code /kaggle/working/data/datasets/processed /kaggle/working/data/lang_grasp/logs
git clone -q https://github.com/hoangducanh1865/fpt_entrance_test.git /kaggle/working/code/fpt_entrance_test   # new code: git pull there
ln -sfn /kaggle/input/grasp-anything-pp /kaggle/working/data/datasets/processed/Grasp-Anything-pp   # the Kaggle Dataset
cd /kaggle/working/code/fpt_entrance_test/lang_grasp && pip install -q -r requirements.txt
python src/train.py experiment=<EXP>                 # short runs
tmux new -d -s <EXP> "python src/train.py experiment=<EXP> 2>&1 | tee /kaggle/working/data/lang_grasp/logs/<EXP>.log"   # long runs
tail -n 20 /kaggle/working/data/lang_grasp/logs/<EXP>.log
```

The line-by-line shell has no full-screen programs (`vim`, `htop`, `tmux attach`): follow a run with `tail`, stop it
with `tmux kill-session -t <EXP>`. Anything outside `/kaggle/working` disappears when the session ends.
If the script fails with HTTP 403 / 404 or a connection error, the session ended or its URL changed: copy the new URL
into `.env`.

### Background run (batch mode: no browser, up to 12 h)

Push the code to GitHub first (Kaggle clones the pushed commit), then, from the repo root (`--task` is the folder
whose `requirements.txt` is installed and where the command runs):

```bash
python scripts/kaggle_run.py "python src/train.py experiment=<EXP>" --task lang_grasp --kernel <SLUG> [--dataset <OWNER>/<DATASET>]
kaggle kernels status <USER>/<SLUG>                  # queued, running, complete or error
kaggle kernels output <USER>/<SLUG> -p logs/kaggle/<SLUG>    # the log and /kaggle/working (logs/, checkpoints)
```

Example: `python scripts/kaggle_run.py "python src/train.py experiment=grconvnet_clip" --task lang_grasp --kernel baseline`.
Kernels pushed before the task folders existed (commit `a6c62fb` and older) write under `data/fpt_entrance_test/logs/`
in their output, later ones under `data/<task>/logs/`.
Add `--dry-run` to only write the kernel folder. The kernel also shows on kaggle.com (Your Work > Code).

## 3. Colab

Code and data are separate; on Colab everything stays on Drive, under `<BASE>` = `/content/drive/MyDrive/it/fpt/` (Kaggle: the same
layout under `/kaggle/working/`; local: the folder two levels above the repo, `<BASE>/code/fpt_entrance_test`):

| Folder | Content |
|---|---|
| `code/fpt_entrance_test/` | the code, one folder per task: updated with `git pull` only, nothing is ever written into it |
| `data/datasets/raw/` | the Hugging Face downloads (`Grasp-Anything/`, `Grasp-Anything-pp/`) |
| `data/datasets/processed/Grasp-Anything-pp/` | packed splits `*.h5`, text caches `*_text_*.npz`, `stats.json`; `test/` = 20 samples drawn for inspection |
| `data/lang_grasp/logs/<EXP>/<DATE>_<TIME>/` | one training run: `.hydra/config.yaml`, `metrics.csv`, `checkpoints/` |
| `data/lang_grasp/logs/eval/<DATE>_<TIME>/` | one evaluation: `results.json`, `predictions_<split>.npz` |
| `data/lang_grasp/logs/scripts/` | logs of the data scripts |
| `data/lang_grasp/analysis/` | figures and tables of the checks and the error analysis |

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
ssh <HOST>.trycloudflare.com                         # lands in the repo root on Drive
tmux new -s main                                     # anything long runs inside tmux: detach Ctrl-b d
tmux attach -t main                                  # back to it after an SSH drop
nvidia-smi
cd lang_grasp && python src/train.py experiment=<EXP>
```

Code: `/content/drive/MyDrive/it/fpt/code/fpt_entrance_test`, kept between sessions. Update it only by: edit on the Mac,
`git push`, then `git pull` there (cell 2 does it). Never edit code on Colab. Data and logs: Drive.
Changing the runtime type (GPU to CPU or back) gives a new machine: run cells 1, 2, 4 again and use the new host.

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `dial tcp: lookup <HOST>: no such host` | you connected before the host was in DNS, and macOS cached "no such host" | `sudo dscacheutil -flushcache; sudo killall -HUP mDNSResponder`, then ssh again (or wait a few minutes) |
| `websocket: bad handshake` | the tunnel is down (cell 4 stopped, runtime restarted) | run cell 4 again, use the new host |
| cell 4 prints `sshd is not listening on 127.0.0.1:2022` and a log | the SSH server did not start | paste the log to Claude |
| Colab Secrets error about `HF_TOKEN` | Secrets only work in the Colab web UI, not in VS Code | define `HF_TOKEN = "..."` in the local notebook |
