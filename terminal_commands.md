# Terminal commands

One codebase, three platforms. Every command below runs from the repo root and is the same everywhere: the code
detects the platform (`src/utils/platform.py`) and reads its paths from `configs/paths/<platform>.yaml`.

| Platform | GPU | Role | How you reach it |
|---|---|---|---|
| `local` (Mac, PC) | PC: RTX 3050 | edit code, light tests on fake data, data packing | your terminal |
| `kaggle` | 2x T4, 30 h per week | interactive sessions and background runs up to 12 h | your terminal: `kaggle_shell.py`, `kaggle_run.py` (section 2) |
| `colab` | T4 | interactive GPU work | `notebooks/colab.ipynb` in VS Code or the browser; SSH with a paid plan (section 3) |

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

Open `notebooks/colab.ipynb` in VS Code with the Google Colab extension (Select Kernel > Colab > New Colab Server >
GPU T4), or in the browser (File > Open notebook > GitHub). Run cells 1-2 once per session, then put your command in
cell 3, for example `COMMAND = "python src/train.py experiment=<EXP>"`.

### SSH (only with a paid plan and compute units left)

The Colab FAQ forbids SSH on free runtimes. With a paid plan, once:

```bash
brew install cloudflared
```

and add to `~/.ssh/config` (the tunnel host name changes every session, so no host key is kept):

```text
Host *.trycloudflare.com
    User root
    ProxyCommand cloudflared access ssh --hostname %h
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR
    ServerAliveInterval 30
```

Run cell 4 of the notebook (keep it running) and connect with the host it prints:

```bash
ssh <HOST>.trycloudflare.com                         # lands in /content/fpt_entrance_test
tmux new -s main                                     # detach: Ctrl-b d, back: tmux attach -t main
python src/train.py experiment=<EXP>
```
