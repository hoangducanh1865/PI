# Language-driven Grasp Detection

Given an RGB image and a text prompt such as "grasp the blue bottle", predict a 2D grasp rectangle
(x, y, w, h, θ): its center, width, height and angle in the image plane. Dataset:
[Grasp-Anything++](https://airvlab.github.io/grasp-anything/) (Vuong et al., CVPR 2024).

**Status:** work in progress. The repository skeleton and the cloud workflow are in place; the method and the results
will follow.

## Layout

The code follows [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) and runs unchanged on a
local machine, Google Colab and Kaggle.

| Path | Content |
|---|---|
| `configs/` | [Hydra](https://hydra.cc) configs: `train.yaml`, `eval.yaml`; `experiment/` one file per run (only what differs); `paths/` one file per platform |
| `src/` | entry points `train.py`, `eval.py`; `utils/` |
| `scripts/` | `kaggle_shell.py` (shell on a Kaggle session from the terminal), `kaggle_run.py` (background run on Kaggle), `colab_ssh.sh` (SSH into a Colab runtime) |
| `.env.example` | local settings to copy into `.env` (gitignored) |
| `tests/` | config and platform checks (`pytest`) |
| `paper/` | LaTeX source of the paper (CVPR 2026 template) |
| `terminal_commands.md` | every command, per platform |

## Quick start

```bash
pip install -r requirements.txt
python src/train.py experiment=baseline --cfg job    # print the composed config of a run
pytest
```

The same commands run on Kaggle (from your terminal, in an interactive session or as a background run) and on Colab:
see [terminal_commands.md](terminal_commands.md).

## Author

Duc Anh Hoang
