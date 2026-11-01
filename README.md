# Physical AI research code

Code for the Physical AI programme of QACI (FPT Frontier Innovators 2026). Each project is a self-contained folder with
its own README, requirements and scripts; commands run from inside it (`cd <folder>`).

| Folder | Project |
|---|---|
| [`lang_grasp/`](lang_grasp/) | **RTC-Grasp**: region-text contrastive grounding for language-driven grasp detection on Grasp-Anything++ (code and paper) |
| [`video_eval/`](video_eval/) | Evaluating generated robot videos: IRASim and Cosmos-Predict scored against real-video references (in progress) |
| [`sim2real_worldmodel/`](sim2real_worldmodel/) | Does synthetic robot data (simulator or world model) move like human demonstrations? (early stage) |

Shared by every project:

| Path | Content |
|---|---|
| `scripts/kaggle_run.py` | run one command on a Kaggle GPU as a batch kernel, from a local terminal |
| `scripts/kaggle_shell.py` | run commands in an open Kaggle notebook session through its Jupyter server URL |
| `scripts/colab_ssh.sh`, `notebooks/colab.ipynb` | SSH into a Colab runtime (paid plans) |
| `scripts/data_sync.py` | move the working data between machines through a private Hugging Face dataset |
| `terminal_commands.md` | every command, per platform and per project |
| `pyproject.toml` | lint settings (`ruff`) |

Data never goes into the repository: each platform has `<BASE>/code/` for the code and `<BASE>/data/` for datasets and
outputs, and each project reads its paths from `configs/paths/<platform>.yaml`. Secrets go in `.env` (see
`.env.example`), which is ignored by git.

## Author

Duc Anh Hoang
