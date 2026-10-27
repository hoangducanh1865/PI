# Physical AI research code

Code written for the Physical AI research programme of QACI (FPT), one folder per task. Each task folder is
self-contained (its own code, configs, tests and `requirements.txt`) and its commands run from inside it.

| Folder | Task |
|---|---|
| [`lang_grasp/`](lang_grasp/) | Entrance test: region-text contrastive grounding for language-driven grasp detection on Grasp-Anything++ (code and paper) |

Shared by every task: `scripts/` (run commands on Kaggle and Colab from a local terminal), `terminal_commands.md`
(every command, per platform) and `pyproject.toml` (lint settings). Data never goes into the repo: each platform has
`<BASE>/code/<repo>/` for the code and `<BASE>/data/` for the data, with the datasets in `data/datasets/` and one folder
per task (`data/<task>/`) for everything a task writes.

## Author

Duc Anh Hoang
