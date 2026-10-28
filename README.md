# Physical AI research code

Code written for the Physical AI programme of QACI (FPT Frontier Innovators 2026): two assignments, each in its own
folder, and one research direction that grows out of the second.

| Folder | Work | Status |
|---|---|---|
| [`lang_grasp/`](lang_grasp/) | Assignment 1 (Prof. Anh Nguyen): language-driven grasp detection on Grasp-Anything++; region-text contrastive grounding over a reproduced GR-ConvNet + CLIP baseline | done: code and 2-page paper |
| [`video_eval/`](video_eval/) | Assignment 2 (Dr. Ngan Le): evaluating generated videos for robot trajectories; reproduces IRASim and the baselines released with it on Bridge, and scores them with real-video references | in progress |
| [`sim2real_worldmodel/`](sim2real_worldmodel/) | Research direction: does synthetic robot data (simulator or world model) move like human demonstrations? | early: simulator benchmark, MimicGen trajectories |

Each folder has its own README, `requirements.txt` and scripts, and its commands run from inside it (`cd <folder>`).
`lang_grasp/` follows [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) (configs, `src/`,
tests); the other two are script-based so far.

Shared by every folder: `scripts/` (run commands on Kaggle and Colab from a local terminal; `data_sync.py` moves the
working data between machines through a private Hugging Face repo), `terminal_commands.md`
(every command, per platform and per folder) and `pyproject.toml` (lint settings). Data never goes into the repo: each
platform has `<BASE>/code/<repo>/` for the code and `<BASE>/data/` for the data, with the datasets in `data/datasets/`
and one folder per task (`data/<task>/`) for everything a task writes.

## Author

Duc Anh Hoang
