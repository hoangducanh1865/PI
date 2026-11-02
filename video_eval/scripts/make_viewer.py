"""
Offline results viewer for the submission: one HTML page that plays the real video of a chosen clip and the videos of
every model we ran on it (also bad ones, and Frame-Ada from corrupted actions) side by side, in sync, with that clip's
scores (appearance, 2-D track, EVID error and executability).

Usage (from video_eval/; after the eval scripts and scripts/executability.py have written their CSV files):
    python scripts/make_viewer.py --gen <GEN_DIR> [--extra-gen <DIR> ...] --out <OUT_DIR> [--paper paper/main.pdf]
        [--limit N]

<GEN_DIR> gives the clips (every clip with a real video) and the models it holds; a model missing there (VDM, run on 24
clips only) is taken with its scores from the first --extra-gen folder that holds it. A model without a clip shows an
empty frame. Clips with every model come first; --limit keeps the first N clips (for the zip). Copies the videos to
<OUT_DIR>/videos/ and writes <OUT_DIR>/index.html with the scores embedded, so the folder works without a server or a
network; it is also the project page (GitHub Pages, branch gh-pages).

Author: Duc Anh Hoang
"""

import argparse
import csv
import json
import os
import shutil

import numpy as np

MODELS = [  # folder, label, group
    ("real", "Real video", "References"),
    ("still", "Still first frame", "References"),
    ("frame_ada/seed3407", "IRASim-Frame-Ada", "Models, from the recorded actions"),
    ("video_ada/seed3407", "IRASim-Video-Ada", "Models, from the recorded actions"),
    ("lvdm/seed3407", "LVDM", "Models, from the recorded actions"),
    ("vdm/seed3407", "VDM (24 clips)", "Models, from the recorded actions"),
    ("cosmos_action/seed0", "Cosmos-Predict2.5", "Models, from the recorded actions"),
    ("cosmos_base/seed0", "Cosmos-Predict2.5 without actions (24 clips)", "Models, from the recorded actions"),
    ("frame_ada_noise_s/seed3407", "Noise 0.5 cm", "IRASim-Frame-Ada from corrupted actions"),
    ("frame_ada_noise_l/seed3407", "Noise 1.5 cm", "IRASim-Frame-Ada from corrupted actions"),
    ("frame_ada_freeze/seed3407", "All zero", "IRASim-Frame-Ada from corrupted actions"),
    ("frame_ada_mirror/seed3407", "Mirrored", "IRASim-Frame-Ada from corrupted actions"),
]
SCORES = [  # file, column, label, digits
    ("pixel", "psnr", "PSNR", 2),
    ("pixel", "ssim", "SSIM", 3),
    ("pixel", "lpips", "LPIPS", 3),
    ("motion_cotracker", "ee_ndtw", "2-D NDTW", 3),
    ("idm_cosmos3", "pos_err_cm", "EVID error (cm)", 2),
    ("executability", "executable", "Executable", 0),
]

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>EVID</title>
<style>
body { margin: 0; padding: 24px 16px; background: #fff; color: #111; font: 15px/1.5 -apple-system, Helvetica, Arial,
  sans-serif; }
main { max-width: 1680px; margin: 0 auto; }
h1 { font-size: 26px; margin: 0 0 6px; text-align: center; }
h2 { font-size: 19px; margin: 28px 0 6px; }
.author { text-align: center; margin: 0 0 12px; font-size: 17px; }
.links { text-align: center; margin: 0 0 18px; }
.links a { display: inline-block; margin: 0 6px; padding: 6px 16px; border-radius: 18px; background: #111; color: #fff;
  text-decoration: none; }
.abstract { max-width: 900px; margin: 0 auto 8px; }
p.lead { margin: 0 0 16px; color: #444; }
.controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 12px; }
select, button { font: inherit; padding: 4px 10px; border: 1px solid #999; border-radius: 4px; background: #fff; }
input[type=range] { width: 280px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; }
h3 { font-size: 16px; margin: 16px 0 6px; }
.missing { aspect-ratio: 320 / 256; border: 1px dashed #bbb; color: #777; display: flex; align-items: center;
  justify-content: center; text-align: center; font-size: 13px; padding: 8px; }
figure { margin: 0; }
video { width: 100%; display: block; border: 1px solid #ddd; }
figcaption { font-size: 14px; margin-top: 4px; }
table { border-collapse: collapse; margin-top: 18px; font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid #ddd; padding: 4px 12px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { background: #f2f2f2; }
.note { color: #555; font-size: 13px; margin-top: 10px; max-width: 900px; }
.scroll { overflow-x: auto; }
</style></head><body><main>
<h1>EVID: Reference-free Evaluation of Action Fidelity in Videos from Robot World Models via Inverse Dynamics</h1>
<p class="author">Duc Anh Hoang</p>
<p class="links"><a href="report.pdf">Paper</a><a href="https://github.com/hoangducanh1865/PI/tree/main/video_eval">Code</a>
</p>
<p class="abstract">Action-conditioned video world models predict the video a robot would record when it executes
given actions, and they are evaluated mostly by how closely their frames match the recorded video. Such appearance
metrics need the real video, which does not exist for new actions, and do not check whether the robot performs the
commanded motion. Reproducing IRASim and its baselines and adding Cosmos-Predict2.5 on 105 Bridge and 57 Language-Table
clips, we find that SSIM rates a still video as high as a correct generation and above every generation from corrupted
actions. We propose EVID, which closes the loop from actions to video and back: an off-the-shelf inverse dynamics model
reads the end-effector trajectory from the generated video, EVID measures its distance to the commanded trajectory, and
inverse kinematics checks that the real arm could execute it. Without any real video, EVID separates real videos from a
still one by a factor of six, grows monotonically with the corruption, and ranks the models as the reference-based
metrics do. It shows that the action input, not the quality of the frames, decides whether the robot follows the
commands, while the most frequent remaining errors are objects that change or vanish.</p>
<h2>Results viewer</h2>
<p class="lead">The 105 Bridge clips, every video generated from the same first frame and the same actions; all models
we ran are shown, good or bad. VDM and Cosmos-Predict2.5 without actions were run on 24 of the clips, listed first.
Arrows give the better direction. EVID error: distance between the end-effector path read back from the video by the
Cosmos3-Nano inverse dynamics model and the commanded path; it needs no real video.</p>
<div class="controls">
  <label>Clip <select id="clip"></select></label>
  <button id="play">Play</button>
  <label>Frame <input id="frame" type="range" min="0" max="15" step="1" value="0"> <span id="frameText">0</span> / 15
  </label>
</div>
<p id="instruction"></p>
<div id="videos"></div>
<div class="scroll"><table id="scores"></table></div>
<p class="note">Executable: every pose read back has a WidowX 250 inverse-kinematics solution within the joint limits
and every step stays under the velocity limit (1 = yes). The real video row is the reader's own error (its floor); the
still first frame is the floor of motion.</p>
</main>
<script>
const DATA = __DATA__;
const FPS = 5;
const clipSel = document.getElementById("clip"), slider = document.getElementById("frame");
const frameText = document.getElementById("frameText"), playBtn = document.getElementById("play");
let videos = [], playing = false;
DATA.clips.forEach(c => clipSel.add(new Option(
  `${c.id}: ${c.instruction}${c.videos.length === DATA.models.length ? " (all models)" : ""}`, c.id)));
function arrow(label) { return /LPIPS|EVID/.test(label) ? " \\u2193" : " \\u2191"; }
function show(id) {
  const clip = DATA.clips.find(c => c.id === id);
  document.getElementById("instruction").textContent = `Instruction: "${clip.instruction}"`;
  const box = document.getElementById("videos");
  box.innerHTML = "";
  videos = [];
  let grid = null, group = null;
  DATA.models.forEach(m => {
    if (m.group !== group) {
      group = m.group;
      const h = document.createElement("h3"); h.textContent = group;
      grid = document.createElement("div"); grid.className = "grid"; box.append(h, grid);
    }
    const fig = document.createElement("figure");
    let v;
    if (clip.videos.includes(m.dir)) {
      v = document.createElement("video");
      v.src = `videos/${m.dir}/${id}.mp4`; v.muted = true; v.playsInline = true; v.preload = "auto";
      videos.push(v);
    } else {
      v = document.createElement("div"); v.className = "missing"; v.textContent = "not generated for this clip";
    }
    const cap = document.createElement("figcaption"); cap.textContent = m.label;
    fig.append(v, cap); grid.append(fig);
  });
  videos[0].addEventListener("timeupdate", () => {
    const f = Math.min(15, Math.floor(videos[0].currentTime * FPS));
    slider.value = f; frameText.textContent = f;
  });
  videos[0].addEventListener("ended", () => { playing = false; playBtn.textContent = "Play"; });
  const t = document.getElementById("scores");
  const head = DATA.scores.map(s => `<th>${s}${s === "Executable" ? "" : arrow(s)}</th>`).join("");
  const cell = (m, s) => {
    const v = clip.values[m.dir] && clip.values[m.dir][s];
    return `<td>${v === undefined || v === null ? "-" : v}</td>`;
  };
  const rows = DATA.models.map(m => `<tr><td>${m.label}</td>` + DATA.scores.map(s => cell(m, s)).join("") + "</tr>");
  t.innerHTML = "<tr><th>Model</th>" + head + "</tr>" + rows.join("");
  playing = false; playBtn.textContent = "Play"; slider.value = 0; frameText.textContent = 0;
}
playBtn.onclick = () => {
  playing = !playing; playBtn.textContent = playing ? "Pause" : "Play";
  videos.forEach(v => { if (playing) { if (v.ended) v.currentTime = 0; v.play(); } else v.pause(); });
};
slider.oninput = () => {
  playing = false; playBtn.textContent = "Play"; frameText.textContent = slider.value;
  videos.forEach(v => { v.pause(); v.currentTime = (Number(slider.value) + 0.5) / FPS; });
};
clipSel.onchange = () => show(clipSel.value);
show(DATA.clips[0].id);
</script></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gen", required=True)
    parser.add_argument("--extra-gen", nargs="*", default=[], help="folders for models missing in --gen")
    parser.add_argument("--out", required=True)
    parser.add_argument("--paper", help="PDF of the report, copied as <OUT_DIR>/report.pdf")
    parser.add_argument("--limit", type=int, help="keep only the first N clips")
    args = parser.parse_args()

    # A model of --gen takes its scores from --gen only, one of an extra folder from the extra folders, so the sets are
    # never mixed.
    source = {}
    for m, _, _ in MODELS:
        found = [g for g in [args.gen, *args.extra_gen] if os.path.isdir(os.path.join(g, m))]
        if found:
            source[m] = found[0]
    has = {m: {c[:-4] for c in os.listdir(os.path.join(g, m)) if c.endswith(".mp4")} for m, g in source.items()}
    clips = sorted(has["real"], key=lambda c: (-sum(c in v for v in has.values()), c))[: args.limit]
    pool = {m: [args.gen] if g == args.gen else args.extra_gen for m, g in source.items()}
    values = {}
    for name, column, label, digits in SCORES:
        for g in [args.gen, *args.extra_gen]:
            path = os.path.join(g, f"metrics_{name}.csv")
            for row in csv.DictReader(open(path)) if os.path.exists(path) else []:
                if g in pool.get(row["model"], []) and row["clip"] in clips and row[column] not in ("", "nan"):
                    value = round(float(row[column]), digits) if digits else int(float(row[column]))
                    values.setdefault(row["clip"], {}).setdefault(row["model"], {}).setdefault(label, value)
    data = {"models": [{"dir": m, "label": label, "group": group} for m, label, group in MODELS if m in source],
            "scores": [s[2] for s in SCORES], "clips": []}  # fmt: skip
    shutil.rmtree(os.path.join(args.out, "videos"), ignore_errors=True)
    for clip in clips:
        instruction = str(np.load(os.path.join(args.gen, "states", f"{clip}.npz"))["instruction"])
        shown = [m for m in source if clip in has[m]]
        data["clips"].append({"id": clip, "instruction": instruction, "values": values.get(clip, {}), "videos": shown})
        for m in shown:
            os.makedirs(os.path.join(args.out, "videos", m), exist_ok=True)
            shutil.copy(os.path.join(source[m], m, f"{clip}.mp4"), os.path.join(args.out, "videos", m, f"{clip}.mp4"))
    if args.paper:
        shutil.copy(args.paper, os.path.join(args.out, "report.pdf"))
    with open(os.path.join(args.out, "index.html"), "w") as f:
        f.write(PAGE.replace("__DATA__", json.dumps(data)))
    print(f"wrote {args.out}/index.html: {len(clips)} clips, {len(source)} models")


if __name__ == "__main__":
    main()
