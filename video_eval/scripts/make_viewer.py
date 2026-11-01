"""
Offline results viewer for the submission: one HTML page that plays the real and the generated videos of a chosen clip
side by side, in sync, with that clip's scores (appearance, 2-D track, EVID cycle error and executability).

Usage (from video_eval/; after the eval scripts and scripts/executability.py have written their CSV files):
    python scripts/make_viewer.py --gen <GEN_DIR> --out <OUT_DIR> [--paper paper/main.pdf]

Uses the clips that every shown model generated (the 24 core clips). Copies the videos to <OUT_DIR>/videos/ and writes
<OUT_DIR>/index.html with the scores embedded, so the folder works without a server or a network; it is also the
project page (GitHub Pages, branch gh-pages).

Author: Duc Anh Hoang
"""

import argparse
import csv
import json
import os
import shutil

import numpy as np

MODELS = [  # folder, label
    ("real", "Real video"),
    ("frame_ada/seed3407", "IRASim-Frame-Ada"),
    ("cosmos_action/seed0", "Cosmos-Predict2.5, actions"),
    ("cosmos_base/seed0", "Cosmos-Predict2.5, no actions"),
    ("still", "Still first frame"),
]
SCORES = [  # file, column, label, digits
    ("pixel", "psnr", "PSNR", 2),
    ("pixel", "ssim", "SSIM", 3),
    ("pixel", "lpips", "LPIPS", 3),
    ("motion_cotracker", "ee_ndtw", "2-D NDTW", 3),
    ("idm_cosmos3", "pos_err_cm", "EVID cycle error (cm)", 2),
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
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px; }
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
<p class="abstract">Video world models are compared mostly by how their frames look. EVID reads the end-effector
trajectory back from a generated video with an inverse dynamics model and measures how far it deviates from the
commanded actions, and whether a real WidowX 250 arm could execute it, without any real video.</p>
<h2>Results viewer</h2>
<p class="lead">Bridge clips generated from the same first frame and actions. Arrows give the better direction.
EVID cycle error: distance between the end-effector path read back from the video by the Cosmos3-Nano inverse dynamics
model and the commanded path; it needs no real video.</p>
<div class="controls">
  <label>Clip <select id="clip"></select></label>
  <button id="play">Play</button>
  <label>Frame <input id="frame" type="range" min="0" max="15" step="1" value="0"> <span id="frameText">0</span> / 15
  </label>
</div>
<p id="instruction"></p>
<div class="grid" id="videos"></div>
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
DATA.clips.forEach(c => clipSel.add(new Option(`${c.id}: ${c.instruction}`, c.id)));
function arrow(label) { return /LPIPS|cycle/.test(label) ? " \\u2193" : " \\u2191"; }
function show(id) {
  const clip = DATA.clips.find(c => c.id === id);
  document.getElementById("instruction").textContent = `Instruction: "${clip.instruction}"`;
  const box = document.getElementById("videos");
  box.innerHTML = "";
  videos = DATA.models.map(m => {
    const fig = document.createElement("figure");
    const v = document.createElement("video");
    v.src = `videos/${m.dir}/${id}.mp4`; v.muted = true; v.playsInline = true; v.preload = "auto";
    const cap = document.createElement("figcaption"); cap.textContent = m.label;
    fig.append(v, cap); box.append(fig); return v;
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
    parser.add_argument("--out", required=True)
    parser.add_argument("--paper", help="PDF of the report, copied as <OUT_DIR>/report.pdf")
    args = parser.parse_args()

    clips = sorted(
        c[:-4] for c in os.listdir(os.path.join(args.gen, MODELS[3][0]))
        if c.endswith(".mp4") and all(os.path.exists(os.path.join(args.gen, m, c)) for m, _ in MODELS)
    )  # fmt: skip
    values = {}
    for name, column, label, digits in SCORES:
        with open(os.path.join(args.gen, f"metrics_{name}.csv")) as f:
            for row in csv.DictReader(f):
                if row["clip"] in clips and row[column] not in ("", "nan"):
                    value = round(float(row[column]), digits) if digits else int(float(row[column]))
                    values.setdefault(row["clip"], {}).setdefault(row["model"], {})[label] = value
    data = {"models": [{"dir": m, "label": label} for m, label in MODELS], "scores": [s[2] for s in SCORES],
            "clips": []}  # fmt: skip
    for clip in clips:
        instruction = str(np.load(os.path.join(args.gen, "states", f"{clip}.npz"))["instruction"])
        data["clips"].append({"id": clip, "instruction": instruction, "values": values.get(clip, {})})
        for m, _ in MODELS:
            os.makedirs(os.path.join(args.out, "videos", m), exist_ok=True)
            shutil.copy(os.path.join(args.gen, m, f"{clip}.mp4"), os.path.join(args.out, "videos", m, f"{clip}.mp4"))
    if args.paper:
        shutil.copy(args.paper, os.path.join(args.out, "report.pdf"))
    with open(os.path.join(args.out, "index.html"), "w") as f:
        f.write(PAGE.replace("__DATA__", json.dumps(data)))
    print(f"wrote {args.out}/index.html: {len(clips)} clips, {len(MODELS)} videos each")


if __name__ == "__main__":
    main()
