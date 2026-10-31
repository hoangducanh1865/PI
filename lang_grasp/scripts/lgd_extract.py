"""
Lay out the official Grasp-Anything(++) files the way the official LGD code reads them, for its test objects only.

Usage (on the machine that runs the LGD code, after downloading from Hugging Face):
    python scripts/lgd_extract.py --raw <RAW_DIR> --lgd <LGD_REPO> --out <OUT_DIR>
<RAW_DIR>: Grasp-Anything/{image_part_aa,image_part_ab,scene_description.zip} (airvlab/Grasp-Anything) and
Grasp-Anything-pp/grasp_label_positive.zip (airvlab/Grasp-Anything-pp). <LGD_REPO>: github.com/Fsoft-AIC/LGD at
0223d43. The objects are the union of its split/grasp-anything++/test/{seen,unseen}.obj, which is all the released
training (train_network.py --seen 1) and evaluation (--seen 1 / 0) commands of the README read. Output:
    <OUT_DIR>/ga/image/<image>.jpg, <OUT_DIR>/ga/prompt/<image>.pkl (scene_description), <OUT_DIR>/ga/mask/ (empty)
    <OUT_DIR>/gapp/positive_grasp/<image>_<object>_<part>.pt, <OUT_DIR>/gapp/grasp_instructions/ (empty)
matching utils/data/grasp_anywhere_data.py (dataset path = ga, add_file_path = gapp).

Author: Duc Anh Hoang
"""

import argparse
import io
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import open_zip, read_in_file_order  # noqa: E402
from resplit_lgd import StringListUnpickler  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--lgd", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    objects = set()
    for name in ("seen", "unseen"):
        with open(os.path.join(args.lgd, "split", "grasp-anything++", "test", f"{name}.obj"), "rb") as f:
            objects |= set(StringListUnpickler(io.BytesIO(f.read())).load())
    images = {o.split("_")[0] for o in objects}
    for d in ("ga/image", "ga/prompt", "ga/mask", "gapp/positive_grasp", "gapp/grasp_instructions"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)

    with zipfile.ZipFile(os.path.join(args.raw, "Grasp-Anything-pp", "grasp_label_positive.zip")) as z:
        names = [n for n in z.namelist() if n.endswith(".pt") and os.path.basename(n)[:-5] in objects]
        for n in names:
            with open(os.path.join(args.out, "gapp", "positive_grasp", os.path.basename(n)), "wb") as f:
                f.write(z.read(n))
    print(f"positive grasps: {len(names)} files for {len(objects)} objects", flush=True)

    with zipfile.ZipFile(os.path.join(args.raw, "Grasp-Anything", "scene_description.zip")) as z:
        for h in images:
            with open(os.path.join(args.out, "ga", "prompt", f"{h}.pkl"), "wb") as f:
                f.write(z.read(f"scene_description/{h}.pkl"))
    print(f"scene descriptions: {len(images)}", flush=True)

    raw = os.path.join(args.raw, "Grasp-Anything")
    with open_zip(os.path.join(raw, "image_part_aa"), os.path.join(raw, "image_part_ab")) as z:
        files = read_in_file_order(z, [f"image/{h}.jpg" for h in sorted(images)], lambda b: b, every=200)
    for n, b in files.items():
        with open(os.path.join(args.out, "ga", n), "wb") as f:
            f.write(b)
    print(f"images: {len(files)} of {len(images)}", flush=True)  # .pkl files are copied as bytes, never unpickled


if __name__ == "__main__":
    main()
