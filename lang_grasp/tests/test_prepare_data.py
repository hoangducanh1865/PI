"""
scripts/prepare_data.py on a tiny fake raw dataset: the split zip is read across its two parts, the authors' 90/10
cut and the Unseen set are kept, grasps are rescaled to 224 and part_distinct is set per object.

Author: Duc Anh Hoang
"""

import importlib.util
import io
import json
import os
import pickle
import zipfile

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")
Image = pytest.importorskip("PIL.Image")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("prepare_data", os.path.join(ROOT, "scripts", "prepare_data.py"))
prepare_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_data)

A, B = "a" * 64, "b" * 64  # image hashes
GRASP = [1.0, 208.0, 104.0, 100.0, 20.0, 30.0]  # quality, x, y, w, h, theta


def mask(rows):
    m = np.zeros((416, 416), dtype=bool)
    m[rows] = True
    return m


# A_0: three parts with different grasps, each inside its own part mask (part-distinct); A_1: one part;
# B_0 (unseen): two parts sharing the same grasps and mask.
SAMPLES = {
    f"{A}_0_0": ([1.0, 208.0, 50.0, 100.0, 20.0, 30.0], mask(slice(40, 60))),
    f"{A}_0_1": ([1.0, 208.0, 200.0, 100.0, 20.0, 30.0], mask(slice(190, 210))),
    f"{A}_0_2": ([1.0, 208.0, 350.0, 100.0, 20.0, 30.0], mask(slice(340, 360))),
    f"{A}_1_0": (GRASP, mask(slice(0, 416))),
    f"{B}_0_0": (GRASP, mask(slice(0, 416))),
    f"{B}_0_1": (GRASP, mask(slice(0, 416))),
}


def write_zip(path, entries):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in entries.items():
            z.writestr(name, data)


def to_bytes(save, obj):
    buf = io.BytesIO()
    save(obj, buf)
    return buf.getvalue()


def test_prepare_data(tmp_path):
    raw, pp, ga = tmp_path / "raw", tmp_path / "raw" / "Grasp-Anything-pp", tmp_path / "raw" / "Grasp-Anything"
    pp.mkdir(parents=True)
    ga.mkdir()
    write_zip(
        pp / "grasp_instructions.zip",
        {f"grasp_instructions/{k}.pkl": pickle.dumps(f"prompt {k[-3:]}") for k in SAMPLES},
    )
    write_zip(
        pp / "grasp_label_positive.zip",
        {f"grasp_label_positive/{k}.pt": to_bytes(torch.save, torch.tensor([g])) for k, (g, _) in SAMPLES.items()},
    )
    write_zip(
        pp / "part_mask.zip",
        {f"part_mask/{k}.npy": to_bytes(lambda m, f: np.save(f, m), m) for k, (_, m) in SAMPLES.items()},
    )

    rng = np.random.default_rng(0)
    images = {
        h: to_bytes(lambda im, f: im.save(f, "JPEG"), Image.fromarray(rng.integers(0, 255, (416, 416, 3), np.uint8)))
        for h in ("0" * 64, A, "1" * 64, B)
    }  # unused images around the wanted ones
    write_zip(tmp_path / "images.zip", {f"image/{h}.jpg": b for h, b in images.items()})
    data = (tmp_path / "images.zip").read_bytes()
    (ga / "image_part_aa").write_bytes(data[: len(data) // 2])  # the zip is cut in two, like the real one
    (ga / "image_part_ab").write_bytes(data[len(data) // 2 :])

    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "seen.obj").write_bytes(pickle.dumps([f"{A}_0", f"{A}_1", "c" * 64 + "_0"]))  # last one has no label
    (splits / "unseen.obj").write_bytes(pickle.dumps([f"{B}_0"]))

    out = tmp_path / "processed"
    prepare_data.main(["--raw", str(raw), "--out", str(out), "--splits", str(splits)])

    stats = json.loads((out / "stats.json").read_text())
    assert [stats[s]["samples"] for s in ("train", "test_seen", "test_unseen")] == [3, 1, 2]  # floor(0.9 * 4) = 3
    assert stats["split_objects_without_labels"] == 1
    with h5py.File(out / "train.h5") as f:
        assert [k.decode() for k in f["keys"]] == [f"{A}_0_0", f"{A}_0_1", f"{A}_0_2"]
        assert f["images"].shape == (1, 224, 224, 3)
        np.testing.assert_allclose(
            f["grasps"][0], [1.0, 112.0, 50 * 224 / 416, 100 * 224 / 416, 20 * 224 / 416, 30.0], rtol=1e-5
        )
        assert f["part_distinct"][:].all()
        assert np.unpackbits(f["masks"][1]).reshape(224, 224)[108].all()  # row 200 of 416 is in the part
    with h5py.File(out / "test_unseen.h5") as f:
        assert not f["part_distinct"][:].any()
        assert list(f["grasp_offsets"]) == [0, 1, 2]

    from src.data.grasp_anything import GraspAnything

    regions = GraspAnything(str(out), "test_unseen", regions=True)  # one object (B_0) with two parts
    sample = regions[0]
    assert sample["regions"].shape == (2, 56, 56)  # the object + the rest of the scene
    assert sample["region_valid"].tolist() == [True, False]  # the full-image mask leaves no "rest"
    train = GraspAnything(str(out), "train", regions=True)  # object A_0 alone in train: parts in 3 rows
    assert train.slots == 2 and train[1]["region_valid"].tolist() == [True, True]
    assert torch.allclose(train[0]["regions"][0], train[2]["regions"][0])  # the same object for every part
