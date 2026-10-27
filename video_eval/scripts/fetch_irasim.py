"""
Fetch only the needed files of the IRASim release (checkpoints 29 to 34 GB and evaluation data 63 to 194 GB per
dataset), without downloading whole archives.

Usage (from video_eval/):
    python scripts/fetch_irasim.py list <ARCHIVE> [--max-seconds 120]
    python scripts/fetch_irasim.py get <ARCHIVE> --out <DIR> --match <REGEX> [--match <REGEX> ...] [--max-files N]
    e.g. python scripts/fetch_irasim.py get bridge_checkpoints_data --out /tmp/irasim --match 'frame_ada/.*\\.pt$'

<ARCHIVE> is a file name of the release without extension (bridge_checkpoints_data, bridge_evaluation_data, ...).
Members are written under <DIR> with their path inside the archive (opensource_robotdata/...), the layout IRASim's
configs expect. Checked on 2026-10-27: the checkpoint archives are plain tar despite their .tar.gz name and the
server serves byte ranges, so a member is fetched alone by reading the 512-byte tar headers; the train and evaluation
archives are real gzip, so they are streamed from the start and only matching members are kept (--max-files stops
the stream early).

Author: Duc Anh Hoang
"""

import argparse
import os
import re
import shutil
import tarfile
import time
import urllib.request

URL = "https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/opensource_IRASim_v1/{}.tar.gz"
BLOCK = 512


def open_range(url: str, start: int, length: int | None = None):
    end = "" if length is None else start + length - 1
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    return urllib.request.urlopen(request, timeout=60)


def is_gzip(url: str) -> bool:
    return open_range(url, 0, 2).read() == b"\x1f\x8b"


def plain_members(url: str):
    """(TarInfo, data offset) of every member of an uncompressed remote tar, one header request each."""
    offset = 0
    while True:
        header = open_range(url, offset, BLOCK).read()
        if header == b"\0" * BLOCK:
            return
        info = tarfile.TarInfo.frombuf(header, "utf-8", "surrogateescape")
        yield info, offset + BLOCK
        offset += BLOCK + (info.size + BLOCK - 1) // BLOCK * BLOCK


def write(source, path: str, size: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    start = time.time()
    with open(path, "wb") as f:
        shutil.copyfileobj(source, f, length=16 << 20)
    assert os.path.getsize(path) == size, f"{path}: {os.path.getsize(path)} bytes, expected {size}"
    print(f"{path}  {size / 1e9:.2f} GB in {time.time() - start:.0f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("list", "get"))
    parser.add_argument("archive", help="e.g. bridge_checkpoints_data")
    parser.add_argument("--out", help="folder the members are written under (get)")
    parser.add_argument("--match", action="append", default=[], help="regex on the member path (get), repeatable")
    parser.add_argument("--max-files", type=int, help="stop after this many matching files (get)")
    parser.add_argument("--max-seconds", type=float, default=120, help="stop listing a gzip archive after this (list)")
    args = parser.parse_args()
    url = URL.format(args.archive)
    patterns = [re.compile(m) for m in args.match]
    start, kept, last_dir = time.time(), 0, None

    if not is_gzip(url):
        for info, data_offset in plain_members(url):
            if args.action == "list":
                print(f"{info.name}  {info.size / 1e9:.3f} GB  offset {data_offset}")
            elif info.isfile() and any(p.search(info.name) for p in patterns):
                write(open_range(url, data_offset, info.size), os.path.join(args.out, info.name), info.size)
        return

    with urllib.request.urlopen(url, timeout=60) as response, tarfile.open(fileobj=response, mode="r|gz") as tar:
        for info in tar:
            if args.action == "list":
                folder = os.path.dirname(info.name)
                if folder != last_dir:  # one line per folder: the archive holds tens of thousands of files
                    print(f"{time.time() - start:6.0f} s  {folder}/  e.g. {os.path.basename(info.name)}", flush=True)
                    last_dir = folder
                if time.time() - start > args.max_seconds:
                    return
            elif info.isfile() and any(p.search(info.name) for p in patterns):
                write(tar.extractfile(info), os.path.join(args.out, info.name), info.size)
                kept += 1
                if args.max_files and kept >= args.max_files:
                    return


if __name__ == "__main__":
    main()
