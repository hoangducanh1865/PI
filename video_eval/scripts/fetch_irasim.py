"""
Fetch only the needed files of the IRASim release (checkpoints 29 to 34 GB and evaluation data 63 to 194 GB per
dataset), without downloading whole archives.

Usage (from video_eval/):
    python scripts/fetch_irasim.py list <ARCHIVE> [--max-seconds 120]
    python scripts/fetch_irasim.py get <ARCHIVE> --out <DIR> --match <REGEX> [--match <REGEX> ...] [--max-files N]
    e.g. python scripts/fetch_irasim.py get bridge_checkpoints_data --out /tmp/irasim --match 'frame_ada/.*\\.pt$'

<ARCHIVE> is a file name of the release without extension (bridge_checkpoints_data, bridge_evaluation_data, ...).
Members are written under <DIR> with their path inside the archive (opensource_robotdata/...), the layout IRASim's
configs expect. Source: the authors' Hugging Face mirror fangqi/IRASim, where each archive is cut into 2 GiB parts
(on Kaggle 82 MB/s, against 0.5 MB/s from the original server; 2026-10-27). Checked the same day: the checkpoint
archives are plain tar despite their .tar.gz name, so a member is fetched alone by reading the 512-byte tar headers
with byte ranges; the train and evaluation archives are real gzip, so they are streamed from the start and only
matching members are kept (--max-files stops the stream early).

Author: Duc Anh Hoang
"""

import argparse
import io
import json
import os
import re
import shutil
import tarfile
import time
import urllib.request

LISTING = "https://huggingface.co/api/datasets/fangqi/IRASim/tree/main"
FILE = "https://huggingface.co/datasets/fangqi/IRASim/resolve/main/{}"
PART = 2**31  # every part but the last is 2 GiB: archive byte b is byte b % PART of part b // PART
BLOCK = 512


class Archive:
    """One release archive seen as a single byte sequence over its parts on Hugging Face."""

    def __init__(self, name: str):
        with urllib.request.urlopen(LISTING, timeout=60) as response:
            parts = sorted(f["path"] for f in json.load(response) if f["path"].startswith(f"{name}.tar.gz.sfpart_"))
        assert parts, f"no parts of {name} on the mirror"
        self.urls = [FILE.format(p) for p in parts]

    def chunks(self, start: int, length: int | None = None):
        """Bytes [start, start + length) as chunks, crossing part boundaries; to the end if length is None."""
        end = None if length is None else start + length
        while end is None or start < end:
            part, offset = divmod(start, PART)
            if part >= len(self.urls):
                return
            stop = PART if end is None else min(PART, offset + end - start)
            request = urllib.request.Request(self.urls[part], headers={"Range": f"bytes={offset}-{stop - 1}"})
            with urllib.request.urlopen(request, timeout=60) as response:
                while chunk := response.read(16 << 20):
                    start += len(chunk)
                    yield chunk
            if part == len(self.urls) - 1:  # the last part is shorter than PART
                return

    def read(self, start: int, length: int) -> bytes:
        return b"".join(self.chunks(start, length))

    def stream(self, start: int = 0, length: int | None = None) -> io.BufferedReader:
        return io.BufferedReader(Reader(self.chunks(start, length)), buffer_size=16 << 20)


class Reader(io.RawIOBase):
    """A read-only file over a generator of byte chunks (for tarfile and shutil)."""

    def __init__(self, chunks):
        self.chunks, self.rest = chunks, b""

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        if not self.rest:
            self.rest = next(self.chunks, b"")
        n = min(len(buffer), len(self.rest))
        buffer[:n], self.rest = self.rest[:n], self.rest[n:]
        return n


def plain_members(archive: Archive):
    """(TarInfo, data offset) of every member of an uncompressed tar, one header request each."""
    offset = 0
    while True:
        header = archive.read(offset, BLOCK)
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
    archive = Archive(args.archive)
    patterns = [re.compile(m) for m in args.match]
    start, kept, last_dir = time.time(), 0, None

    if archive.read(0, 2) != b"\x1f\x8b":  # not gzip: plain tar
        for info, data_offset in plain_members(archive):
            if args.action == "list":
                print(f"{info.name}  {info.size / 1e9:.3f} GB  offset {data_offset}")
            elif info.isfile() and any(p.search(info.name) for p in patterns):
                write(archive.stream(data_offset, info.size), os.path.join(args.out, info.name), info.size)
        return

    with tarfile.open(fileobj=archive.stream(), mode="r|gz") as tar:
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
