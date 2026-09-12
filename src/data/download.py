"""Fetch the raw corpus.

Prefers the real Kaggle file if credentials happen to be configured, otherwise
falls back to an ungated HuggingFace mirror of the same dataset. The fallback is
what makes `git clone && make all` actually work for someone else.
"""

import os
import shutil
import subprocess
import sys

import requests

from src import config

MIRROR_URL = (
    "https://huggingface.co/api/datasets/TNE-AI/"
    "customer-support-on-twitter-conversation/parquet/default/train/0.parquet"
)


def have_kaggle_creds():
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    return (config.ROOT / "kaggle.json").exists() or (
        os.path.expanduser("~/.kaggle/kaggle.json") and os.path.exists(
            os.path.expanduser("~/.kaggle/kaggle.json")
        )
    )


def from_kaggle():
    if shutil.which("kaggle") is None:
        print("kaggle CLI not installed (`pip install kaggle`), skipping")
        return False
    print("downloading thoughtvector/customer-support-on-twitter from Kaggle ...")
    rc = subprocess.call(
        [
            "kaggle", "datasets", "download",
            "-d", "thoughtvector/customer-support-on-twitter",
            "-p", str(config.RAW), "--unzip",
        ]
    )
    return rc == 0 and config.RAW_KAGGLE_CSV.exists()


def from_mirror():
    dest = config.RAW_CONVERSATIONS
    if dest.exists() and dest.stat().st_size > 200_000_000:
        print(f"already have {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        return True
    print(f"downloading mirror -> {dest.name} (~207 MB, this is the slow bit)")
    tmp = dest.with_suffix(".part")
    with requests.get(MIRROR_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    sys.stdout.write(f"\r  {done / 1e6:6.0f} / {total / 1e6:.0f} MB  {pct:5.1f}%")
                    sys.stdout.flush()
    print()
    tmp.replace(dest)
    return True


def main():
    if have_kaggle_creds() and from_kaggle():
        print(f"ok -> {config.RAW_KAGGLE_CSV}")
        return
    from_mirror()
    print(f"ok -> {config.RAW_CONVERSATIONS}")


if __name__ == "__main__":
    main()
