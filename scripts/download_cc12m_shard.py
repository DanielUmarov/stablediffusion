"""
download_cc12m_shard.py

Purpose
-------
Download a single shard from the CC12M (Conceptual Captions 12M) dataset
distributed in WebDataset (.tar) format.

Why this script exists
----------------------
CC12M is too large to download in full (~12M samples). Instead, it is hosted
as many independent tar shards, each containing ~10k image–caption pairs.

This script downloads ONE shard at a time in order to:
- validate the preprocessing + latent pipeline incrementally
- avoid unnecessary storage and bandwidth usage
- mirror how diffusion models are trained in practice (shard-based streaming)

Design philosophy
-----------------
- Shards are the unit of scale, not individual images
- Downloads must be resumable and cache-aware
- This script should be safe to re-run (idempotent via HF cache)

This script does *not* perform extraction or preprocessing.
It only retrieves raw data.
"""

from pathlib import Path
import argparse
from huggingface_hub import hf_hub_download

# Hugging Face dataset repository containing CC12M in WebDataset format
REPO_ID = "laion/conceptual-captions-12m-webdataset"

def download_shard(shard_id: str, out_dir: Path) -> Path:
    """
    Download a single CC12M tar shard by shard ID.

    Parameters
    ----------
    shard_id : str
        Five-digit shard identifier (e.g. "00000", "00042").
    out_dir : Path
        Local directory where shards are stored.

    Returns
    -------
    Path
        Path to the downloaded tar file on disk.
    """
    shard_id = shard_id.zfill(5)
    filename = f"data/{shard_id}.tar"

    out_dir.mkdir(parents=True, exist_ok=True)

    # hf_hub_download handles:
    # - resumable downloads
    # - local caching
    # - versioning
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        local_dir=str(out_dir),
    )

    return Path(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download one CC12M shard")
    parser.add_argument("--shard", required=True, help="Shard ID (e.g. 00000)")
    parser.add_argument(
        "--out_dir",
        default="data/raw/cc12m_shards",
        help="Local directory for downloaded shards"
    )
    args = parser.parse_args()

    shard_path = download_shard(args.shard, Path(args.out_dir))
    print("Downloaded CC12M shard to:", shard_path)
