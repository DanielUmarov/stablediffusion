"""
download_cc12m_shard.py

Purpose
-------
Download a single shard from the CC12M (Conceptual Captions 12M) dataset
distributed in WebDataset (.tar) format.

Why this script exists
----------------------
CC12M is too large to download in full. Instead, it is hosted as many
independent tar shards, each containing thousands of image–caption pairs.
This script downloads ONE shard to:

- validate the data pipeline
- enable early preprocessing and sanity checks
- avoid unnecessary storage and bandwidth usage

This approach mirrors how large-scale diffusion models (e.g. Stable Diffusion)
are trained in practice: streaming or shard-based datasets rather than
monolithic archives.
"""

from pathlib import Path
from huggingface_hub import hf_hub_download

# Hugging Face dataset repository containing CC12M in WebDataset format
REPO_ID = "laion/conceptual-captions-12m-webdataset"

# Shard filename inside the dataset repository.
# Each shard is ~1GB and contains ~10k image–caption pairs.
FILENAME = "data/00000.tar"

# Local directory where the shard will be stored
OUT_DIR = Path("data/raw/cc12m_shards")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Download the shard from Hugging Face Hub.
# hf_hub_download handles:
# - resumable downloads
# - local caching
# - correct versioning
#
# The returned path points to the downloaded tar file on disk.
path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    repo_type="dataset",
    local_dir=str(OUT_DIR),
)

print("Downloaded CC12M shard to:", path)
