#!/usr/bin/env bash
set -euo pipefail

BUCKET="s3://danielumarov-diffusion-data"
S3_BASE="${BUCKET}/latents/sd15-vae"

LOCAL_TAR_DIR="data/raw/cc12m_shards/data"
TMP_DIR="data/raw/tmp_cc12m"

COUNT=${1:-10}  # “next 10 tars” by default

# --------- Find LAST processed tar id on S3 (based on manifest.jsonl) ---------
LAST=$(
  aws s3 ls "${S3_BASE}/" --recursive \
    | awk '{print $4}' \
    | grep -oE 'cc12m_256_[0-9]{5}/manifest\.jsonl' \
    | sed -E 's/cc12m_256_([0-9]{5}).*/\1/' \
    | sort -n \
    | tail -n 1 || true
)

if [ -z "${LAST}" ]; then
  START=0
  echo "[AUTO] No processed tars found on S3. Starting at 00000."
else
  START=$((10#${LAST} + 1))
  echo "[AUTO] Last processed tar on S3: ${LAST}. Starting at $(printf "%05d" "${START}")."
fi

END=$((START + COUNT - 1))
echo "[AUTO] Will process next ${COUNT} tars: $(printf "%05d" "${START}")..$(printf "%05d" "${END}")"

# optional local state log
STATE_FILE="state/cc12m_256_processed.txt"
mkdir -p "$(dirname "$STATE_FILE")"

# --------- Main loop ---------
for i in $(seq "${START}" "${END}"); do
  T=$(printf "%05d" "${i}")

  S3_OUT="${S3_BASE}/cc12m_256_${T}"
  LOCAL_PROC="data/processed_${T}"
  LOCAL_OUT="data/latents/sd15-vae/cc12m_256_${T}"

  echo "=============================="
  echo "TAR ${T}"
  echo "=============================="

  # Skip if already uploaded (manifest exists)
  if aws s3 ls "${S3_OUT}/manifest.jsonl" >/dev/null 2>&1; then
    echo "[skip] ${T} already exists on S3: ${S3_OUT}"
    echo "${T}" >> "$STATE_FILE" || true
    continue
  fi

  # Clean any leftovers
  rm -rf "${TMP_DIR}" "${LOCAL_PROC}" "${LOCAL_OUT}"
  mkdir -p "${TMP_DIR}"

  # Download tar if missing locally
  if [ ! -f "${LOCAL_TAR_DIR}/${T}.tar" ]; then
    echo "[download] HF data/${T}.tar"
    python3 scripts/download_cc12m_shard.py --shard "${T}" --out_dir "data/raw/cc12m_shards"
  fi

  # Extract one tar (avoids filename collisions)
  echo "[extract] ${LOCAL_TAR_DIR}/${T}.tar -> ${TMP_DIR}"
  tar -xf "${LOCAL_TAR_DIR}/${T}.tar" -C "${TMP_DIR}"

  # Scan/filter (writes filelist.txt)
  echo "[scan/filter]"
  python3 preprocessing/scan_and_filter.py \
    --raw_dir "${TMP_DIR}" \
    --out_dir "${LOCAL_PROC}" \
    --target_size 256

  # Build pairs.tsv from filelist.txt + .txt captions
  echo "[pairs.tsv]"
  python3 - <<PY
from pathlib import Path
T="${T}"
fl=Path(f"data/processed_{T}/filelist.txt")
out=Path(f"data/processed_{T}/pairs.tsv")
n=0
with fl.open() as f, out.open("w", encoding="utf-8") as g:
  for line in f:
    img=Path(line.strip())
    cap=img.with_suffix(".txt")
    if cap.exists():
      txt=cap.read_text(encoding="utf-8", errors="ignore").strip().replace("\t"," ")
      g.write(f"{img}\t{txt}\n")
      n+=1
print("pairs", T, n)
PY

  # Build latents (your script)
  echo "[latents]"
  python3 preprocessing/build_latents.py \
    --pairs "data/processed_${T}/pairs.tsv" \
    --batch_size 128 \
    --shard_size 5000 \
    --dtype fp16 \
    --out_dir "${LOCAL_OUT}"

  # Upload to S3
  echo "[upload] ${LOCAL_OUT} -> ${S3_OUT}"
  aws s3 sync "${LOCAL_OUT}" "${S3_OUT}" --only-show-errors

  # Verify + log
  aws s3 ls "${S3_OUT}/" --summarize --human-readable
  echo "${T}" >> "$STATE_FILE" || true

  # Cleanup to keep disk small
  rm -rf "${TMP_DIR}" "${LOCAL_PROC}" "${LOCAL_OUT}"
done
