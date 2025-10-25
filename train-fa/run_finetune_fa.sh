#!/usr/bin/env bash
set -euo pipefail

# --- repo root + PYTHONPATH ---
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
if REPO_DEFAULT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
fi
ROOT="$REPO_DEFAULT_ROOT" 

stage=0
stop_stage=6
nj=8

# data build (stage 0)
AUDIO_DIRS=""
CSV_PATH=""
TEXT_COL="text_normalized"
FILE_COL="audio_filepath"
SEP=","
DEV_FRAC="0.02"
UTT_PREFIX="m040"

# tokenization/lang
tokenizer="espeak"
lang="fa"

# utterance-length filter (in seconds)
MIN_LEN=1
MAX_LEN=30

# training knobs
SEED=1337
BASE_LR=1e-4
NUM_ITERS=10000                  # if >0, overrides epochs
NUM_EPOCHS=0                     # used when NUM_ITERS=0
MAX_DURATION=100                 # LR schedule ref (ZipVoice arg --max-duration)
SAVE_EVERY_N=1000
PROCESSES=2                      # number of GPUs/processes for accelerate

download_dir="download/"

# ---------- usage ----------
usage() {
  cat <<USAGE
Usage: $0 [options]

General:
  --root PATH              Project root (defaults to Git toplevel)

Data (stage 0):
  --audio-dirs DIRS        One or more audio roots (comma-separated or repeat flag)
  --csv-path PATH          CSV with file/text columns
  --text-col NAME          Default: ${TEXT_COL}
  --file-col NAME          Default: ${FILE_COL}
  --sep SEP                CSV separator, default "," (use "\\t" for TSV)
  --dev-frac F             Default: ${DEV_FRAC}
  --utt-prefix PFX         Default: ${UTT_PREFIX}

Training:
  --num-iters N            Default: ${NUM_ITERS} (if >0, steps-mode)
  --num-epochs N           Default: ${NUM_EPOCHS} (used when --num-iters=0)
  --max-duration D         Default: ${MAX_DURATION}
  --save-every-n N         Default: ${SAVE_EVERY_N}
  --seed S                 Default: ${SEED}
  --lr LR                  Default: ${BASE_LR}
  --min-len S              Default: ${MIN_LEN} sec
  --max-len S              Default: ${MAX_LEN} sec
  --gpus K                 Default: ${PROCESSES}

Pipeline:
  --stage N                Start stage (default ${stage})
  --stop-stage N           Stop stage  (default ${stop_stage})
USAGE
}

# ---------- parse CLI overrides ----------
AUDIO_DIRS_LIST=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)         ROOT="$2"; shift 2;;
    --audio-dirs)   AUDIO_DIRS_LIST+=("$2"); shift 2;;
    --csv-path)     CSV_PATH="$2"; shift 2;;
    --text-col)     TEXT_COL="$2"; shift 2;;
    --file-col)     FILE_COL="$2"; shift 2;;
    --sep)          SEP="$2"; shift 2;;
    --dev-frac)     DEV_FRAC="$2"; shift 2;;
    --utt-prefix)   UTT_PREFIX="$2"; shift 2;;
    --num-iters)    NUM_ITERS="$2"; shift 2;;
    --num-epochs)   NUM_EPOCHS="$2"; shift 2;;
    --max-duration) MAX_DURATION="$2"; shift 2;;
    --save-every-n) SAVE_EVERY_N="$2"; shift 2;;
    --seed)         SEED="$2"; shift 2;;
    --lr|--base-lr) BASE_LR="$2"; shift 2;;
    --min-len)      MIN_LEN="$2"; shift 2;;
    --max-len)      MAX_LEN="$2"; shift 2;;
    --gpus|--procs|--num-processes) PROCESSES="$2"; shift 2;;
    --stage)        stage="$2"; shift 2;;
    --stop-stage)   stop_stage="$2"; shift 2;;
    -h|--help)      usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done

if ((${#AUDIO_DIRS_LIST[@]})); then
  IFS=, read -r AUDIO_DIRS <<<"${AUDIO_DIRS_LIST[*]}"
fi

cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# ---------- Stage 0: Build TSVs ----------
if [[ ${stage} -le 0 && ${stop_stage} -ge 0 ]]; then
  if [[ -z "${AUDIO_DIRS}" || -z "${CSV_PATH}" ]]; then
    echo "Stage 0 requires --audio-dirs and --csv-path" >&2
    exit 1
  fi
  echo "Stage 0: Build TSVs from CSV"
  python3 "train-fa/build_tsv.py" \
    --audio-dirs "${AUDIO_DIRS}" \
    --csv-path   "${CSV_PATH}" \
    --text-col   "${TEXT_COL}" \
    --file-col   "${FILE_COL}" \
    --sep        "${SEP}" \
    --dev-frac   "${DEV_FRAC}" \
    --out-dir    data/raw \
    --utt-prefix "${UTT_PREFIX}"
fi

# sanity
for subset in train dev; do
  [[ -f "data/raw/custom_${subset}.tsv" ]] || { echo "Missing data/raw/custom_${subset}.tsv" >&2; exit 1; }
done

#############################################
# Stage 1: Manifests
#############################################
if [[ ${stage} -le 1 && ${stop_stage} -ge 1 ]]; then
  echo "Stage 1: Prepare manifests"
  for subset in train dev; do
    python3 zipvoice/bin/prepare_dataset.py \
      --tsv-path "data/raw/custom_${subset}.tsv" \
      --prefix custom-finetune \
      --subset "raw_${subset}" \
      --num-jobs "${nj}" \
      --output-dir data/manifests
  done
fi

#############################################
# Stage 2: Tokens
#############################################
if [[ ${stage} -le 2 && ${stop_stage} -ge 2 ]]; then
  echo "Stage 2: Add tokens"
  for subset in train dev; do
    python3 zipvoice/bin/prepare_tokens.py \
      --input-file  "data/manifests/custom-finetune_cuts_raw_${subset}.jsonl.gz" \
      --output-file "data/manifests/custom-finetune_cuts_${subset}.jsonl.gz" \
      --tokenizer "${tokenizer}" \
      --lang "${lang}"
  done
fi

#############################################
# Stage 3: Fbank
#############################################
if [[ ${stage} -le 3 && ${stop_stage} -ge 3 ]]; then
  echo "Stage 3: Compute Fbank"
  for subset in train dev; do
    python3 zipvoice/bin/compute_fbank.py \
      --source-dir data/manifests \
      --dest-dir   data/fbank \
      --dataset    custom-finetune \
      --subset     "${subset}" \
      --num-jobs   "${nj}"
  done
fi

#############################################
# Stage 4: Pretrained weights
#############################################
if [[ ${stage} -le 4 && ${stop_stage} -ge 4 ]]; then
  echo "Stage 4: Download pretrained"
  hf_repo=k2-fsa/ZipVoice
  mkdir -p "${download_dir}"
  huggingface-cli download --local-dir "${download_dir}" ${hf_repo} zipvoice/model.pt
  huggingface-cli download --local-dir "${download_dir}" ${hf_repo} zipvoice/model.json
  huggingface-cli download --local-dir "${download_dir}" ${hf_repo} zipvoice/tokens.txt
fi

#############################################
# Stage 5: Train (Accelerate)
#############################################
if [[ ${stage} -le 5 && ${stop_stage} -ge 5 ]]; then
  echo "Stage 5: Finetune (${PROCESSES} GPU(s))"
  export NCCL_DEBUG=INFO
  export NCCL_IB_DISABLE=1
  export OMP_NUM_THREADS=2
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((PROCESSES-1)))"

  accelerate launch --multi_gpu --num_processes="${PROCESSES}" --mixed_precision=fp16 \
      "train-fa/train_zipvoice_fa.py" \
      --finetune 1 \
      --seed "${SEED}" \
      --base-lr "${BASE_LR}" \
      --num-iters "${NUM_ITERS}" \
      --num-epochs "${NUM_EPOCHS}" \
      --save-every-n "${SAVE_EVERY_N}" \
      --max-duration "${MAX_DURATION}" \
      --min-len "${MIN_LEN}" \
      --max-len "${MAX_LEN}" \
      --model-config "${download_dir}/zipvoice/model.json" \
      --checkpoint   "${download_dir}/zipvoice/model.pt" \
      --tokenizer "${tokenizer}" --lang "${lang}" \
      --token-file "${download_dir}/zipvoice/tokens.txt" \
      --dataset custom \
      --train-manifest "data/fbank/custom-finetune_cuts_train.jsonl.gz" \
      --dev-manifest   "data/fbank/custom-finetune_cuts_dev.jsonl.gz" \
      --exp-dir "exp/zipvoice_finetune"
fi

#############################################
# Stage 6: Average
#############################################
if [[ ${stage} -le 6 && ${stop_stage} -ge 6 ]]; then
  echo "Stage 6: Average checkpoints"
  python3 zipvoice/bin/generate_averaged_model.py \
      --iter 10000 \
      --avg  2 \
      --model-name zipvoice \
      --exp-dir "exp/zipvoice_finetune"
fi

echo "Done."
