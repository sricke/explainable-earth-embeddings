#!/usr/bin/env bash
set -euo pipefail

# Run 2 WildSat SatCLIP<->GritLM alignment experiments (MLP-only):
# 1) lambda_alignment=0.0 on GPU 0
# 2) lambda_alignment=0.1 on GPU 3
#
# Base config: configs/train.yaml
# Output logs: logs/wildsat_chunk_experiments/<timestamp>/

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_CFG="${REPO_ROOT}/configs/train.yaml"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_ROOT}/logs/wildsat_chunk_experiments/${TS}"
mkdir -p "${LOG_DIR}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif [[ -x "/home/libe2152/miniconda3/envs/xai/bin/python" ]]; then
  PYTHON_BIN="/home/libe2152/miniconda3/envs/xai/bin/python"
else
  PYTHON_BIN="python"
fi

if [[ ! -f "${BASE_CFG}" ]]; then
  echo "Base config not found: ${BASE_CFG}" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c "import yaml" >/dev/null 2>&1; then
  echo "PyYAML not found in ${PYTHON_BIN}. Set PYTHON_BIN to your training env python." >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_LIST="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d '[:space:]' | tr '\n' ' ')"
  if [[ "${GPU_LIST}" != *"0"* || "${GPU_LIST}" != *"3"* ]]; then
    echo "Required GPUs 0 and 3 are not both visible. Visible GPUs: ${GPU_LIST}" >&2
    exit 1
  fi
fi

build_cfg_for_lambda() {
  local lambda_alignment="$1"
  local label="$2"
  local cfg_out="/tmp/train_${label}.yaml"

  "${PYTHON_BIN}" - "${BASE_CFG}" "${cfg_out}" "${lambda_alignment}" "${label}" <<'PY'
import sys
from pathlib import Path
import yaml

base_cfg = Path(sys.argv[1])
cfg_out = Path(sys.argv[2])
lambda_alignment = float(sys.argv[3])
label = sys.argv[4]

with base_cfg.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("model", {})
# MLP-only training:
# - keep pretrained GritLM frozen
# - train only projection head
cfg["model"]["train_text_model"] = False
cfg["model"]["finetune_mode"] = "mlp"
cfg["model"]["text_projection_head"] = "mlp2"
cfg["model"]["checkpoint_projection_only"] = True
cfg["model"]["lambda_alignment"] = lambda_alignment
cfg["model"]["text_chunk_granularity"] = None

cfg.setdefault("trainer", {})
cfg["trainer"]["devices"] = [0]
logger = cfg["trainer"].get("logger")
if isinstance(logger, dict):
    init_args = logger.setdefault("init_args", {})
    init_args["mode"] = "online"
    init_args["name"] = label

callbacks = cfg.get("trainer", {}).get("callbacks", [])
for cb in callbacks:
    if cb.get("class_path", "").endswith("ModelCheckpoint"):
        init_args = cb.setdefault("init_args", {})
        init_args["save_weights_only"] = True
        base_dir = str(init_args.get("dirpath", ""))
        init_args["dirpath"] = f"{base_dir.rstrip('/')}/{label}"

with cfg_out.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
  echo "${cfg_out}"
}

run_one() {
  local lambda_alignment="$1"
  local gpu_id="$2"
  local label="$3"
  local cfg_out
  cfg_out="$(build_cfg_for_lambda "${lambda_alignment}" "${label}")"
  local log_file="${LOG_DIR}/${label}.log"

  {
    echo "=================================================================="
    echo "Starting ${label}: lambda_alignment=${lambda_alignment}, GPU=${gpu_id}"
    echo "Config: ${cfg_out}"
    echo "Logs: ${log_file}"
    echo "=================================================================="
  } | tee -a "${log_file}"

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu_id}" \
    "${PYTHON_BIN}" -c "from main import cli_main; cli_main('${cfg_out}')" \
    2>&1 | tee -a "${log_file}"
}

echo "Running two experiments sequentially:"
echo "- mlp_lambda0_gpu0  (lambda_alignment=0.0, GPU 0)"
echo "- mlp_lambda01_gpu3 (lambda_alignment=0.1, GPU 3)"

run_one "0.0" "0" "mlp_lambda0_gpu0"
run_one "0.1" "3" "mlp_lambda01_gpu3"

echo "Both experiments completed. Logs directory: ${LOG_DIR}"
