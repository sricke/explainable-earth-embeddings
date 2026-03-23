#!/usr/bin/env bash
set -euo pipefail

# Sweep SatCLIP<->WildSat/GritLM MLP alignment over:
#  - lambda_alignment: 0.0, 0.1, 0.5
#  - scheduler: none, CosineAnnealingWarmRestarts
#  - weight_decay: 0.05, 0.1, 0.25
#  - MLP dropout: 0.0, 0.1, 0.2
#
# Total runs: 3 * 2 * 3 * 3 = 54
#
# Usage:
#   GPU_ID=0 ./run_satclip_wildsat_hparam_sweeps.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_CFG="${REPO_ROOT}/configs/train.yaml"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_ROOT}/logs/wildsat_hparam_sweeps/${TS}"
HOME_DIR="/media/volume/xAi-data"
mkdir -p "${LOG_DIR}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif [[ -x "${HOME_DIR}/miniconda3/envs/xai/bin/python" ]]; then
  PYTHON_BIN="${HOME_DIR}/miniconda3/envs/xai/bin/python"
else
  PYTHON_BIN="python"
fi

GPU_ID="${GPU_ID:-0}"

if [[ ! -f "${BASE_CFG}" ]]; then
  echo "Base config not found: ${BASE_CFG}" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c "import yaml" >/dev/null 2>&1; then
  echo "PyYAML not found in ${PYTHON_BIN}. Set PYTHON_BIN to your training env python." >&2
  exit 1
fi

build_cfg() {
  local lambda_alignment="$1"
  local scheduler_mode="$2"
  local weight_decay="$3"
  local dropout="$4"
  local label="$5"
  local cfg_out="/tmp/train_${label}.yaml"

  "${PYTHON_BIN}" - "${BASE_CFG}" "${cfg_out}" "${lambda_alignment}" "${scheduler_mode}" "${weight_decay}" "${dropout}" "${label}" <<'PY'
import sys
from pathlib import Path
import yaml

base_cfg = Path(sys.argv[1])
cfg_out = Path(sys.argv[2])
lambda_alignment = float(sys.argv[3])
scheduler_mode = sys.argv[4]
weight_decay = float(sys.argv[5])
dropout = float(sys.argv[6])
label = sys.argv[7]

with base_cfg.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("model", {})
cfg["model"].setdefault("hyperparameters", {})
cfg["model"]["hyperparameters"]["finetune_mode"] = "mlp"
cfg["model"]["hyperparameters"]["weight_decay"] = weight_decay
cfg["model"]["hyperparameters"]["lambda_alignment"] = lambda_alignment
cfg["model"].setdefault("text_model", {})
cfg["model"]["text_model"]["train_text_model"] = False
cfg["model"]["text_model"]["projection_head"] = "mlp2"
cfg["model"]["text_model"]["projection_dropout"] = dropout
if scheduler_mode == "none":
    cfg["model"]["hyperparameters"]["lr_scheduler"] = None
elif scheduler_mode == "cosine":
    cfg["model"]["hyperparameters"]["lr_scheduler"] = {
        "class_path": "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
        "init_args": {"T_0": 50, "eta_min": 1e-7},
    }
else:
    raise ValueError(
        f"Unsupported scheduler_mode={scheduler_mode!r}. Use 'none' or 'cosine'."
    )
cfg["model"]["hyperparameters"]["logit_scale_temperature"] = 0.07
cfg["model"]["hyperparameters"]["logit_scale_max"] = 100.0

cfg.setdefault("trainer", {})
cfg["trainer"]["devices"] = [0]
logger = cfg["trainer"].get("logger")
if isinstance(logger, dict):
    init_args = logger.setdefault("init_args", {})
    init_args["mode"] = "online"
    init_args["name"] = label
    init_args["group"] = "wildsat_hparam_sweeps"
    # Keep W&B metric logging, but disable model/checkpoint artifacts.
    init_args["log_model"] = False

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
  local scheduler_mode="$2"
  local weight_decay="$3"
  local dropout="$4"
  local label="$5"
  local cfg_out
  cfg_out="$(build_cfg "${lambda_alignment}" "${scheduler_mode}" "${weight_decay}" "${dropout}" "${label}")"
  local log_file="${LOG_DIR}/${label}.log"

  {
    echo "=================================================================="
    echo "Starting ${label}"
    echo "  lambda_alignment=${lambda_alignment}"
    echo "  scheduler=${scheduler_mode}"
    echo "  weight_decay=${weight_decay}"
    echo "  mlp_dropout=${dropout}"
    echo "  gpu=${GPU_ID}"
    echo "Config: ${cfg_out}"
    echo "Logs: ${log_file}"
    echo "=================================================================="
  } | tee -a "${log_file}"

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${PYTHON_BIN}" -c "from main import cli_main; cli_main('${cfg_out}')" \
    2>&1 | tee -a "${log_file}"
}

schedulers=("none" "cosine")
lambda_alignments=("0.0" "0.1" "0.5")
weight_decays=("0.05" "0.1" "0.25")
dropouts=("0.0" "0.1" "0.2")

echo "Running WildSat hyperparameter sweeps on GPU ${GPU_ID}"
echo "Logs directory: ${LOG_DIR}"

for lam in "${lambda_alignments[@]}"; do
  for scheduler_mode in "${schedulers[@]}"; do
    for wd in "${weight_decays[@]}"; do
      for p in "${dropouts[@]}"; do
        safe_lam="${lam//./p}"
        safe_sched="${scheduler_mode//[^a-zA-Z0-9]/}"
        safe_wd="${wd//./p}"
        safe_p="${p//./p}"
        label="wildsat_lam-${safe_lam}_sched-${safe_sched}_wd-${safe_wd}_dropout-${safe_p}"
        run_one "${lam}" "${scheduler_mode}" "${wd}" "${p}" "${label}"
      done
    done
  done
done

echo "Sweep completed. Logs directory: ${LOG_DIR}"
