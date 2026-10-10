#!/usr/bin/env bash
# One-time environment setup for streaming fine-tuning on an NVIDIA H100 box.
#
#   bash train_utils/streaming/setup_h100.sh            # creates ./.venv-h100
#   source .venv-h100/bin/activate
#
# Installs the current stable PyTorch (2.14.1) from the CUDA 13.0 wheel index
# (the PyPI default since 2.11; H100 needs driver >= 580 for CUDA 13, use
# TORCH_INDEX=https://download.pytorch.org/whl/cu126 on an older driver),
# torchaudio/torchcodec for audio I/O, and the package with its train extras.
set -euo pipefail

cd "$(dirname "$0")/../.."
PY="${PYTHON_VERSION:-3.12}"
VENV="${VENV:-.venv-h100}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python "$PY" "$VENV"
export VIRTUAL_ENV="$PWD/$VENV"
uv pip install "torch==2.14.1" "torchaudio==2.11.0" "torchcodec" --index-url "$TORCH_INDEX"
uv pip install -e ".[train,fp8,tests]" "pyarrow>=15" "soundfile" "huggingface_hub"

"$VENV/bin/python" - <<'EOF'
import torch, torchaudio
print("torch", torch.__version__, "cuda", torch.version.cuda, "torchaudio", torchaudio.__version__)
assert torch.cuda.is_available(), "CUDA not available: check the driver / TORCH_INDEX"
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print("gpu", name, "sm", cap)
assert torch.cuda.is_bf16_supported(), "bf16 unsupported on this GPU"
# inductor must be able to build a kernel (needs a C compiler + triton)
f = torch.compile(lambda x: (x.sin() * 2).sum())
print("torch.compile ok:", float(f(torch.randn(16, device="cuda"))))
EOF
echo "Environment ready: source $VENV/bin/activate"
