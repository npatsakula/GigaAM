#!/usr/bin/env bash
# One-time environment setup for streaming fine-tuning on an NVIDIA H100 box.
#
#   bash train_utils/streaming/setup_h100.sh
#
# The repository is a uv project: pyproject.toml pins PyTorch 2.14.1 (the
# current stable release) and routes torch/torchaudio/torchcodec to the CUDA
# 13.0 wheel index on Linux; uv.lock freezes every transitive version. This
# script only runs `uv sync` from that configuration into ./.venv and then
# checks that CUDA, bf16 and torch.compile kernel builds work on this GPU.
#
# CUDA 13.0 wheels need NVIDIA driver >= 580. On an older driver, switch the
# index in pyproject.toml ([[tool.uv.index]] url -> .../whl/cu126) and re-lock.
set -euo pipefail
cd "$(dirname "$0")/../.."

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# .python-version pins CPython 3.12: uv downloads a managed interpreter if
# the box has none, so no system python / -dev headers are involved and every
# dependency installs from prebuilt wheels (the system 3.14 on some images has
# no wheels for editdistance and friends).
# --frozen: install exactly uv.lock, never re-resolve on the server.
uv python install 3.12
uv sync --frozen --extra torch --extra train --extra fp8 --extra tests

uv run --frozen python - <<'EOF'
import torch, torchaudio
print("torch", torch.__version__, "cuda", torch.version.cuda, "torchaudio", torchaudio.__version__)
assert torch.cuda.is_available(), "CUDA not available: check the driver / wheel index"
print("gpu", torch.cuda.get_device_name(0), "sm", torch.cuda.get_device_capability(0))
assert torch.cuda.is_bf16_supported(), "bf16 unsupported on this GPU"
# inductor must be able to build a kernel (needs a C compiler + triton)
f = torch.compile(lambda x: (x.sin() * 2).sum())
print("torch.compile ok:", float(f(torch.randn(16, device="cuda"))))
EOF
echo "Environment ready in .venv (use 'uv run ...' or 'source .venv/bin/activate')"
