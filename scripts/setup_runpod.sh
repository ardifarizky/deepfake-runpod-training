#!/usr/bin/env bash
# One-time RunPod setup. Uses system Python from the official PyTorch image.
set -Eeuo pipefail
python -m pip install --upgrade pip
python -m pip install \
  'timm==1.0.29' 'pandas>=2.2,<3' 'scikit-learn>=1.5,<2' \
  'matplotlib>=3.8,<4' 'openpyxl>=3.1,<4' 'tabulate>=0.9,<1'
python -c "import torch; print('CUDA:', torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available()"
mkdir -p /workspace/data /workspace/outputs /workspace/logs
python /workspace/project/scripts/download_sdxl.py --output /workspace/data/sdxl_faces --manifest /workspace/data/sdxl_external.csv
