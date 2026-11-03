#!/usr/bin/env bash
set -euo pipefail
CAMPAIGN="$HOME/rsna-resnet50-20261013"
mkdir -p "$CAMPAIGN"
exec 8>"$CAMPAIGN/bootstrap.lock"
flock -n 8 || exit 0
trap 'code=$?; printf "{\"state\":\"setup_failed\",\"exit_code\":%s}\n" "$code" > "$CAMPAIGN/deployment_status.json"' ERR
printf '{"state":"checking_package"}\n' > "$CAMPAIGN/deployment_status.json"
printf '%s  %s\n' "$1" "$HOME/resnet50_cloud_10epochs.tar.gz" | sha256sum -c -
tar -xzf "$HOME/resnet50_cloud_10epochs.tar.gz" -C "$CAMPAIGN"
cd "$CAMPAIGN/resnet50_cloud"
chmod 700 .auth
chmod 600 .auth/kaggle.json
export KAGGLE_CONFIG_DIR="$PWD/.auth"
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
nvidia-smi
df -h .
if [[ ! -x .venv/bin/python ]]; then
    if ! python3 -m venv --system-site-packages .venv; then
        sudo -n apt-get update
        sudo -n apt-get install -y python3-venv
        python3 -m venv --system-site-packages .venv
    fi
fi
export PATH="$PWD/.venv/bin:$PATH"
python -m pip install --upgrade pip
if ! python -c 'import torch,torchvision; assert tuple(map(int,torch.__version__.split("+")[0].split(".")[:2])) >= (2,6); assert torch.cuda.is_available(); torch.ones(1,device="cuda").sum().item()'; then
    # Official matched pair, also matching the local test runtime.
    python -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126
fi
python -m pip install timm==1.0.22 numpy pandas scikit-learn kaggle
python -c 'import torch,torchvision,timm; assert torch.cuda.is_available(); torch.ones(1,device="cuda").sum().item(); print(torch.__version__,torchvision.__version__,timm.__version__,torch.cuda.get_device_name())'
python -m pip freeze > "$CAMPAIGN/environment.txt"
printf '{"state":"staging_and_training"}\n' > "$CAMPAIGN/deployment_status.json"
bash scripts/launch_resnet50_cloud.sh
printf '{"state":"training_complete","epochs":10}\n' > "$CAMPAIGN/deployment_status.json"
