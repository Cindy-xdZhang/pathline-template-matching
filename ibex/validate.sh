#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)
cd "${repo_root}"

source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONPATH="${repo_root}/src"

python -m unittest discover -s tests -p test_all.py -v
python scripts/smoke_test_template_library.py
python scripts/validate_data_access.py \
  --registry config/datasets.yaml \
  --environment ibex \
  --read-raw-sample \
  --require-all-cache \
  --sample-max-spatial-dim 8 \
  --output outputs/Other_ProjectBootstrap_1.1/ibex_data_access.json
