# 3.1.2 deployment

Create caches and the environment on the data volume so large CUDA/TensorRT wheels do not fill the system partition:

```bash
git clone https://github.com/Maester-raven/Fashion.git
cd Fashion
mkdir -p /data/conda_pkgs_fashion312 /data/tmp_fashion312 /data/pip_cache_fashion312 /data/envs
CONDA_PKGS_DIRS=/data/conda_pkgs_fashion312 TMPDIR=/data/tmp_fashion312 PIP_CACHE_DIR=/data/pip_cache_fashion312   conda env create --prefix /data/envs/fashion312_release_cleanroom_v2   -f environments/environment_3_1_2_release.yml
conda activate /data/envs/fashion312_release_cleanroom_v2
python scripts/setup_3_1_2_model_assets.py --release-tag 3.1.2-zero-one-n-functional-v1
python scripts/verify_3_1_2_environment.py
python scripts/verify_3_1_2_assets.py
PYTHONPATH=src python scripts/verify_3_1_2_installation.py
PYTHONPATH=src python -m fashion_3_1_2.cli --image example.jpg --parent-bbox 0 0 512 512 --query "find all sleeves" --output output.json
```

Use `--parent-crop` instead of `--image/--parent-bbox` for an already cropped garment. The environment deliberately includes the frozen 3.1.1 TensorRT/MMCV layer so the same environment can run both 3.1.1 and 3.1.2. No 3.1.1 model asset is included in the 3.1.2 asset release.
