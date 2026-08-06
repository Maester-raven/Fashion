# Fashion 3.1.3 Installation

Recommended environment: Python 3.10, CUDA GPU, PyTorch with CUDA support, NumPy, Pillow.

```bash
python -m pip install -e .
python -m pip install -r requirements_3_1_3_runtime.txt
python scripts/verify_install.py
```

Real inference also requires the two checkpoint files documented in `MODEL_ASSETS.md`.
