# Fashion 3.1.3 Troubleshooting

- If model loading fails, run `python scripts/verify_assets.py ...` and compare SHA256.
- If imports fail, install with `python -m pip install -e .`.
- If latency is slow, confirm CUDA is available and that inference is warm in-memory.
- If a local-part query returns no candidates, provide a parent mask.
