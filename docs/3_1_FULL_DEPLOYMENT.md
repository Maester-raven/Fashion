# Fashion 3.1 full deployment

Use module-level environments for clean-room deployment. `environments/environment_3_1.yml` is retained as a legacy convenience environment, not the primary clean-room path.

```bash
git clone https://github.com/Maester-raven/Fashion.git
cd Fashion
python scripts/setup_module_environment.py --module 3.1.1 --backend conda --prefix .venvs/fashion311
python scripts/setup_module_environment.py --module 3.1.2 --backend conda --prefix .venvs/fashion312
python scripts/setup_module_environment.py --module 3.1.3 --backend venv --prefix .venvs/fashion313
python scripts/download_models.py --module all --model-dir models --backend curl --resume --install
python scripts/download_models.py --module all --model-dir models --verify-only
```

3.1.1 TensorRT engines are target-machine generated from the released `epoch_5.pth`. 3.1.2 supports generic_all, constrained_subset, and no_target through natural-language query parsing without a fixed part_name runtime input. 3.1.3 keeps the accepted `Fashion313Runtime.predict` interface unchanged.
