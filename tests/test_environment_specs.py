import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fashion311_runtime_pins_torch_fsspec_dependency():
    req = (ROOT / "envs/3_1_1/requirements-runtime.txt").read_text().splitlines()
    assert "fsspec==2026.4.0" in [line.strip() for line in req]
    lock = json.loads((ROOT / "envs/3_1_1/runtime-lock.json").read_text())
    assert lock["required_packages"]["fsspec"] == "2026.4.0"


def test_fashion312_and_313_specs_do_not_pin_fashion311_fsspec_fix():
    for module in ["3_1_2", "3_1_3"]:
        module_dir = ROOT / "envs" / module
        texts = []
        for path in ["requirements-runtime.txt", "runtime-lock.json", "environment-minimal.yml"]:
            p = module_dir / path
            if p.exists():
                texts.append(p.read_text())
        assert "fsspec==2026.4.0" not in "\n".join(texts)
