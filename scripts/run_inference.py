import sys
from pathlib import Path
# FASHION313_REPO_SRC_BOOTSTRAP: prefer this checkout when scripts run from repo root.
_repo_src = Path(__file__).resolve().parents[1] / "src"
if _repo_src.exists():
    sys.path.insert(0, str(_repo_src))
from fashion313_runtime.cli import main
raise SystemExit(main())
