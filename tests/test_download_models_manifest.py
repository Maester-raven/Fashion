import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('download_models', Path('scripts/download_models.py'))
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
def test_all_assets_have_public_release_urls_and_hashes():
    assets=mod.select_assets('all'); assert {a['module'] for a in assets}=={'3.1.1','3.1.2','3.1.3'}; assert all(a['sha256'] and len(a['sha256'])==64 for a in assets); assert all(mod.asset_url(a).startswith('https://github.com/Maester-raven/Fashion/releases/download/') for a in assets)
def test_312_assets_are_five_release_packages():
    assets=mod.select_assets('3.1.2'); assert len(assets)==5 and all(a['extract'] for a in assets)
