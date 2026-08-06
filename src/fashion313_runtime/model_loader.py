
import hashlib, json, os
from pathlib import Path
import numpy as np
import torch
from .exceptions import ModelLoadError, AssetIntegrityError
from . import model_defs

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def _package_root():
    return Path(__file__).resolve().parents[2]

class LoadedModels:
    def __init__(self, project_root=None, attribute_checkpoint=None, region_family_checkpoint=None, device='cuda', expected_attribute_sha=None, expected_region_family_sha=None):
        self.package_root = _package_root()
        config_candidates = [
            self.package_root / "config",
            self.package_root / "configs" / "3_1_3",
        ]
        self.config_dir = next((p for p in config_candidates if (p / "task_attribute_mapping.json").exists()), config_candidates[0])
        self.attribute_checkpoint = Path(attribute_checkpoint or self.package_root / "models" / "fashion313_attribute_model_v1.pth")
        self.region_family_checkpoint = Path(region_family_checkpoint or self.package_root / "models" / "fashion313_region_family_model_v1.pth")
        self.device = torch.device(device if torch.cuda.is_available() or not str(device).startswith('cuda') else 'cpu')
        if not self.attribute_checkpoint.exists():
            raise ModelLoadError(f'attribute checkpoint not found: {self.attribute_checkpoint}')
        if not self.region_family_checkpoint.exists():
            raise ModelLoadError(f'region-family checkpoint not found: {self.region_family_checkpoint}')
        self.attribute_sha256 = sha256_file(self.attribute_checkpoint)
        self.region_family_sha256 = sha256_file(self.region_family_checkpoint)
        if expected_attribute_sha and self.attribute_sha256 != expected_attribute_sha:
            raise AssetIntegrityError('attribute checkpoint SHA mismatch')
        if expected_region_family_sha and self.region_family_sha256 != expected_region_family_sha:
            raise AssetIntegrityError('region-family checkpoint SHA mismatch')
        mapping = json.loads((self.config_dir / 'task_attribute_mapping.json').read_text())
        _, old_groups, _ = model_defs.load_c2_mapping(self.config_dir)
        attr_model = model_defs.P1Model(old_groups, mapping, dropout=0.1)
        attr_ck = torch.load(self.attribute_checkpoint, map_location='cpu', weights_only=False)
        missing, unexpected = attr_model.load_state_dict(attr_ck['model_state'], strict=True)
        if missing or unexpected:
            raise ModelLoadError(f'attribute strict load failed: {missing} {unexpected}')
        attr_model.eval().to(self.device)
        for p in attr_model.parameters(): p.requires_grad_(False)
        rf_mapping = json.loads((self.config_dir / 'region_family_class_mapping.json').read_text())
        rf_model = model_defs.RegionFamilyHead(768 + 2*len(model_defs.GEOM_NAMES), len(rf_mapping['local_family_to_index']))
        rf_ck = torch.load(self.region_family_checkpoint, map_location='cpu', weights_only=False)
        rmissing, runexpected = rf_model.load_state_dict(rf_ck['model_state'], strict=True)
        if rmissing or runexpected:
            raise ModelLoadError(f'region-family strict load failed: {rmissing} {runexpected}')
        rf_model.eval().to(self.device)
        for p in rf_model.parameters(): p.requires_grad_(False)
        geom_norm = json.loads((self.config_dir / 'region_family_geometry_normalization.json').read_text())
        self.rep = model_defs
        self.rf = model_defs
        self.mapping, self.old_groups, self.rf_mapping = mapping, old_groups, rf_mapping
        self.attr_model, self.rf_model = attr_model, rf_model
        self.geom_mean = np.asarray(geom_norm['mean'], dtype=np.float32)
        self.geom_std = np.asarray(geom_norm['std'], dtype=np.float32)
        self.attribute_loaded_ratio = 1.0
        self.region_family_loaded_ratio = 1.0
