#!/usr/bin/env bash
set -euo pipefail
python -m fashion313_runtime.cli --image examples/example.jpg --target-mask examples/target_mask.png --parent-mask examples/parent_mask.png --attribute-checkpoint models/attribute_model.pth --region-family-checkpoint models/region_family_model.pth --output examples/result.json
