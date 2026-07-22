import json

LIMITATIONS = {
    "returns_all_instances": False,
    "supports_spatial_constraints": False,
    "supports_relation_constraints": False,
}

def validate_result(result):
    assert isinstance(result, dict)
    assert "query_text" in result
    assert "presence" in result
    assert "status" in result
    assert isinstance(result.get("instances"), list)
    assert len(result["instances"]) <= 1
    for inst in result["instances"]:
        assert "bbox" in inst and len(inst["bbox"]) == 4
        assert "mask_rle" in inst
        assert "size" in inst["mask_rle"] and "counts" in inst["mask_rle"]
    json.dumps(result, ensure_ascii=False)
    return True
