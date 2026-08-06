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


def validate_output(x):
 ins=x['instances']; assert x['instance_count']==len(ins); assert len({i['instance_id'] for i in ins})==len(ins); assert len({i['candidate_id'] for i in ins})==len(ins); assert [i['rank'] for i in ins]==list(range(1,len(ins)+1)); assert all(i['bbox_xyxy_parent'][2]>i['bbox_xyxy_parent'][0] and i['bbox_xyxy_parent'][3]>i['bbox_xyxy_parent'][1] for i in ins); assert all(sum(i['mask_rle']['counts'])==i['mask_rle']['size'][0]*i['mask_rle']['size'][1] for i in ins); return True
