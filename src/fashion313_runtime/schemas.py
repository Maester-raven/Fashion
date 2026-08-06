import json, math

def validate_prediction(obj):
    json.dumps(obj)
    assert obj['status'] == 'ok'
    seen_tasks = set()
    for pred in obj['predictions']:
        assert pred['task_id'] not in seen_tasks
        seen_tasks.add(pred['task_id'])
        ranks = [c['rank'] for c in pred['candidates']]
        assert ranks == list(range(1, len(ranks)+1))
        assert pred['candidate_count'] == len(pred['candidates'])
        ids = set()
        for c in pred['candidates']:
            assert c['attribute_id'] not in ids
            ids.add(c['attribute_id'])
            assert math.isfinite(float(c['confidence']))
    return True
