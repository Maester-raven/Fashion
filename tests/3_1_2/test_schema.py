from fashion_3_1_2.schemas import validate_result
r={"query_text":"find pocket","presence":False,"status":"empty","instances":[]}
assert validate_result(r)
