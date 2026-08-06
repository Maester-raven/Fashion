from fashion_3_1_2.runtime.constraints import apply_spatial_constraints, constraint_names, parse_spatial_constraints
INSTANCES=[{'instance_id':'a','rank':1,'bbox_xyxy_parent':[0,40,20,80]},{'instance_id':'b','rank':2,'bbox_xyxy_parent':[80,40,100,80]},{'instance_id':'c','rank':3,'bbox_xyxy_parent':[40,0,60,20]},{'instance_id':'d','rank':4,'bbox_xyxy_parent':[40,80,60,100]}]
def test_parse_english_and_chinese():
    assert constraint_names(parse_spatial_constraints('find the leftmost zipper'))==['leftmost']
    assert constraint_names(parse_spatial_constraints('找右侧的口袋'))==['right']
def test_extreme_filters_to_single_candidate():
    assert [x['instance_id'] for x in apply_spatial_constraints(INSTANCES, parse_spatial_constraints('rightmost detail'),100,100)]==['b']
def test_region_filters_subset_without_reordering_semantic_order():
    out=apply_spatial_constraints(INSTANCES, parse_spatial_constraints('upper region'),100,100); assert [x['instance_id'] for x in out]==['c'] and out[0]['rank']==1
