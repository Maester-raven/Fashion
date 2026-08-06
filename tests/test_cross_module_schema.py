from fashion_system.common.schema_3_1 import parent_bbox_to_full_image, select_highest_confidence
def test_parent_bbox_to_full_image(): assert parent_bbox_to_full_image([1,2,5,6],[10,20,30,40])==[11,22,15,26]
def test_select_highest_confidence(): assert select_highest_confidence([{'score':0.1},{'confidence':0.9}])['confidence']==0.9
