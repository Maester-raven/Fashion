from fashion_3_1_2.coordinate_utils import is_valid_xyxy, clip_xyxy, full_image_bbox_to_parent_crop
assert is_valid_xyxy([1,2,3,4])
assert clip_xyxy([-1,-1,5,5], 4, 4) == [0,0,4,4]
assert full_image_bbox_to_parent_crop([15,25,35,45], [10,20,100,120]) == [5.0,5.0,25.0,25.0]
