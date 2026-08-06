FORMAL_SUPPORTED_FAMILIES = ['garment_instance','collar','sleeve','neckline','pocket']
FAMILY_TO_TASKS = {
    'garment_instance': ['garment_instance__closure_opening_design','garment_instance__garment_length','garment_instance__waistline_design'],
    'collar': ['collar__collar_design'],
    'sleeve': ['sleeve__sleeve_design','sleeve__sleeve_length'],
    'neckline': ['neckline'],
    'pocket': ['pocket'],
}
LEGACY_TASK_TO_C2_GROUP = {'neckline': 'neckline__part_design_neckline', 'pocket': 'pocket__style_named_garment'}
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]
DINO_MEAN255 = [123, 116, 103]
INPUT_SIZE = 518
