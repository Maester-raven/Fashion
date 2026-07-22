from enum import Enum

class Status(str, Enum):
    SUCCESS = "success"
    EMPTY = "empty"
    PRESENT_BUT_NO_CANDIDATE = "present_but_no_candidate"
    SUCCESS_WITH_COARSE_MASK_FALLBACK = "success_with_coarse_mask_fallback"
    INVALID_INPUT = "invalid_input"
    PARENT_IMAGE_MISSING = "parent_image_missing"
    EMPTY_QUERY = "empty_query"
    PRESENCE_RUNTIME_FAILURE = "presence_runtime_failure"
    SELECTOR_RUNTIME_FAILURE = "selector_runtime_failure"
    CANDIDATE_BBOX_INVALID = "candidate_bbox_invalid"
    SAM_RUNTIME_FAILURE = "sam_runtime_failure"
    SAM_EMPTY_MASK = "sam_empty_mask"
    SAM_DECODE_FAILURE = "sam_decode_failure"
    OUTPUT_SERIALIZATION_FAILURE = "output_serialization_failure"
