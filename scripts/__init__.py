from enum import Enum

class RenderMode(str, Enum):
    HUMAN = "human"
    RGB_ARRAY = "rgb_array"


class ObservationType(str, Enum):
    GRID = "grid"
    IMAGE = "image"