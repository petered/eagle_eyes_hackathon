from abc import ABCMeta
from abc import ABCMeta
from dataclasses import dataclass
from typing import Tuple, Sequence, Mapping

from artemis.general.custom_types import BGRImageArray
from hackathon.data_utils.data_loading import Annotation


@dataclass
class Detection:
    ijhw_box: Tuple[int, int, int, int]  # A (row_index, column_index, height, width) bounding box.  Values in [0, 1] : Relative to total height/width of the image-
    score: float
    label: str = ''
    description: str = ''






# ---- Interfaces for what contenstant must produce ----


class IDetectionModel(metaclass=ABCMeta):
    """
    Interface for a detection model
    """

    def get_name(self) -> str:
        """ Return the name of this detector """
        raise NotImplementedError()

    def detect(self, image: BGRImageArray) -> Sequence[Detection]:
        """ Detect objects in an image.  Return a list of detections."""

class IDetectorLoader(metaclass=ABCMeta):

    def load_detector(self) -> IDetectionModel:
        """ Load a detector given a path to a file or folder in which it is stored """


