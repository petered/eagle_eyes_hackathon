import shutil
from typing import NewType, Tuple, TypedDict, Optional, Mapping, Sequence, Callable
import tensorflow as tf
import numpy as np
import os

from dataclasses import dataclass

from artemis.general.custom_types import BGRImageArray
from artemis.general.utils_for_testing import hold_tempdir, hold_tempfile
from hackathon.model_utils.interfaces import Detection, IDetectionModel

TensorIJCoords = NewType('TensorIJCoords', tf.Tensor)
TensorLTRBBox = NewType('TensorLTRBBox', tf.Tensor)
TensorImage = NewType('TensorImage', tf.Tensor)  # A (height, width, n_colors) uint8 image
TensorFloatImage = NewType('TensorFloatImage', tf.Tensor)
TensorPixelFloatArray = NewType('TensorPixelArray', tf.Tensor)  # A (N, n_colors) array of flattened pixels
TensorFloatVector = NewType('TensorFloatVector', tf.Tensor)  # Vector of floats
TensorImageCoords = NewType('TensorFloatImage', TensorFloatImage)
TensorHeatmap = NewType('TensorHeatmap', TensorFloatImage)  # A (height, width) heatmap
TensorMaskVector = NewType('TensorMaskVector', tf.Tensor)  # A boolean vector
TensorMaskImage = NewType('TensorMaskImage', tf.Tensor)  # A (height, width) boolean mask
TensorInvCovMat = NewType('TensorCovMat', tf.Tensor)  # A (n_colors x n_colors) covariance matrix,
TensorColor = NewType('TensorColor', tf.Tensor)  # A (n_colors, ) vector representing a color
TensorIndexImage = NewType('TensorColor', tf.Tensor)  # A (height, width) array of indices
TensorIndexVector = NewType('TensorIndexVector', tf.Tensor)  # A vector of indices
TensorLTRBBoxes = NewType('TensorLTRBBoxes', tf.Tensor)  # An array of int32 boxes, specified by (Left, Right, Top, Bottom) pixel
TensorIJHWBoxes = NewType('TensorIJHWBoxes', tf.Tensor)  # An array of int32 boxes, specified by (y-center (from top down), x_center, height, width) pixel
TensorInt = NewType('TensorInt', tf.Tensor)  # Just an integer
TensorDistanceMat = NewType('TensorDistanceMat', tf.Tensor)
TensorFloat = NewType('TensorFloat', tf.Tensor)
TensorInt = NewType('TensorInt', tf.Tensor)
TensorLabelVector = NewType("TensorLabelVector", tf.Tensor)
SortedPointScoreTuple = Tuple[TensorIJCoords, TensorFloatVector]
TensorFloatMatrix = NewType("TensorMatrix", tf.Tensor)
TensorBoolMatrix = NewType("TensorBoolMatrix", tf.Tensor)
ScoreFloatVector = TensorFloatVector


class BoxInfoDict(TypedDict, total=False):
    ijhw_coords: TensorIJHWBoxes
    ids: TensorIndexVector  # May not be included - use standardize_box_dict to ensure existence
    scores: ScoreFloatVector  # May not be included - use standardize_box_dict to ensure existence
    labels: Optional[TensorLabelVector]  # May not be included - use standardize_box_dict to ensure existence


def tensor_box_detection_dict_to_detections(tensor_box_detections: BoxInfoDict, hackily_hard_code_width: bool = True) -> Sequence[Detection]:
    # tensor_box_detections = standardize_box_dict(tensor_box_detections, fixed_size=None)
    box_ids = tensor_box_detections['ids']
    ijhw_boxes = tensor_box_detections['ijhw_coords'].numpy()
    scores = tensor_box_detections['scores'].numpy()
    labels = tensor_box_detections['labels'].numpy() if 'labels' in tensor_box_detections else np.array([b''] * len(box_ids))
    return [Detection(
        ijhw_box=ijhw_boxes[i],
        score=scores[i],
        label=labels[i].decode('ascii')
    ) for i in range(len(ijhw_boxes))]


def load_signatures_from_standard_model(
        file_path: str,  # Path to save, e.g. "~/Downloads/my_model.eagle.zip"
) -> Mapping[str, Callable]:
    with hold_tempdir() as tempdir:
        saved_model_dir = os.path.expanduser(tempdir)
        # Temporarily copy the file to one with a zip extension so it'll work
        with hold_tempfile(ext=".zip") as temp_file:
            shutil.copy(file_path, temp_file)
            shutil.unpack_archive(temp_file, saved_model_dir, format='zip')
        loaded_model = tf.saved_model.load(saved_model_dir)
        return loaded_model.signatures


@dataclass
class TFPrebuiltModel(IDetectionModel):
    signature_func: Callable[[BGRImageArray], BoxInfoDict]

    @classmethod
    def from_model_file(cls, path_to_model: str) -> 'TFPrebuiltModel':
        assert os.path.exists(path_to_model), f"'{path_to_model}' does not exist."
        # interpreter = tf.lite.Interpreter(path_to_model)
        sig_func = load_signatures_from_standard_model(path_to_model)['image_to_boxes']
        return cls(signature_func=sig_func)

    def detect(self, image: BGRImageArray) -> Sequence[Detection]:
        rgb_image = image[..., ::-1]
        raw_detection_dict = self.signature_func(input_image=rgb_image)
        return tensor_box_detection_dict_to_detections(raw_detection_dict)



