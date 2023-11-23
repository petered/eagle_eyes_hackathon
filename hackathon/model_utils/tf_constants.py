from typing import NewType, Tuple
import tensorflow as tf

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



DEFAULT_ANOMALY_SIZE: int = 5
DEFAULT_ANOMOLY_BOX_FRACTION = 0.05  # If the algorithm expects the anomoly to be contained within a bot, use this proportion of the image width by default.
DEFAULT_SMALL_SURROUND_FACTOR: float = 2
DEFAULT_SURROUND_FACTOR: float = 10.
DEFAULT_LOCAL_MAXIMA_WIDTH: float = 0.2

ANDROID_CAMERA_PREVIEW_SIZE = (640, 480)
DJI_VIDEO_SIZE = (1280, 720)  # (width, height) of video that comes in with the prbious
MAX_VIDEO_SIZE = (4000, 3000)  # (width, height) of video that comes in with the prbious

DEFAULT_COLOUR_SMEAR_WIDTH = 51  # Default width to expand around a point of interest for visibility
DEFAULT_BOXING_DOWNSAMPLE_FACTOR = 4
DEFAULT_THRESHOLD_OVER_MEAN = 200  # When the density exceeds this many times the mean over the image, filter it in.
DEFAULT_RELATIVE_BOX_SIZE = 0.05  # Proportion of image width to filter as "center"
DEFAULT_WARP_WIDTH = 0.08
DEFAULT_BOX_IOU_THRESHOLD = 0.25
DEFAULT_BOX_MEMORY_FRAMES = 5
DEFAULT_BOX_FEATURE_DIM = 3
MAX_SIMULTANEOUS_BOXES = 10


# For auto-thresholding - some reasonable paramters for determining a reasonable threshold
DEFAULT_TOP_MAXIMA = 20  # Number of local maxima to consider when computing threshold
DEFAULT_THRESHOLD_OVER_MEAN_OF_MAXIMA = 1.5  # Make the threshold
MIDRANGE_THRESHOLD_OVER_MEAN_OF_MAXIMA = 2.5  # Make the threshold

INVCOV_COLOR_DEFAULT_EPSILON = 1e-6

MODEL_SAVE_PATH = asset_path('models')

BACKUP_DRONE_FOLDER = '/Volumes/Memorex USB/drone'
EXTERNAL_DRIVE_PATH = '/Volumes/WD_4TB/drone'
