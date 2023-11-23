from hackathon.model_utils.interfaces import IDetectionModel
import numpy as np


class SimpleMahalonablisModel(IDetectionModel):

    def __init__(self, mean: np.ndarray, covariance: np.ndarray, threshold: float):
        self.mean = mean
        self.covariance = covariance
        self.threshold = threshold

    def predict(self, image: np.ndarray) -> List[Detection]:
        """ Predict detections on an image.  """

