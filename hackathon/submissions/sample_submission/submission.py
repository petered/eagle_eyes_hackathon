
"""
Eagle Eyes Hackathon Template file

In the end, you only need to produce two things:
-
"""
from typing import Sequence, Optional, Iterator

import numpy as np
from dataclasses import dataclass

from hackathon.data_utils.data_loading import AnnotatedImageDataLoader, BGRImageArray, DEFAULT_DATASET_FOLDER
from hackathon.model_utils.interfaces import IDetectionModel, IDetectorLoader, Detection
import tensorflow as tf

from hackathon.model_utils.prebuilt_models import load_v2p5_model
from hackathon.model_utils.scoring_utils import evaluate_models_on_dataset
from hackathon.submissions.sample_submission.submission_utils import tf_holy_box_blur, tf_local_maxima, tf_round_to_odd, compute_mahalonabis_dist_sq


@dataclass
class MyModel(IDetectionModel):
    """
    This is the Reed-Xiaoli detector.
    It takes the Mahalonabis distance of each pixel with respect to the local pixel distribution.
    Where the local pixel distribution is defined by a local mean and a global covariance matrix.
    """
    inner_box_width: int = 10  # Inner width of the dohnut for background subtraction
    outer_box_width: int = 30  # Outer width of the dohnut for background subtraction
    relative_box_size: float = 0.05  # Proportion of image width for local maxima detection
    n_detections: int = 3  # Number of detections to return

    def get_name(self) -> str:
        """ Return the name of this detector """
        return "RX Detector"

    def detect(self, image: BGRImageArray) -> Sequence[Detection]:
        """ Detect objects in an image.  Return a list of detections in this frame."""
        tf_image = tf.constant(image)
        image_float = tf.image.convert_image_dtype(tf_image, tf.float32)
        background_image = tf_holy_box_blur(image_float, inner_box_width=self.inner_box_width, outer_box_width=self.outer_box_width)
        background_subtracted_image = image_float - background_image
        heatmap = tf.reshape(compute_mahalonabis_dist_sq(tf.reshape(background_subtracted_image, (-1, 3))), tf.shape(tf_image)[:2])
        width = tf_round_to_odd(self.relative_box_size * tf.cast(tf.shape(heatmap)[1], tf.float32))
        local_maxima_ij, maxima = tf_local_maxima(heatmap, width=width)
        sorting_ixs = tf.argsort(maxima, direction='DESCENDING')[:self.n_detections]
        local_maxima_ij = tf.gather(local_maxima_ij, sorting_ixs)
        maxima = tf.gather(maxima, sorting_ixs)

        return [Detection(
            ijhw_box=(i, j, 50, 50),
            score=score,
        ) for (i, j), score in zip(local_maxima_ij.numpy(), maxima.numpy())]


@dataclass
class MyModelLoader(IDetectorLoader):

    path: Optional[str] = None

    def load_detector(self) -> MyModel:
        """ Load a detector given a path to a file or folder in which it is stored """
        # TODO: Fill in whatever you need to load your model here.
        return MyModel()  # No need to deal with path since nothing is being trained and saved.



@dataclass
class MyModelTrainer:
    n_epochs: int

    def iter_train_and_save_model(self, loader: AnnotatedImageDataLoader) -> Iterator[str]:
        """ Train the model, periodically yielding """
        for e in range(self.n_epochs):
            frame: AnnotatedFrame = loader.load(np.random.randint(len(loader)))
            raise NotImplementedError('TODO: Fill in training code here.')
            path = f"my_model_epoch_{e}.pth"
            yield path


if __name__ == '__main__':
    evaluate_models_on_dataset(
        detectors={
            'My Model': MyModelLoader().load_detector(),
            'V2.5': load_v2p5_model(),
        },
        data_loader=AnnotatedImageDataLoader.from_folder(),
        show_debug_images=True
    )
