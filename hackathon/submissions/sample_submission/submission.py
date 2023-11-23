"""
Eagle Eyes Hackathon Sample Submission.

This
"""
from dataclasses import dataclass
from typing import Sequence, Optional

import tensorflow as tf

from artemis.image_processing.image_builder import ImageBuilder
from artemis.image_processing.image_utils import heatmap_to_color_image, BoundingBox, BGRColors
from hackathon.data_utils.data_loading import AnnotatedImageDataLoader, BGRImageArray
from hackathon.model_utils.interfaces import IDetectionModel, IDetectorLoader, Detection
from hackathon.model_utils.scoring_utils import evaluate_models_on_dataset
from hackathon.submissions.sample_submission.submission_utils import tf_holy_box_blur, tf_local_maxima, tf_round_to_odd, compute_mahalonabis_dist_sq
from hackathon.ui_utils.tk_utils.tkshow import tkshow


@dataclass
class MyModel(IDetectionModel):
    """
    This is the Local Reed-Xiaoli detector.
    It takes the Mahalonabis distance of each pixel with respect to the "background" distribution.
    Where the "background" distribution for a pixel (i, j) is defined by a mean of the pixels in
    a dohnut-shaped region around (i, j) and the covariance over the entire image.
    """
    inner_box_width: int = 10  # Inner width of the dohnut for background subtraction
    outer_box_width: int = 30  # Outer width of the dohnut for background subtraction
    relative_box_size: float = 0.05  # Proportion of image width for local maxima detection
    n_detections: int = 3  # Number of detections to return
    debug: bool = False

    def get_name(self) -> str:
        """ Return the name of this detector """
        return "Local RX Detector"

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

        if self.debug:
            # Show a display that lets you flick between the image and the heatmap with the local maxima drawn on.
            image_builder = ImageBuilder.from_image(heatmap_to_color_image(heatmap.numpy()))
            for ix, (i, j) in enumerate(local_maxima_ij.numpy()):
                image_builder = image_builder.draw_box(BoundingBox.from_ijhw(i, j, 100, 100, score=maxima.numpy()[ix]), colour=BGRColors.RED, thickness=2, as_circle=True)
            tkshow({'Image': image, 'Heatmap': image_builder.get_image()})

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


def main():
    model = MyModelLoader().load_detector()
    evaluate_models_on_dataset(
        detectors={model.get_name(): model},
        data_loader=AnnotatedImageDataLoader.from_folder().get_mini_version()
    )


if __name__ == '__main__':
    main()
