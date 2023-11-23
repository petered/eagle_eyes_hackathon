from hackathon.data_utils.data_loading import AnnotatedImageDataLoader
from hackathon.model_utils.scoring_utils import evaluate_models_on_dataset
from hackathon.ui_utils.visualization_utils import render_detections_unto_image
from submission import MyModelLoader
import cv2


def test_submission_template():
    """ If this test runs without error, your submission is in the correct format."""

    model = MyModelLoader().load_detector()
    evaluate_models_on_dataset(
        detectors={model.get_name(): model},
        data_loader=AnnotatedImageDataLoader.from_folder().get_mini_version()
    )


def test_display_submission(show: bool = False):

    model = MyModelLoader().load_detector()
    annotated_image = AnnotatedImageDataLoader.from_folder()[2]
    detections = model.detect(annotated_image.image)
    display_image = render_detections_unto_image(annotated_image, detections, title=model.get_name())

    assert display_image is not None and display_image.shape == annotated_image.image.shape
    if show:
        cv2.imshow('Display', display_image)
        cv2.waitKey(1000)
    print("Display test passed.")


if __name__ == '__main__':
    test_submission_template()
    test_display_submission(show=True)
