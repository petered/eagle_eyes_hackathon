
"""
Eagle Eyes Hackathon Template file.

This template serves as an example of how you might have a trainiable model.

We will test your model loader with the following code:

    model = MyModelLoader().load_detector()
    evaluate_models_on_dataset(
        detectors={model.get_name(): model},
        data_loader=AnnotatedImageDataLoader.from_folder(SECRET_TEST_DATASET_FOLDER)
    )

So if you pass test_submission.py you should be good to go.
"""
from dataclasses import dataclass
from typing import Sequence, Optional, Iterator
from hackathon.data_utils.data_loading import AnnotatedImageDataLoader, BGRImageArray
from hackathon.model_utils.interfaces import IDetectionModel, IDetectorLoader, Detection
from hackathon.model_utils.prebuilt_models import load_v2p5_model
from hackathon.model_utils.scoring_utils import evaluate_models_on_dataset


class MyModel(IDetectionModel):

    def get_name(self) -> str:
        # TODO: Fill in the name of your model here
        return 'MyModelName'

    def detect(self, image: BGRImageArray) -> Sequence[Detection]:
        """ Detect objects in an image.  Return a list of detections in this frame."""
        raise NotImplementedError(f"TODO: Implement detection for {self.get_name()}")


@dataclass
class MyModelLoader(IDetectorLoader):
    """
    This is the class that we will use to load your model.
    If your model is something that must be loaded from a file (e.g. if it uses parameters
    that are learned from data), you can handle the loading in the load_detector method.
    Otherwise, you can just return an instance of your model.
    """
    path: Optional[str] = None

    def load_detector(self) -> MyModel:
        """ Load a detector given a path to a file or folder in which it is stored """
        # TODO: Fill in whatever you need to load your model here.
        return MyModel()


def main():
    model = MyModelLoader().load_detector()
    evaluate_models_on_dataset(
        detectors={model.get_name(): model},
        data_loader=AnnotatedImageDataLoader.from_folder().get_mini_version()
    )


if __name__ == '__main__':
    main()
