
"""
Eagle Eyes Hackathon Template file for trainable models.

This template can serve as a good starting point for making a trainable mode.


We will test your model loader with the following code:

    model = MyModelLoader().load_detector()
    evaluate_models_on_dataset(
        detectors={model.get_name(): model},
        data_loader=AnnotatedImageDataLoader.from_folder().get_mini_version()
    )

So if you pass test_submission.py you should be good to go.
"""
from dataclasses import dataclass
from typing import Sequence, Optional, Iterator
from hackathon.data_utils.data_loading import AnnotatedImageDataLoader, BGRImageArray, AnnotatedImage
from hackathon.model_utils.interfaces import IDetectionModel, IDetectorLoader, Detection
from hackathon.model_utils.prebuilt_models import load_v2p5_model
from hackathon.model_utils.scoring_utils import evaluate_models_on_dataset
import numpy as np

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


@dataclass
class MyModelTrainer:
    n_epochs: int

    def iter_train_and_save_model(self, loader: AnnotatedImageDataLoader) -> Iterator[str]:
        """ Train the model, periodically yielding """
        for e in range(self.n_epochs):
            shuffled_indices = np.random.permutation(len(loader))
            for i in shuffled_indices:
                frame: AnnotatedImage = loader[i]
                print(f"Training on frame: {i}/{len(loader)} of shape {frame.image.shape}.  TODO: Insert actual training code")

            path = f"my_model_epoch_{e}.pth"
            print(f"Saving model to {path}")
            yield path


def main(n_epochs: int, use_mini_dataset: bool):

    loader = AnnotatedImageDataLoader.from_folder()
    if use_mini_dataset:
        loader = loader.get_mini_version()

    trainer = MyModelTrainer(n_epochs=n_epochs)

    for epoch, save_path in enumerate(trainer.iter_train_and_save_model(loader)):
        print(f"Saved model to {save_path} after epoch {epoch}.  Evaluating...")
        evaluate_models_on_dataset(
            detectors={
                'My Model': MyModelLoader().load_detector(),
                'V2.5': load_v2p5_model(),  # Optionally, you can compare your model to the leading prebuilt model.
            },
            data_loader=AnnotatedImageDataLoader.from_folder(),  # Add ".get_mini_version()" to test on a smaller dataset
            show_debug_images=True
        )


if __name__ == '__main__':
    main(n_epochs=10, use_mini_dataset=True)
