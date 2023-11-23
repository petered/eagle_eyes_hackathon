



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


class MyModel(IDetectionModel):

    def get_name(self) -> str:
        # TODO: Fill in the name of your model here
        return 'MyModelName'

    def detect(self, image: BGRImageArray) -> Sequence[Detection]:
        """ Detect objects in an image.  Return a list of detections in this frame."""
        raise NotImplementedError()


@dataclass
class MyModelLoader(IDetectorLoader):

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
            frame: AnnotatedFrame = loader.load(np.random.randint(len(loader)))
            raise NotImplementedError('TODO: Fill in training code here.')
            path = f"my_model_epoch_{e}.pth"
            yield path


def evaluate_model_on_dataset():

    # Train
    loader = AnnotatedImageDataLoader.from_folder(DEFAULT_DATASET_FOLDER)
    trainer = MyModelTrainer(n_epochs=5)
    for i, save_path in enumerate(trainer.iter_train_and_save_model(loader)):
        print(f"Saved model to {save_path}")

        # Evaluate
        evaluate_model_on_dataset()


if __name__ == '__main__':
    evaluate_model_on_dataset()
