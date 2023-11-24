from hackathon.data_utils.data_loading import AnnotatedImageDataLoader
from hackathon.model_utils.prebuilt_models import load_v2p5_model
from hackathon.model_utils.scoring_utils import evaluate_models_on_dataset
from hackathon.submissions.sample_submission.submission import MyModel
from hackathon.ui_utils.ui_view_dataset import open_annotation_database_viewer


def demo_view_dataset_with_detections():
    """
    Shows how you can use the dataset-viewer GUI to view detections as well

    Use Left/Right arrow keys to switch between raw/annotated images.
    Zoom with Z/X/C, and Pan with W/A/S/D.
    """
    data_loader = AnnotatedImageDataLoader.from_folder().get_mini_version()
    results = evaluate_models_on_dataset(
        detectors={
            'MyModel': MyModel(),
            'V2.5': load_v2p5_model(),
        },
        data_loader=data_loader,
    )
    open_annotation_database_viewer(results=results, data_loader=data_loader)


if __name__ == '__main__':
    demo_view_dataset_with_detections()
