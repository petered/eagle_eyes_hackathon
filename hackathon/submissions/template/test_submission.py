from hackathon.data_utils.data_loading import AnnotatedImageDataLoader
from hackathon.model_utils.scoring_utils import evaluate_models_on_dataset
from hackathon.submissions.template.submission import MyModelLoader


def test_submission_template():
    """ If this test runs without error, your submission is in the correct format."""

    model_loader = MyModelLoader()
    model = model_loader.load_detector()
    evaluate_models_on_dataset(
        detectors={model.get_name(): model},
        data_loader=AnnotatedImageDataLoader.from_folder()
    )


if __name__ == '__main__':
    test_submission_template()

