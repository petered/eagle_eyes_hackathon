import pickle

from hackathon.data_utils.data_loading import AnnotatedImageDataLoader, get_default_dataset_folder
from hackathon.model_utils.prebuilt_models import load_v0_model, load_v1_model, load_v2p5_model
from hackathon.model_utils.scoring_utils import evaluate_models_on_dataset


def test_model_scoring_comparison_and_result_io():
    # Run all the models against each other on a mini dataset
    result = evaluate_models_on_dataset(
        detectors={
            'V0': load_v0_model(),
            'V1': load_v1_model(),
            'V2.5': load_v2p5_model(),
        },
        data_loader=AnnotatedImageDataLoader.from_folder(get_default_dataset_folder()).get_mini_version()
    )

    # Check that we can save and load the result, and that it's the same
    result_ser = pickle.dumps(result)
    result_again = pickle.loads(result_ser)
    assert result == result_again


if __name__ == '__main__':
    test_model_scoring_comparison_and_result_io()
