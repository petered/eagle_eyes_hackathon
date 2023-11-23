from hackathon.data_utils.data_loading import AnnotatedImageDataLoader, DEFAULT_DATASET_FOLDER
from hackathon.model_utils.scoring_utils import evaluate_models_on_dataset
from hackathon.model_utils.tf_model_utils import TFPrebuiltModel
from video_scanner.app_utils.utils import AssetModels


def test_model_scoring_comparison():
    evaluate_models_on_dataset(
        detectors={
            'V0': TFPrebuiltModel.from_model_file(AssetModels.V0_COLOR),
            'V1': TFPrebuiltModel.from_model_file(AssetModels.V1_COLOR),
            'V2.5': TFPrebuiltModel.from_model_file(AssetModels.PHOTO_DEFAULT),
        },
        data_loader = AnnotatedImageDataLoader.from_folder(DEFAULT_DATASET_FOLDER)
    )


if __name__ == '__main__':
    test_model_scoring_comparison()
