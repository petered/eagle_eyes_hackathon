from hackathon.data_utils.data_loading import AnnotatedImage, AnnotatedImageDataLoader
from hackathon.model_utils.interfaces import IDetectionModel
from hackathon.submissions.sample_submission.submission import MyModel
from hackathon.ui_utils.visualization_utils import just_show, render_detections_unto_image


def demo_show_model_results_on_case():

    # model: IDetectionModel = load_v2p5_model()
    model: IDetectionModel = MyModel()

    example_annotated_image: AnnotatedImage = AnnotatedImageDataLoader.from_folder()\
        .lookup_annotated_image(case_name='WALK_DAY_ROCK', item_ix=2)

    detections = model.detect(example_annotated_image.image)

    display_image = render_detections_unto_image(example_annotated_image, detections, title=model.get_name())

    just_show(display_image, hang_time=20)


if __name__ == '__main__':
    demo_show_model_results_on_case()