from hackathon.data_utils.data_loading import AnnotatedImage, AnnotatedImageDataLoader
from hackathon.model_utils.interfaces import IDetectionModel
from hackathon.submissions.sample_submission.submission import MyModel
from hackathon.ui_utils.tk_utils.tkshow import tkshow
from hackathon.ui_utils.visualization_utils import just_show, render_detections_unto_image


def demo_debug_detector():
    """
    Look inside the detector to see how you can use tkshow for debugging
    """
    model: IDetectionModel = MyModel(debug=True)  # Look at the effects of this flag in MyModel
    example_annotated_image: AnnotatedImage = AnnotatedImageDataLoader.from_folder()\
        .lookup_annotated_image(case_name='WALK_DAY_ROCK', item_ix=2)
    model.detect(example_annotated_image.image)


if __name__ == '__main__':
    demo_debug_detector()