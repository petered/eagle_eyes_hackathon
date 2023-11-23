from typing import Sequence, Iterator, Optional

from dataclasses import replace

import numpy as np
import cv2
from artemis.image_processing.image_builder import ImageBuilder
from artemis.image_processing.image_utils import BoundingBox, BGRColors
from artemis.plotting.data_conversion import put_list_of_images_in_array, put_data_in_grid, put_data_in_image_grid
from hackathon.data_utils.data_loading import AnnotatedImageDataLoader, BGRImageArray, AnnotatedImage
from hackathon.model_utils.interfaces import Detection
from hackathon.model_utils.scoring_utils import FrameDetectionResult, DatasetDetectionResult


def just_show(image: BGRImageArray, title: Optional[str] = None, hang_time: Optional[float] = None) -> BGRImageArray:
    """ Show an image.  """
    cv2.imshow(title or 'Display', image)
    hang_time_ms = int(1000 * hang_time) if hang_time is not None else 1000000
    cv2.waitKey(hang_time_ms)


def render_detections_unto_image(
        annotated_image: AnnotatedImage,
        detections: Sequence[Detection],
        title: Optional[str] = None,
) -> BGRImageArray:
    """ Draw detections onto an image.  """

    base_image = annotated_image.render()
    builder = ImageBuilder.from_image(base_image, copy=False)
    for d in detections:
        builder.draw_box(BoundingBox.from_ijhw(*d.ijhw_box, score=d.score), colour=annotated_image.image[tuple(d.ijhw_box[:2])], secondary_colour=BGRColors.WHITE, thickness=3, as_circle=True)
    if title is not None:
        builder.draw_corner_text(title, colour=BGRColors.WHITE, shadow_color=BGRColors.BLACK, thickness=2)
    return builder.get_image()


def render_per_frame_result(
        per_frame_result: FrameDetectionResult,
        data_loader: AnnotatedImageDataLoader,
) -> BGRImageArray:
    """ Iterate over images showing the results of a detector on a dataset.  """

    data_loader.lookup_annotated_image(case_name=per_frame_result.case_name, item_ix=per_frame_result.in_case_index)
    annotated_image_info = data_loader.lookup_annotated_image(case_name=per_frame_result.case_name, item_ix=per_frame_result.in_case_index)

    base_image = annotated_image_info.render()

    per_detector_images = [
        render_detections_unto_image(
            annotated_image=replace(annotated_image_info, image=base_image.copy()),
            detections=detections,
            title=detector_name
        ) for detector_name, detections in per_frame_result.detections.items()]

    image_array = put_list_of_images_in_array(per_detector_images)
    return put_data_in_image_grid(image_array, fill_colour=BGRColors.BLACK)


    # for detector_name, detections in per_frame_result.detections.items():
    #     image = base_image.copy()
    #     image = render_detections_unto_image(annotated_image_info, detections, title=detector_name)







