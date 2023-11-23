from eagle_eyes.demos.demo_profile_tflite_model import hold_tempdir
from hackathon.data_utils.data_loading import AnnotatedImageDatasetBuilder, Annotation, AnnotatedImage
import numpy as np


def test_build_and_load_dataset():

    with hold_tempdir() as fdir:
        builder = AnnotatedImageDatasetBuilder.from_folder(root_folder=fdir)
        ai1 = AnnotatedImage(
                image=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
                annotations=[Annotation(ijhw_box=(12, 16, 10, 10), label='label1', value=1)]
            )
        builder.add_annotations(case='case1', annotated_image=ai1)
        ai2 = AnnotatedImage(
                image=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
                annotations=[Annotation(ijhw_box=(0, 0, 10, 10), label='label1', value=1), Annotation(ijhw_box=(20, 30, 10, 10), label='label2', value=2)]
            )
        builder.add_annotations(case='case1', annotated_image=ai2)
        ai3 = AnnotatedImage(
                image=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
                annotations=[Annotation(ijhw_box=(0, 0, 10, 10), label='label1', value=1)]
            )
        builder.add_annotations(case='case2', annotated_image=ai3)
        builder.save()

        loader = AnnotatedImageDatasetBuilder.from_folder(root_folder=fdir)
        assert len(loader) == 3
        assert loader.lookup_annotated_image(case_name='case1', item_ix=0).annotations == ai1.annotations
        assert np.array_equal(loader.lookup_annotated_image(case_name='case1', item_ix=0).image, ai1.image)
        assert loader.lookup_annotated_image(case_name='case1', item_ix=1).annotations == ai2.annotations
        assert np.array_equal(loader.lookup_annotated_image(case_name='case1', item_ix=1).image, ai2.image)
        assert loader.lookup_annotated_image(case_name='case2', item_ix=0).annotations == ai3.annotations


if __name__ == "__main__":
    test_build_and_load_dataset()




