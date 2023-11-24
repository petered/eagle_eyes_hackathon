import dataclasses
import json
import os
import shutil
from dataclasses import dataclass
from functools import cached_property
from typing import Tuple, Optional, TypedDict, Dict, List, Iterator, Callable, Sequence

import cv2

from artemis.general.custom_types import BGRImageArray
from artemis.image_processing.image_builder import ImageBuilder
from artemis.image_processing.image_utils import BoundingBox, BGRColors
from dataclasses_serialization.json import JSONSerializer
from eagle_eyes.demos.demo_profile_tflite_model import hold_tempdir
from video_scanner.app_utils.utils import make_backup_file_copy

DEFAULT_DATASET_FOLDER = os.path.expanduser(os.path.join('~', 'Downloads', 'eagle_eyes_hackathon_dataset'))


def get_default_dataset_folder() -> str:
    return DEFAULT_DATASET_FOLDER


def set_default_dataset_folder(folder: str):
    global DEFAULT_DATASET_FOLDER
    DEFAULT_DATASET_FOLDER = folder


@dataclass
class SourceDataDict(TypedDict):
    """ Data that can be attached to an annotation to identify the source """
    file_path: str
    frame_ix: int


@dataclass
class Annotation:
    ijhw_box: Tuple[int, int, int, int]  # (i, j, h, w) box, where i, j are the row, column of the center pixel, and h, w are the height and width of the box
    label: str = ''  # Optionally a label
    value: int = 1  # 1 = true-positive, -1 = true-negative, 0 = neutral (don't care)
    description: str = ''  # A description of the annotation
    tags: Tuple[str, ...] = ()  # Optional - identifiers for the annotation
    source_data: Optional[SourceDataDict] = None  # Optional - data that can be attached to an annotation to identify the source


@dataclass
class AnnotatedImage:
    image: BGRImageArray # The image
    annotations: List[Annotation]  # A sequence of annotations (because one image can have multiple annotations)

    def render(self, thickness=3) -> BGRImageArray:
        builder = ImageBuilder.from_image(self.image)
        # Show existing boxes
        for annotation in (self.annotations or ()):
            i, j = annotation.ijhw_box[:2]
            builder.draw_box(BoundingBox.from_ijhw(*annotation.ijhw_box, label=annotation.label),
                             colour=self.image[i, j],
                             secondary_colour=BGRColors.BLACK,
                             show_score_in_label=False,
                             thickness=thickness,
                             )
        return builder.get_image()


@dataclass
class AnnotatedImageSource:
    """ Contains data needed to load an annotated image """
    source_path: str  # Path to the image-file
    annotations: List[Annotation]  # A sequence of annotations (because one image can have multiple annotations)
    original_source_identifier: Optional[str] = None  # Optional - an identifier for the original source of the image


@dataclass
class Case:
    """ A collection of images involved in a particular 'case' - ie a scenerio, flight, or location """
    name: str
    images: List[AnnotatedImageSource]  # Pairs of (image_path, annotations)
    description: str = ''


@dataclass
class AnnotatedImageInfo:
    """ Contains all data pertaining to an annotated image """
    case_name: str
    item_ix: int
    source_path: str
    annotated_image: AnnotatedImage

    def get_datapoint_name(self) -> str:
        return f'{self.case_name}-{self.item_ix}'


@dataclass
class AnnotatedImageDataLoader:

    case_dict: Dict[str, Case]  # A mapping from case name to a sequence of (image_path, annotations) tuples
    root_folder: str   # The root folder where all images are stored

    @classmethod
    def _dataset_path(cls, root_folder: str):
        return os.path.join(root_folder, 'dataset.json')

    def filter(self, case_filter: Callable[[str], bool]) -> 'AnnotatedImageDataLoader':
        """ Return a filtered version of this dataset """
        if case_filter is None:
            return self
        case_dict = self.case_dict
        if case_filter is not None:
            case_dict = {case_name: case for case_name, case in case_dict.items() if case_filter(case_name)}
        return AnnotatedImageDataLoader(root_folder=self.root_folder, case_dict=case_dict)

    def get_mini_version(self, n_cases: int = 3) -> 'AnnotatedImageDataLoader':
        """ Return a mini version of this dataset """
        cases = list(self.case_dict.keys())[:n_cases]
        return self.filter(lambda case_name: case_name in cases)

    @classmethod
    def from_folder(cls, root_folder: Optional[str] = None, create_if_not_exists: bool = False) -> 'AnnotatedImageDataLoader':
        root_folder = root_folder or get_default_dataset_folder()
        root_folder = os.path.expanduser(root_folder)
        if not os.path.exists(root_folder):
            if create_if_not_exists:
                os.makedirs(root_folder)
            else:
                raise FileNotFoundError(f"Dataset Folder {root_folder} does not exist.  You'll need to download and extract the dataset first.")
        dataset_path = cls._dataset_path(root_folder)
        if os.path.exists(dataset_path):
            with open(dataset_path, 'r') as json_file:
                json_data = json.load(json_file)
        else:
            json_data = {}
        cases = JSONSerializer.deserialize(Dict[str, Case], json_data)
        return cls(root_folder=root_folder, case_dict=cases)

    @classmethod
    def from_merged_datasets(cls, root_folder: str, datasets: Sequence['AnnotatedImageDataLoader']) -> 'AnnotatedImageDataLoader':
        """ Merge multiple datasets together """

        with hold_tempdir() as tempdir:
            builder = AnnotatedImageDatasetBuilder.from_folder(root_folder=tempdir)
            case_dict = {}

            for dataset in datasets:
                for case_name, index, annotated_image_src in dataset.iter_case_index_annotated_image_sources():
                    annotated_image = dataset.lookup_annotated_image(case_name, index)
                    builder.add_annotations(
                        case=case_name,
                        annotated_image=annotated_image,
                        source_identifier=annotated_image_src.original_source_identifier,
                    )
                    print(f"Added {len(annotated_image.annotations)} annotations for {case_name}")
            builder.save()

            # Move all contents of tempdir to root_folder
            if os.path.exists(root_folder):
                make_backup_file_copy(root_folder)
                shutil.rmtree(root_folder, ignore_errors=True)
            shutil.move(tempdir, root_folder)



        # Now merge both /images subfolders

        return cls(root_folder=root_folder, case_dict=case_dict)


    @cached_property
    def index_to_case_and_item(self):
        return [(case_name, item_ix) for case_name, case in self.case_dict.items() for item_ix in range(len(case.images))]

    def __len__(self) -> int:
        return sum(len(case.images) for case in self.case_dict.values())

    def __iter__(self) -> Iterator[AnnotatedImage]:
        for case_name, item_ix, annotated_image_source in self.iter_case_index_annotated_image_sources():
            yield self.lookup_annotated_image(case_name, item_ix)

    def __getitem__(self, index: int) -> AnnotatedImage:
        case_name, item_ix = self.index_to_case_and_item[index]
        return self.lookup_annotated_image(case_name, item_ix)

    def iter_annotated_image_infos(self) -> Iterator[AnnotatedImageInfo]:
        for case_name, item_ix, annotated_image_source in self.iter_case_index_annotated_image_sources():
            annotated_image = self.lookup_annotated_image(case_name, item_ix)
            yield AnnotatedImageInfo(
                case_name=case_name,
                item_ix=item_ix,
                source_path=annotated_image_source.source_path,
                annotated_image=annotated_image
            )

    def iter_case_index_annotated_image_sources(self) -> Iterator[Tuple[str, int, AnnotatedImageSource]]:
        for case_name, case in self.case_dict.items():
            for item_ix in range(len(case.images)):
                yield case_name, item_ix, self.case_dict[case_name].images[item_ix]

    def lookup_annotated_image(self, case_name: str, item_ix: int) -> AnnotatedImage:
        try:
            case = self.case_dict[case_name]
        except KeyError:
            raise KeyError(f"Case {case_name} not found in dataset.  Available cases are {list(self.case_dict.keys())}")
        annotated_image_source = case.images[item_ix]
        full_path = os.path.join(self.root_folder, annotated_image_source.source_path)
        image = cv2.imread(full_path)
        assert image is not None, f"Could not read image at path {full_path}"
        return AnnotatedImage(image=image, annotations=annotated_image_source.annotations)


class AnnotatedImageDatasetBuilder(AnnotatedImageDataLoader):

    def add_annotations(self, case: str, annotated_image: AnnotatedImage, source_identifier: Optional[str] = None):
        """ Add annotations to an image """

        if case not in self.case_dict:
            self.case_dict[case] = Case(name=case, images=[])

        source_path = f'{self.root_folder}/images/{case}-{len(self.case_dict[case].images)}.png'
        os.makedirs(os.path.dirname(source_path), exist_ok=True)
        cv2.imwrite(source_path, annotated_image.image)
        print(f"Saved image to {source_path}")
        relative_path = os.path.relpath(source_path, self.root_folder)
        new_annotation_source = AnnotatedImageSource(source_path=relative_path, annotations=annotated_image.annotations, original_source_identifier=source_identifier)
        self.case_dict[case] = dataclasses.replace(
            self.case_dict[case],
            images=list(self.case_dict[case].images) + [new_annotation_source]
        )

    def save(self):
        json_obj = JSONSerializer.serialize(self.case_dict)
        dataset_path = self._dataset_path(self.root_folder)
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        with open(dataset_path, 'w') as json_file:
            json.dump(json_obj, json_file, indent=2)
