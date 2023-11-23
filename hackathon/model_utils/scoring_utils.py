import time
from typing import Sequence, Tuple, Mapping, Optional, Dict, List
import numpy as np
from dataclasses import dataclass, field
from artemis.general.custom_types import Array
from artemis.plotting.cv2_plotting import just_show
from hackathon.data_utils.data_loading import AnnotatedImageDataLoader, AnnotatedImageInfo, Annotation
from hackathon.model_utils.display_utils import get_auc_result_tables
from hackathon.model_utils.interfaces import Detection



@dataclass
class FrameDetectionResult:
    case_name: str
    in_case_index: int
    image_path: str
    datapoint_name: str
    detections: Mapping[str, Sequence[Detection]]
    annotations: Sequence[Annotation]
    compute_times: Mapping[str, float]

    def get_prediction_results(self) -> Mapping[str, 'PredictionResult']:
        """ Get the prediction result for this frame.  """
        return {name: PredictionResult(
            prediction_scores=[d.score for d in detections],
            are_predictions_correct=are_predictions_in_boxes(
                predictions_ij=[d.ijhw_box[:2] for d in detections],
                ground_truths_ijhw=[a.ijhw_box for a in self.annotations]
            ),
            n_ground_truths=len(self.annotations)
        ) for name, detections in self.detections.items()}



@dataclass
class DatasetDetectionResult:
    per_frame_results: List[FrameDetectionResult] = field(default_factory=dict)

    def get_pr_auc(self, case_filter: Optional[str] = None) -> Mapping[str, float]:
        """ Return the PrAUC score for this result.  """
        results = self.per_frame_results if case_filter is None else [r for r in self.per_frame_results if r.case_name == case_filter]
        per_frame_prediction_results = [r.get_prediction_results() for r in results]
        return {predictor_name: PredictionResult.from_joined([r[predictor_name] for r in per_frame_prediction_results]).get_pr_auc()
                for predictor_name in per_frame_prediction_results[0].keys()}

    def get_score_summary(self, show_debug_plots: bool = False) -> str:
        """ Return a string summarizing the score.  """

        tables = get_auc_result_tables(self.per_frame_results, debug=show_debug_plots)
        return tables.get_tables_as_string()


def compute_detection_results(
        detectors: Mapping[str, 'IDetectionModel'],
        loader: AnnotatedImageDataLoader,
        show_debug_images: bool = False,
    ) -> DatasetDetectionResult:
    """ Compute the PrAUC score for a detector on a dataset.  """
    per_frame_results: List[FrameDetectionResult] = []
    for i, annotated_image_info in enumerate(loader.iter_annotated_image_infos()):
        print(f"Computing detections for image {i+1}/{len(loader)}: {annotated_image_info.get_datapoint_name()}")
        detections: Dict[str, Sequence[Detection]] = {}
        annotated_image_info: AnnotatedImageInfo
        compute_times: Dict[str, float] = {}
        for name, detector in detectors.items():
            t_start = time.monotonic()
            detections[name] = detector.detect(annotated_image_info.annotated_image.image)
            compute_times[name] = time.monotonic() - t_start

        per_frame_result = FrameDetectionResult(
            image_path=annotated_image_info.source_path,
            datapoint_name=annotated_image_info.get_datapoint_name(),
            case_name=annotated_image_info.case_name,
            in_case_index=annotated_image_info.item_ix,
            detections=detections,
            annotations=annotated_image_info.annotated_image.annotations,
            compute_times=compute_times
        )

        if show_debug_images:
            from hackathon.ui_utils.visualization_utils import render_per_frame_result
            just_show(render_per_frame_result(per_frame_result, loader))

        per_frame_results.append(per_frame_result)
    return DatasetDetectionResult(per_frame_results=per_frame_results)




@dataclass
class PredictionResult:
    """ This represents the result of a set of predictions over a sequence (either a video or a full dataset) """
    prediction_scores: Sequence[float]
    are_predictions_correct: Sequence[bool]  # For each prediction, is it in a box?
    n_ground_truths: int

    @classmethod
    def from_joined(cls, box_prediction_results: Sequence['PredictionResult']) -> 'PredictionResult':
        return PredictionResult(
            prediction_scores=np.concatenate([p.prediction_scores for p in box_prediction_results]),
            are_predictions_correct=np.concatenate([np.asarray(p.are_predictions_correct, dtype=bool) for p in box_prediction_results], dtype=bool),
            n_ground_truths=sum(p.n_ground_truths for p in box_prediction_results)
        )

    def get_n_true_positives(self, threshold: float) -> int:
        return int(np.sum(np.asarray(self.are_predictions_correct)[np.asarray(self.prediction_scores) >= threshold]))

    def get_n_false_negatives_imperfect(self, threshold: float) -> int:
        """
        Note: This is not quite perfect because it doesn't account for the fact that multiple predictions can be in the same box.
        """
        return self.n_ground_truths - self.get_n_true_positives(threshold)

    def get_precision_recall_by_threshold(self, include_superthhreshold: bool = False, use_nan_for_undefined: bool = False,
                                          ) -> Tuple[Array['N', float], Array['N', float], Array['N-1', float]]:
        """
        Think, ok...

            We take each threshold from the lowest-prediction to a little over the highest prediction.
            We then count the number of true positives, false positives, and false negatives at that threshold.
            We then compute the precision and recall at that threshold.
        """

        # Sort prediction scores and corresponding ground truth values in descending order
        sorted_indices = np.argsort(self.prediction_scores)[::-1]
        sorted_scores = np.array(self.prediction_scores)[sorted_indices]
        sorted_ground_truths = np.array(self.are_predictions_correct)[sorted_indices]

        # Compute cumulative true positives, false positives, and false negatives
        tp = np.cumsum(sorted_ground_truths)
        fp = np.cumsum(~sorted_ground_truths)
        fn = self.n_ground_truths - tp

        # Compute precision and recall
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        if include_superthhreshold:
            precision = np.concatenate([[np.nan if use_nan_for_undefined else 1.], precision])   # Precision is undefined (0/0) at infinite threshold, so define it to 1.
            recall = np.concatenate([[0.], recall])  # Recall is 0 at infinite threshold
            sorted_scores = np.concatenate([[np.inf], sorted_scores])

        return precision, recall, sorted_scores

    def get_pr_auc(self) -> float:
        precision, recall, _ = self.get_precision_recall_by_threshold(include_superthhreshold=True)
        return np.trapz(precision, recall)

    def get_precision_recall_at_threshold(self, threshold: float) -> Tuple[float, float]:
        precision, recall, thresholds = self.get_precision_recall_by_threshold(include_superthhreshold=True, use_nan_for_undefined=True)
        return precision[thresholds>=threshold][-1], recall[thresholds>=threshold][-1]

    def get_f1_score_at_threshold(self, threshold: float) -> float:
        precision, recall = self.get_precision_recall_at_threshold(threshold)
        return f1_score_from_precision_recall(precision, recall)


    # def get_confusion_values_at_threshold(self, threshold: float) -> Tuple[int, int, int]:
    #     # n_true_positives = int(self.get_precision_recall_at_threshold(threshold)[0]*self.n_ground_truths)
    #     n_true_positives = sum(self.are_predictions_correct[self.prediction_scores >= threshold])
    #     n_false_positives = len(self.prediction_scores) - n_true_positives
    #     n_false_negatives = self.n_ground_truths - n_true_positives
    #     return n_true_positives, n_false_positives, n_false_negatives


def f1_score_from_precision_recall(precision: float, recall: float) -> float:
    there_are_no_detections = np.isnan(precision)
    there_ara_no_targets = np.isnan(recall)
    if there_are_no_detections and there_ara_no_targets:
        return 1.
    elif there_are_no_detections or there_ara_no_targets:
        return 0.
    elif precision == 0 and recall == 0:
        return 0.
    else:
        return 2 * precision * recall / (precision + recall)


def does_box_contain_point(box_ijhw: Tuple[int, int, int, int], point_ij: Tuple[int, int]) -> bool:
    """ Return whether a box contains a point.  """
    i, j, h, w = box_ijhw
    pi, pj = point_ij
    return i - h // 2 <= pi <= i + h // 2 and j - w // 2 <= pj <= j + w // 2


def are_predictions_in_boxes(predictions_ij: Sequence[Tuple[int, int]], ground_truths_ijhw: Sequence[Tuple[int, int, int, int]]
                             ) -> Sequence[bool]:
    """ Return a sequence of booleans, indicating whether each prediction is in a box.  """
    return [any(does_box_contain_point(gt_box, p_ij) for gt_box in ground_truths_ijhw) for p_ij in predictions_ij]


def evaluate_models_on_dataset(
        detectors: Mapping[str, 'IDetectionModel'],
        data_loader: 'AnnotatedImageDataLoader',
        show_debug_images: bool = False,
    ) -> DatasetDetectionResult:
    """ Evaluate a set of models on a dataset.  """
    result = compute_detection_results(
        detectors=detectors,
        loader=data_loader,
        show_debug_images=show_debug_images
    )
    pr_auc_scores = result.get_pr_auc()
    print(result.get_score_summary())

    print(f"---\nPR-AUC scores:")
    for name, score in pr_auc_scores.items():
        print(f"  {name}: {score}")
    print(f"Winner: {max(pr_auc_scores, key=lambda k: pr_auc_scores[k])}")
    return result
