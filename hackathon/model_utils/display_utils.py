from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from more_itertools import first




@dataclass
class ResultTables:
    raw_result_table: pd.DataFrame  # Shows PR-AUC for each video
    raw_pr_result_table: pd.DataFrame  # Shows precision/recall for each video
    n_predictions_correct_total_dataframe: pd.DataFrame  # Shows "n_predictions/n_correct_predictions/n_ground_truths" for each video
    summary_result_table: pd.DataFrame  # Shows all results aggregated over videos

    def get_tables_as_string(self) -> str:
        return f"Precision/Recall Table:\n{self.raw_pr_result_table.to_string()}\n\n " \
               f"Video Name (n_ground_truths): n_predictions/n_correct_predictions Table:\n{self.n_predictions_correct_total_dataframe.to_string()}\n\n"\
               f"PR-AUC Table:\n{self.raw_result_table.to_string()}\n\n " \
               f"Summary Table:\n{self.summary_result_table.to_string()}"


def get_auc_result_tables(
        results: Sequence['PerFrameResult'],
        # vidname_to_modelname_to_results: Mapping[str, Mapping[str, 'VideoPerformanceMeasure[MultiframeBoxPredictionResults]']],
        threshold = 0.5,  # For metrics using a threshold like precision, recall
        debug: bool = False
    ) -> ResultTables:
    from hackathon.model_utils.scoring_utils import PredictionResult

    # all_case_names = sorted(set(r.case_name for r in results))

    # Inefficient, but we don't care
    # per_frame_results_by_case = {case_name: [r for r in results if r.case_name==case_name] for case_name in all_case_names}

    index_to_model_to_box_pred_result: Sequence[Mapping[str, PredictionResult]] = [r.get_prediction_results() for r in results]

    all_model_names = sorted(set(m for d in index_to_model_to_box_pred_result for m in d))
    all_case_names = set(r.case_name for r in results)
    vid_to_model_to_box_pred_result: Mapping[str, Mapping[str, PredictionResult]] = {
        case_name: {modelname: PredictionResult.from_joined([pr[modelname] for r in results if r.case_name==case_name for pr in [r.get_prediction_results()]])
                    for modelname in all_model_names}
        for case_name in all_case_names
    }


    # vid_to_model_to_box_pred_result: Mapping[str, Mapping[str, PredictionResult]] = {
    #     case_name: {modelname: PredictionResult.from_joined(per_frame_results_by_case[case_name])}
    #     for case_name in all_case_names
    # }
    model_names = sorted(set(m for d in vid_to_model_to_box_pred_result.values() for m in d))

    video_model_to_n_superthreshold_detections = {vidname: {modelname: sum(1 for s in vpm.prediction_scores if s>=threshold)
                                                    for modelname, vpm in vidresults.items()}
                                          for vidname, vidresults in vid_to_model_to_box_pred_result.items()}
    #
    model_name_to_n_superthreshold_detections = {model_name: sum(model_to_n_unique_detections[model_name] for model_to_n_unique_detections in video_model_to_n_superthreshold_detections.values())
                                         for model_name in model_names}

    model_to_box_pred_result: Mapping[str, PredictionResult] = {
        modelname: PredictionResult.from_joined([vid_to_model_to_box_pred_result[vidname][modelname] for vidname in vid_to_model_to_box_pred_result])
        for modelname in model_names
    }

    # This is roughtly the precision when we consider each detection (over the entire video) as a separate prediction
    model_to_n_correct_predictions_per_detection = {model_name: result.get_n_correct_predictions(threshold=threshold)/(model_name_to_n_superthreshold_detections[model_name] or np.NaN)
                                                    for model_name, result in model_to_box_pred_result.items()}

    def get_pr_string(result: PredictionResult) -> str:
        precision, recall = result.get_precision_recall_at_threshold(threshold)
        return f"{precision:.0%}/{recall:.0%}"

    raw_pr_dataframe = pd.DataFrame({vidname: {modelname: get_pr_string(v) for modelname, v in vidresults.items()}
                                  for vidname, vidresults in vid_to_model_to_box_pred_result.items()}).T

    n_predictions_correct_total_dataframe = pd.DataFrame({f"{vidname} ({model_to_result[first(model_names)].n_ground_truths})": {model_name: f"{video_model_to_n_superthreshold_detections[vidname][model_name]}/{model_to_result[model_name].get_n_correct_predictions(threshold=threshold)}" for model_name in model_names}
                                                                    for vidname, model_to_result in vid_to_model_to_box_pred_result.items()}).T

    raw_auc_dataframe = pd.DataFrame({vidname: {modelname: v.get_pr_auc() for modelname, v in vidresults.items()}
                                  for vidname, vidresults in vid_to_model_to_box_pred_result.items()}).T

    def get_award_string(auc: float, all_auc: Sequence[float]) -> str:
        if auc==max(all_auc) and auc>0:
            # now determine if it is the only one with the highest value and if not give it a "tie" emoji
            return "🏆" if len([a for a in all_auc if a==auc])==1 else "👔"
        else:
            return "  "

    # Now, format each entry in raw_auc_dataframe as a string with 3 significant digits and put a 🏆 next to the highest one in each row
    display_auc_dataframe = pd.DataFrame({vidname: {modelname: f"{auc:.3f}"+get_award_string(auc, vidresults.values()) for modelname, auc in vidresults.items()}
                                    for vidname, vidresults in raw_auc_dataframe.T.to_dict().items()}).T
    # The above was wrong - it did highest for each column... try again
    # display_auc_dataframe = pd.DataFrame({vidname: {modelname: f"{auc:.3f}{' 🏆' if auc==max(raw_auc_dataframe.loc[vidname]) else ''}" for modelname, auc in vidresults.items()}
    #                                 for vidname, vidresults in raw_auc_dataframe.to_dict().items()}).T

    # model_name_to_result: Mapping[str, MultiframeBoxPredictionResults] = {modelname: MultiframeBoxPredictionResults.from_joined(
    #     [model_to_measures[modelname].result for vidname, model_to_measures in vidname_to_modelname_to_results.items()]
    # ) for modelname in all_model_names}

    summary_dataframe = pd.DataFrame(columns=raw_auc_dataframe.columns)
    # summary_dataframe = summary_dataframe.append(pd.Series(raw_dataframe.mean(axis=0), name='score mean'))

    # summary_dataframe = summary_dataframe.append(pd.Series(raw_dataframe.std(axis=0), name='score STD'))
    summary_dataframe = summary_dataframe.append(pd.Series(raw_auc_dataframe.rank(axis=1, ascending=False).mean(axis=0), name='rank mean'))

    mean_auc = pd.Series({model_name: np.nanmean([m2r[model_name].get_pr_auc() for m2r in vid_to_model_to_box_pred_result.values()]) for model_name in model_names}, name='mean AUC')
    summary_dataframe = summary_dataframe.append(mean_auc)
    model_name_to_auc_score = {model_name: result.get_pr_auc() for model_name, result in model_to_box_pred_result.items()}
    overall_auc = pd.Series(model_name_to_auc_score, name='overall AUC')
    summary_dataframe = summary_dataframe.append(overall_auc)
    summary_dataframe = summary_dataframe.append(pd.Series(overall_auc.rank(ascending=False), name='AUC-rank'))

    # Add overall precision and recall
    model_name_to_precision_recall = {model_name: result.get_precision_recall_at_threshold(threshold) for model_name, result in model_to_box_pred_result.items()}
    summary_dataframe = summary_dataframe.append(pd.Series({name: p for name, (p, r) in model_name_to_precision_recall.items()}, name='Precision'))
    summary_dataframe = summary_dataframe.append(pd.Series({name: r for name, (p, r) in model_name_to_precision_recall.items()}, name='Recall'))
    summary_dataframe = summary_dataframe.append(pd.Series({name: 2*(p * r)/(p+r) for name, (p, r) in model_name_to_precision_recall.items()}, name='F1'))

    # Add number of super-threshold detections (by unique ID).  This roughtly tells us how many unique things the model reports
    summary_dataframe = summary_dataframe.append(pd.Series({model_name: model_name_to_n_superthreshold_detections[model_name] for model_name in model_names}, name='N detections'))
    summary_dataframe = summary_dataframe.append(pd.Series({model_name: model_to_n_correct_predictions_per_detection[model_name] for model_name in model_names}, name='Video-Precision'))

    # Ok, video-f1 roughly corresponds to what we want in the end
    summary_dataframe = summary_dataframe.append(pd.Series({model_name: 2*(p * model_to_n_correct_predictions_per_detection[model_name])/(p+model_to_n_correct_predictions_per_detection[model_name])
                                                            for model_name, (p, _) in model_name_to_precision_recall.items()}, name='Video-F1'))

    # summary_dataframe = summary_dataframe.append(pd.Series(raw_dataframe.rank(axis=1, ascending=False).std(axis=0), name='  rank STD'))
    # summary_dataframe = summary_dataframe.append(
    #     pd.Series({modelname: 1000 * np.median([modelname_to_results[modelname].median_cycle_time for vidname, modelname_to_results in vid_to_model_to_box_pred_result.items()])
    #                for modelname in raw_auc_dataframe.columns}, name='median cycle ms'))

    if debug:
        winning_model = summary_dataframe.loc['overall AUC'].idxmax()
        string_output = raw_auc_dataframe.to_string() + '\n' + summary_dataframe.to_string()
        print(string_output)
        # model_name_to_result = dict(sorted(list(model_to_box_pred_result.items()), key=lambda nr: nr[1].get_score()))
        # Make a figure and add some vertical space between the subplots
        fig = plt.figure(figsize=(10, 6))

        def onclick(event):
            ind = event.ind
            print(f"Threshold: {thresholds[ind[0]]}")

        fig.canvas.callbacks.connect('pick_event', onclick)
        fig.subplots_adjust(hspace=0.3)
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)
        for i, (model_name, result) in enumerate(model_to_box_pred_result.items()):
            result: PredictionResult
            precision, recall, thresholds = result.get_precision_recall_by_threshold()
            auc = result.get_pr_auc()
            ax1.plot(recall, precision, label=f"{model_name}: {auc:.3f}")
            ax1.set_xlabel("Recall")
            ax1.set_ylabel("Precision")
            ax1.set_title("PR-AUC")

            # Make it so clicking a point along the plot shows the threshold corresponding to that point

            # cid = fig.canvas.mpl_connect('pick_event', onclick)
            # and now make it work...




            ax1.legend()

            # if i==0:  # Just for winning model

            precision, recall, all_thresholds = result.get_precision_recall_by_threshold()
            f1_score = 2 * precision * recall / (precision + recall)

            # all_thresholds, _, weights = result.get_sorted_heats_labels_weights()
            # tp, tn, fp, fn = result.get_descending_tp_tn_fp_fn()
            # precision = tp / (tp + fp)
            # recall = tp / (tp + fn)
            # f1 = 2 * tp / (2 * tp + fp + fn)

            if model_name==winning_model:  # Winner!
                ax2.plot(all_thresholds, precision, color='C0', label='Precision')
                ax2.plot(all_thresholds, recall, color='C1', label='Recall')
                ax2.plot(all_thresholds, f1_score, color='C2', label='F1 Score')
            else:
                ax2.plot(all_thresholds, precision, color='C0', alpha=0.2)
                ax2.plot(all_thresholds, recall, color='C1', alpha=0.2)
                ax2.plot(all_thresholds, f1_score, color='C2', alpha=0.2)
        ax2.set_xlabel("Threshold")
        ax2.set_title(f"Curves for winning model: {winning_model}")
        ax2.legend()
        plt.show()

    return ResultTables(raw_result_table=display_auc_dataframe, summary_result_table=summary_dataframe, raw_pr_result_table=raw_pr_dataframe, n_predictions_correct_total_dataframe=n_predictions_correct_total_dataframe)
