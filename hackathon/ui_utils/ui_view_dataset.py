import os
import tkinter as tk
from dataclasses import dataclass, replace
from math import ceil
from tkinter.ttk import Treeview
from typing import Tuple, Optional, Sequence

from artemis.image_processing.image_utils import ImageViewInfo
from hackathon.data_utils.data_loading import AnnotatedImageDataLoader, DEFAULT_DATASET_FOLDER, AnnotatedImageSource, AnnotatedImage, Annotation
from hackathon.model_utils.interfaces import Detection
from hackathon.model_utils.scoring_utils import DatasetDetectionResult
from hackathon.ui_utils.tk_utils.alternate_zoomable_image_view import ZoomableImageFrame
from hackathon.ui_utils.tk_utils.tk_utils import hold_tkinter_root_context
from hackathon.ui_utils.tk_utils.ui_choose_parameters import ui_choose_parameters
from hackathon.ui_utils.tk_utils.ui_utils import populate_frame, get_awaiting_input_image, RespectableButton
from hackathon.ui_utils.visualization_utils import render_detections_unto_image


class FrameAnnotationView(tk.Frame):

    def __init__(self,
                 master: tk.Frame,
                 enable_annotation_editing: bool = True,
                 ):
        super().__init__(master)

        # Make this frame have a column layout with 2 columns, the right one being 3x wider than the left one
        # self.grid_columnconfigure(0, weight=1)
        # self.grid_columnconfigure(1, weight=3)
        # self.grid_rowconfigure(0, weight=1)

        # with populate_frame(tk.Frame(self)) as self._left_panel:
        #     self._left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        #     # self._left_panel.grid(column=0, row=0, sticky=tk.NSEW)
        #     # self._left_panel.pack_propagate(False)
        #     self._annotation_table = Table(self._left_panel)
        #     # self._annotation_table.show()
        #     self._annotation_table.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # with populate_frame(tk.Frame(self)) as self._right_panel:
        #     self._right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            # self._right_panel.grid(column=1, row=0, sticky=tk.NSEW)
        self._label = tk.Label(self, text="")
        self._label.pack(side=tk.TOP, fill=tk.X, expand=False)
        self._image_view = ZoomableImageFrame(
            self,
            # single_click_callback=self._on_click,
            mouse_callback=self._on_click,
        )
        self._image_view.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._last_externally_set_annotations: Optional[Sequence[Annotation]] = None

        self._mouse_down_location: Optional[Tuple[int, int]] = None
        self._current_frame_annotation: Optional[AnnotatedImage] = None
        self._current_frame_detections: Optional[Sequence[Detection]] = None

        self._enable_annotation_editing = self._enable_annotation_editing_initially = enable_annotation_editing

        # Delete button to remove last annotation with Delete key
        self.master.bind("<Delete>", lambda event: self._remove_last_annotation())
        self.master.bind("<BackSpace>", lambda event: self._remove_last_annotation())

        # Command A to annotate
        self.master.bind("<Command-e>", lambda event: self.set_enable_annotation_editing(not self._enable_annotation_editing))

        # "N" to mark the current frame as "no annotations" -
        self.master.bind("<n>", lambda event: self._set_no_annotations())

        # Escape to cancel and return the original annotations
        self.master.bind("<Escape>", lambda event: self ._cancel())

        self.master.bind("<Return>", lambda event: self._close())

    def set_enable_annotation_editing(self, enable_annotation_editing: bool):
        self._enable_annotation_editing = enable_annotation_editing
        self.redraw()

    def _cancel(self):
        self._set_frame_annotations(replace(self._current_frame_annotation, annotations=self._last_externally_set_annotations))
        self._close()

    def _set_no_annotations(self):
        self._current_frame_annotation = replace(self._current_frame_annotation, annotations=None)
        self.redraw()

    def _on_click(self, event: tk.Event, xy: Tuple[int, int]) -> bool:

        if not self._enable_annotation_editing:
            return True

        x, y = xy
        is_drag = event.type == tk.EventType.Motion
        is_release = event.type == tk.EventType.ButtonRelease

        if is_drag and self._mouse_down_location is None:
            self._mouse_down_location = (x, y)

        # Get the current box (or None if mouse is not down)
        if self._mouse_down_location is not None:
            xo, yo = self._mouse_down_location
            current_box_ijhw = yo, xo, (y-yo)*2, (x-xo)*2
        else:
            current_box_ijhw = None

        # If it's a release, add the box:
        if current_box_ijhw is not None:
            if is_release:
                self._mouse_down_location = None
                @dataclass
                class LabelValueDesc:
                    label: str
                    value: int
                    description: str
                defaults = LabelValueDesc(label='', value=1, description='')
                values = ui_choose_parameters(LabelValueDesc, initial_params=defaults)
                temp_box = (Annotation(ijhw_box=current_box_ijhw, label=values.label, value=values.value, description=values.description),) if values is not None else ()

            else:
                temp_box = (Annotation(ijhw_box=current_box_ijhw, label=''),)
            annotation = replace(self._current_frame_annotation, annotations=(tuple(self._current_frame_annotation.annotations) or ()) + temp_box)

            if is_release:
                self._current_frame_annotation = annotation

        else:
            annotation = self._current_frame_annotation
        self.redraw(annotation)
        return True

    def _remove_last_annotation(self):
        print("Removing last annotation")
        if self._current_frame_annotation is not None:
            self._current_frame_annotation = replace(self._current_frame_annotation, annotations=self._current_frame_annotation.annotations[:-1])
            self.redraw()

    def _close(self):
        self.master.destroy()

    def set_frame_annotations(self, frame_annotation: Optional[AnnotatedImage], detections: Optional[Sequence[Detection]] = None):

        self._last_externally_set_annotations = frame_annotation.annotations if frame_annotation is not None else None
        self._set_frame_annotations(frame_annotation, detections)

    def _set_frame_annotations(self, frame_annotation: Optional[AnnotatedImage], detections: Optional[Sequence[Detection]] = None):
        self.set_enable_annotation_editing(self._enable_annotation_editing_initially)
        self._current_frame_annotation = frame_annotation
        self._current_frame_detections = detections
        self.redraw()

    def set_view_frame(self, view_frame: ImageViewInfo):
        self._image_view.set_image_frame(view_frame)

    def get_frame_annotation(self) -> AnnotatedImage:
        return self._current_frame_annotation

    def redraw(self, frame_annotation: Optional[AnnotatedImage] = None):
        # print("Redrawing")
        if frame_annotation is None:
            frame_annotation = self._current_frame_annotation

        if frame_annotation is None:
            instructions = "No image loaded ."
            display_image = get_awaiting_input_image(size_xy=(800, 600), text=f'Awaiting Image')
        else:
            if self._enable_annotation_editing:
                instructions = "Click and drag to add a new annotation, delete to remove the last one."
            else:
                instructions = "Command-E to enable annotation editing."
            if self._current_frame_detections is None:
                display_image = frame_annotation.render(thickness=ceil(2/frame_view.zoom_level) if (frame_view:=self._image_view.get_image_view_frame_or_none()) else 1)
            else:
                display_image = render_detections_unto_image(annotated_image=frame_annotation, detections=self._current_frame_detections)

            self._label.config(text=f"{len(frame_annotation.annotations) if frame_annotation.annotations is not None else 'No'} annotations.    {instructions}\nWASD/ZXC to pan/zoom")

        self._image_view.set_image(display_image)


def edit_annotated_frame(
        frame: AnnotatedImage,
        initial_view_frame: Optional[ImageViewInfo] = None,
) -> Optional[Sequence[Annotation]]:

    with hold_tkinter_root_context() as root:

        toplevel = tk.Toplevel()
        # Set size to 1280x720
        toplevel.geometry("1280x720")
        toplevel.title("Edit Annotations")

        view = FrameAnnotationView(toplevel, enable_annotation_editing=False)
        view.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        view.set_frame_annotations(frame)
        if initial_view_frame is not None:
            view.set_view_frame(initial_view_frame)

        # Try just making a fixed-size window instead
        # view = tk.Frame(toplevel)
        # view.pack(side=tk.TOP, fill=tk.BOTH, expand=True)



        def cancel():
            toplevel.destroy()


        # Escape to cancel and return the original annotations
        toplevel.bind("<Escape>", lambda event: toplevel.destroy())

        # toplevel.update()
        toplevel.wait_window()

        return view.get_frame_annotation().annotations


class FrameDatabaseViewer(tk.Frame):

    def __init__(self, master: tk.Frame, data_loader: AnnotatedImageDataLoader, dataset_detection_result: Optional[DatasetDetectionResult] = None):
        super().__init__(master)

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        with populate_frame(tk.Frame(self, width=400)) as self._left_panel:
            self._left_panel.grid(column=0, row=0, sticky=tk.NSEW)
            self._left_panel.grid_propagate(False)

            # Lets do a treeview instead
            self._tree_view = Treeview(self._left_panel, columns=('Name', 'N Annotations', ))
            self._tree_view.heading('#0', text='Index')
            self._tree_view.heading('#1', text='Name')
            self._tree_view.heading('#2', text='#A')
            self._tree_view.column('#0', width=50, stretch=tk.NO)
            self._tree_view.column('#2', width=50, stretch=tk.NO)
            self._tree_view.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            # Show the treeview's contents down to 2 nesting levels
            self._tree_view.bind('<<TreeviewOpen>>', lambda event: self._tree_view.item(event.widget.focus(), open=True))
            # Callback on selecting a row
            self._tree_view.bind('<<TreeviewSelect>>', lambda event: self._on_select_row(event, self._tree_view.selection()[0], self._tree_view.item(self._tree_view.selection()[0])))

        with populate_frame(tk.Frame(self)) as self._right_panel:
            self._right_panel.grid(column=1, row=0, sticky=tk.NSEW)

            # A label at the bottom
            self._bottom_panel = tk.Frame(self._right_panel, height=30)
            self._bottom_panel.pack(side=tk.BOTTOM, fill=tk.X, expand=True)
            self._left_button = RespectableButton(self._bottom_panel, text="<", command=lambda: self._on_change_detector(-1), shortcut="<Left>")
            self._left_button.pack(side=tk.LEFT, fill=tk.X, expand=False)
            self._right_button = RespectableButton(self._bottom_panel, text=">", command=lambda: self._on_change_detector(1), shortcut="<Right>")
            self._right_button.pack(side=tk.LEFT, fill=tk.X, expand=False)
            self._label = tk.Label(self._bottom_panel, text="")
            self._label.pack(side=tk.LEFT, fill=tk.X, expand=True)

            self._frame_annotation_view = FrameAnnotationView(self._right_panel, enable_annotation_editing=False)
            self._frame_annotation_view.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.after(1, self._update_tree_view_with_annotations)
        self._data_loader = data_loader
        self._dataset_detection_result: Optional[DatasetDetectionResult] = dataset_detection_result
        self._current_detector: Optional[str] = None

    def set_data_loader(self, data_loader: AnnotatedImageDataLoader):
        self._data_loader = data_loader

    def _on_change_detector(self, shift: int):
        if self._dataset_detection_result is not None:
            detector_names = [None] + list(self._dataset_detection_result.per_frame_results[0].detections.keys())
            # if self._current_detector is None:
            #     self._current_detector = detector_names[0]
            # else:
            self._current_detector = detector_names[(detector_names.index(self._current_detector) + shift) % len(detector_names)]
            self._redraw_label()

    def _redraw_label(self):
        currently_selected_row = self._tree_view.selection()[0]
        currently_selected_row_index = self._tree_view.index(currently_selected_row)

        all_channels = ["Ground Truth"] + (list(self._dataset_detection_result.per_frame_results[0].detections.keys()) if self._dataset_detection_result is not None else [])
        this_channel = self._current_detector if self._current_detector is not None else "Ground Truth"
        all_other_channels = [c for c in all_channels if c != this_channel]

        other_channels_message = f", Other channels [{', '.join(all_other_channels)}]" if len(all_other_channels) > 0 else ""

        if len(all_channels) == 1:
            # If it's packed
            if self._bottom_panel.winfo_ismapped():
                self._bottom_panel.pack_forget()

        else:
            # If it's not packed
            if not self._bottom_panel.winfo_ismapped():
                self._bottom_panel.pack(side=tk.BOTTOM, fill=tk.X, expand=True)

            if self._current_detector is None:
                self._frame_annotation_view.set_frame_annotations(self._frame_annotation_view.get_frame_annotation(), detections=None)
                self._label.config(text=f"Ground Truth {other_channels_message}")
            else:
                this_frame_result = self._dataset_detection_result.per_frame_results[currently_selected_row_index]
                detections = this_frame_result.detections[self._current_detector]
                pred_result = this_frame_result.get_prediction_results()[self._current_detector]
                self._frame_annotation_view.set_frame_annotations(self._frame_annotation_view.get_frame_annotation(), detections=detections)
                n_correct = pred_result.get_n_true_positives(threshold=-float('inf'))
                n_missed = pred_result.get_n_false_negatives_imperfect(threshold=-float('inf'))
                label = f"{self._current_detector}: {n_correct}/{len(pred_result.prediction_scores)} predictions correct, {pred_result.n_ground_truths-n_missed}/{pred_result.n_ground_truths} targets found {other_channels_message}"
                self._label.config(text=label)

    def _on_select_row(self, event, row_id, row_data):

        name, n_annotations, case_name, index = row_data['values']
        annotated_image = self._data_loader.lookup_annotated_image(case_name, index)
        self._frame_annotation_view.set_frame_annotations(annotated_image)
        self._redraw_label()

    def _update_tree_view_with_annotations(self):
        """ Fill the treeview with nested data: Case > Frame Index > Annotation Count """
        self._tree_view.delete(*self._tree_view.get_children())

        for i, (case_name, incase_ix, annotation_source) in enumerate(self._data_loader.iter_case_index_annotated_image_sources()):
            annotated_image_src: AnnotatedImageSource
            self._tree_view.insert('', 'end', text=str(i), values=(os.path.basename(os.path.splitext(annotation_source.source_path)[0]), len(annotation_source.annotations), case_name, incase_ix, ))


def open_annotation_database_viewer(data_loader: Optional[AnnotatedImageDataLoader] = None, results: Optional[DatasetDetectionResult] = None):

    if data_loader is None:
        data_loader = AnnotatedImageDataLoader.from_folder(DEFAULT_DATASET_FOLDER)

    with hold_tkinter_root_context() as root:

        view = FrameDatabaseViewer(root, data_loader=data_loader, dataset_detection_result=results)
        # Set geometry to 1280x720
        root.geometry("1280x720")
        view.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # view._update_table()

        # Escape to close
        root.bind("<Escape>", lambda event: root.destroy())

        root.mainloop()


if __name__ == '__main__':

    open_annotation_database_viewer()

    # frame = AnnotatedImage(
    #     image_frame=cv2.imread(AssetImages.BASALT_CANYON),
    #     annotations=()
    # )
    # new_annotations = edit_annotated_frame(frame=frame)
    # print(new_annotations)


