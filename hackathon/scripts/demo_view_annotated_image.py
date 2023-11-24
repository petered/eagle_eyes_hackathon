from hackathon.data_utils.data_loading import AnnotatedImageDataLoader
from hackathon.ui_utils.tk_utils.tkshow import tkshow


def demo_view_annotated_image():
    """
    Shows how you can view a raw/annotated image.
    Use Left/Right arrow keys to switch between raw/annotated images.
    Zoom with Z/X/C, and Pan with W/A/S/D.
    """
    data_loader = AnnotatedImageDataLoader.from_folder()
    annotated_image = data_loader[5]
    tkshow({'annotated': annotated_image.render(), 'raw': annotated_image.image})


if __name__ == '__main__':
    demo_view_annotated_image()