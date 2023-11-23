"""
This file contains some pre-build models from Eagle Eyes.

You can try running your models against these models to see how you compare.

"""
import os
import hackathon
from hackathon.model_utils.tf_model_utils import TFPrebuiltModel


ASSET_DIR = os.path.abspath(os.path.join(hackathon.__file__, '..', '..', 'assets'))


def load_v0_model():
    """
    V0 just finds the mean and variance of pixels in the images,
    takes the Mahalanobis distance of each pixel to make a heatmap,
    and takes a local maxima of the the heatmap
    """
    return TFPrebuiltModel.from_model_file(os.path.join(ASSET_DIR, 'V0.eagle'))


def load_v1_model():
    """
    V1 does the same as V0 but with an initial pre-processing step of doing center-surround
    filtering on the image.
    """
    return TFPrebuiltModel.from_model_file(os.path.join(ASSET_DIR, 'V1.ColorAlone.eagle'))


def load_v2p5_model():
    """
    V2.5 does what v1 does, but with some Kernel Density Estimation to postprocsess
    the local maxima.
    """
    return TFPrebuiltModel.from_model_file(os.path.join(ASSET_DIR, 'V2.5.Photo.Fast.eagle'))
