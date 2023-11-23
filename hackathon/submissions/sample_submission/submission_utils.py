from typing import NewType, Union, Optional, Tuple

import tensorflow as tf

TensorIJCoords = NewType('TensorIJCoords', tf.Tensor)
TensorImage = NewType('TensorImage', tf.Tensor)  # A (height, width, n_colors) uint8 image
TensorFloatImage = NewType('TensorFloatImage', tf.Tensor)
TensorPixelFloatArray = NewType('TensorPixelArray', tf.Tensor)  # A (N, n_colors) array of flattened pixels
TensorFloatVector = NewType('TensorFloatVector', tf.Tensor)  # Vector of floats
TensorImageCoords = NewType('TensorFloatImage', TensorFloatImage)
TensorHeatmap = NewType('TensorHeatmap', TensorFloatImage)  # A (height, width) heatmap
TensorMaskVector = NewType('TensorMaskVector', tf.Tensor)  # A boolean vector
TensorMaskImage = NewType('TensorMaskImage', tf.Tensor)  # A (height, width) boolean mask
TensorInvCovMat = NewType('TensorCovMat', tf.Tensor)  # A (n_colors x n_colors) covariance matrix,
TensorColor = NewType('TensorColor', tf.Tensor)  # A (n_colors, ) vector representing a color
TensorIndexImage = NewType('TensorColor', tf.Tensor)  # A (height, width) array of indices
TensorIndexVector = NewType('TensorIndexVector', tf.Tensor)  # A vector of indices
TensorLTRBBoxes = NewType('TensorLTRBBoxes', tf.Tensor)  # An array of int32 boxes, specified by (Left, Right, Top, Bottom) pixel
TensorIJHWBoxes = NewType('TensorIJHWBoxes', tf.Tensor)  # An array of int32 boxes, specified by (y-center (from top down), x_center, height, width) pixel
TensorInt = NewType('TensorInt', tf.Tensor)  # Just an integer
TensorDistanceMat = NewType('TensorDistanceMat', tf.Tensor)
TensorFloat = NewType('TensorFloat', tf.Tensor)
INVCOV_COLOR_DEFAULT_EPSILON = 1e-6


def tf_box_sum_image_from_padded_image(padded_image, width: int):
    integral_image = tf.cumsum(tf.cumsum(padded_image, axis=0), axis=1)
    return integral_image[width:, width:] \
           - integral_image[width:, :-width] \
           - integral_image[:-width, width:] \
           + integral_image[:-width, :-width]


def tf_box_sum(image: Union[TensorImage, TensorFloatImage, TensorHeatmap], width: int, n_iter: int = 1, border_mode='CONSTANT'):
    """ Sum the pixels in the box with the given width around each pixel, repeating this process n_iter times.
    Note that as n_iter gets large, this becomes closer to a gaussian filter.
    """  # TODO: Make probably more efficient by padding only once
    lwidth = width // 2 + 1
    rwidth = width - lwidth
    for _ in range(n_iter):
        padded_image = tf.pad(image, paddings=[(lwidth, rwidth), (lwidth, rwidth)] + [(0, 0)] * (len(image.shape) - 2), mode=border_mode)
        image = tf_box_sum_image_from_padded_image(padded_image, width)
    return image


def tf_box_filter(image: Union[TensorImage, TensorFloatImage, TensorHeatmap], width: int, normalize: bool = True, weights: Optional[TensorHeatmap] = None,
                  weight_eps: float = 1e-6, norm_weights: bool = True, n_iter: int = 1):
    """
    An integral-image based filter that (iteratively) applies a box-filter.
    This is an O((H+width)*(W+width)*n_iter) operation, where H, W are the height and width of the image and width is the width of the box in pixels.
    """
    image = tf.cast(image, tf.float32) if image.dtype != tf.float64 else image

    if weights is not None:
        if norm_weights:
            weights = weights / (width ** 2)
        if len(image.shape) == 3:
            weights = weights[:, :, None]  # Lets us broadcast weights against image

        image = image * weights

    box_image = tf_box_sum(image, width=width, n_iter=n_iter, border_mode='SYMMETRIC')

    if not normalize:
        return box_image if (weights is None or not norm_weights) else box_image * (width ** 2)
    elif weights is None:
        return box_image / (tf.cast(width, image.dtype) ** 2)
    else:
        box_weights = tf_box_sum(weights, width=width, n_iter=n_iter, border_mode='SYMMETRIC')
        return (box_image + weight_eps) / (box_weights + weight_eps)


def tf_holy_box_blur(image: TensorFloatImage, inner_box_width: int, outer_box_width: int, weights: Optional[TensorHeatmap] = None) -> TensorFloatImage:
    if weights is not None:
        image = image * weights
    box_sum = tf_box_filter(image, width=outer_box_width, normalize=False) - tf_box_filter(image, width=inner_box_width, normalize=False)

    if weights is not None:
        weight_sum = tf_box_filter(weights, width=outer_box_width, normalize=False) - tf_box_filter(weights, width=inner_box_width, normalize=False)
        return box_sum / weight_sum
    else:
        return box_sum / tf.cast(outer_box_width ** 2 - inner_box_width ** 2, box_sum.dtype)


def tf_dilate(heatmap, width: int):
    """ Dilate the heatmap with a square kernel
    Note - this is inefficient, see test_dilation_options and
    """
    row_dilation = tf.nn.dilation2d(heatmap[None, :, :, None], filters=tf.zeros((1, width, 1), dtype=heatmap.dtype),
                                    strides=(1, 1, 1, 1), padding="SAME", data_format="NHWC",
                                    dilations=(1, 1, 1, 1))
    full_dilation = tf.nn.dilation2d(row_dilation, filters=tf.zeros((width, 1, 1), dtype=heatmap.dtype),
                                     strides=(1, 1, 1, 1), padding="SAME", data_format="NHWC",
                                     dilations=(1, 1, 1, 1))
    return full_dilation[0, :, :, 0]


def tf_argdilate(heatmap: TensorHeatmap, width: int, epsilon = 1e-6) -> Tuple[TensorIndexImage, TensorHeatmap]:
    """
    Dilate the heatmap, and return
        argdilation, dilation
    Where argdilation contains the indices of the pixels that got dilated to that point (in raster order).
    NOTE: HEATMAP IS EXPECTED TO BE POSITIVE - if it is not, this will be buggy
    Note that in the case of ties pixels with a HIGHER raster index will win.
    """
    tf.assert_less(tf.size(heatmap), 0xCFFFFFF, message=f"This method only works for heatmaps with fewer than {0xFFF} pixels")
    heatmap = tf.cast(heatmap, tf.float32)
    max_heat = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_heat*(1+epsilon))  # epsilon prevents this bug: https://github.com/tensorflow/tensorflow/issues/60078
    # Sneakily piggyback the indices along with the values.  Since the largest index is less than the smallest value increment, it shouldnt affect the result
    heatmap_int = tf.bitwise.left_shift(tf.cast(tf.image.convert_image_dtype(heatmap, tf.uint32), tf.int64), 31) \
                  + tf.reshape(tf.range(tf.size(heatmap), dtype=tf.int64), tf.shape(heatmap))
    heatmap_int_dilated = tf_dilate(heatmap_int, width=width)
    argdilation = tf.bitwise.bitwise_and(heatmap_int_dilated, 0xCFFFFFF)
    return argdilation, heatmap_int_dilated


def tf_local_argmaxima(heatmap, width: int) -> TensorIJCoords:
    arg_ixs, dilated = tf_argdilate(heatmap, width=width)
    local_maxima_ij = tf.where(tf.reshape(tf.range(tf.size(heatmap), dtype=tf.int32), tf.shape(heatmap)) == tf.cast(arg_ixs, dtype=tf.int32))
    return tf.cast(local_maxima_ij, dtype=tf.int32)


def tf_local_maxima(heatmap: TensorHeatmap, width: int, ) -> Tuple[TensorIJCoords, TensorFloatVector]:
    local_maxima_ij = tf_local_argmaxima(heatmap, width)
    return local_maxima_ij, tf.gather_nd(heatmap, local_maxima_ij)


def compute_inverse_correlation(arr: TensorPixelFloatArray, epsilon: float = INVCOV_COLOR_DEFAULT_EPSILON) -> TensorInvCovMat:
    """ You can call this an inverse covariance matrix if the mean has already been subtracted """
    corr = tf.cast(tf.matmul(tf.transpose(arr), arr) / tf.cast(tf.shape(arr)[0], tf.float32), dtype=tf.float64) + tf.eye(tf.shape(arr)[-1], dtype=tf.float64) * epsilon
    invcorr = tf.linalg.inv(corr)
    return tf.cast(invcorr, dtype=tf.float32)


def compute_mahalonabis_dist_sq_from_meansub_and_invcov(meansub_arr: TensorPixelFloatArray, invcov: TensorInvCovMat):
        return tf.reduce_sum((meansub_arr @ (0.5 * invcov)) * meansub_arr, axis=1)


def tf_round_to_odd(number: TensorFloat) -> TensorInt:
    # int(round(x // 2)) * 2 + 1
    return tf.cast(tf.round(number/2)*2+1, tf.int32)


def compute_mahalonabis_dist_sq(arr: TensorPixelFloatArray, epsilon: float = INVCOV_COLOR_DEFAULT_EPSILON):
    meansub_arr = arr - tf.reduce_mean(arr, axis=0)
    invcov = compute_inverse_correlation(meansub_arr, epsilon=epsilon)
    return compute_mahalonabis_dist_sq_from_meansub_and_invcov(meansub_arr, invcov)
