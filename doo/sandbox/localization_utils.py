import itertools
import numpy as np
import scipy.ndimage
import scipy.spatial


from .. import utils
from ..preprocessing import image as preprocessing_image


def _generate_centers_and_final_shape(img_shape,
                                      stride,
                                      first_center):
    """
    generates final shape and centers for applying a sliding window
    """
    dim_ranges = []
    for dim_min, dim_max, dim_stride in zip(first_center, img_shape, stride):
        dim_ranges.append(range(dim_min, dim_max, dim_stride))
    final_shape = tuple([len(dim_range) for dim_range in dim_ranges])
    centers = itertools.product(*dim_ranges)
    return final_shape, centers


def _generate_patches(img, patch_shape, centers):
    """
    generate image patches from centers
    """
    for center in centers:
        ndimage = preprocessing_image.get_block_with_center_and_shape(
            img,
            center,
            patch_shape,
            fill_value=0)
        yield ndimage


def sliding_window_apply(img,
                         fn,
                         patch_shape,
                         batch_size,
                         stride,
                         first_center):
    """
    applies a function in a sliding window fashion to patches in an image

    batch_size:
    maximum number of patches to run through the fn at a time, -1 for all
    patches at once

    first_center:
    the center point where all iteration starts at (eg. (0, 0))
    """
    assert len(stride) == len(img.shape) == len(first_center)

    # generate centers
    final_shape, centers = _generate_centers_and_final_shape(img.shape,
                                                             stride,
                                                             first_center)

    # generate patches
    patches = _generate_patches(img, patch_shape, centers)

    # batch batches
    batches = utils.toolz.partition_all(batch_size, patches)

    # run fn
    result_list = []
    for batch in batches:
        result = fn(batch)
        result_list.append(result)
    flat_result = np.concatenate(result_list)

    # reshape into img
    reshaped_result = flat_result.reshape(*final_shape)
    return reshaped_result


def find_maxima(img,
                blur_sigma,
                max_filter_size,
                threshold):
    """
    find local maxima of an image
    """
    blurred = scipy.ndimage.gaussian_filter(img, sigma=blur_sigma)
    maxed = scipy.ndimage.maximum_filter(blurred, size=max_filter_size)
    points = zip(*np.where(((maxed == blurred) & (blurred > threshold))))
    return points


def convert_points(points, stride, first_center):
    """
    converts points that were computed in a strided fashion to the point
    that they would correspond to in the original image
    """
    if len(points) > 0:
        return np.array(points) * stride + first_center
    else:
        return np.zeros((0, len(stride)))

def match_true_and_predicted(list_of_true_points,
                             list_of_predicted_points,
                             correctness_threshold):
    """
    given 2 lists of lists of tuples (points), the former as the true points
    and the latter as predicted points, returns a map with lists of lists of
    matched and unmatched points.
    """
    true_matched = []
    true_unmatched = []
    pred_matched = []
    pred_unmatched = []
    for p_trues, p_preds in zip(list_of_true_points,
                                list_of_predicted_points):
        # make a copy to mutate
        is_true_matched = [False] * len(p_trues)
        is_pred_matched = [False] * len(p_preds)
        for true_idx, p_true in enumerate(p_trues):
            for pred_idx, p_pred in enumerate(p_preds):
                dist = scipy.spatial.distance.euclidean(p_true, p_pred)
                if dist < correctness_threshold:
                    is_pred_matched[pred_idx] = True
                    is_true_matched[true_idx] = True
        true_matched_one_image = []
        true_unmatched_one_image = []
        for true_idx, p_true in enumerate(p_trues):
            if is_true_matched[true_idx]:
                true_matched_one_image.append(p_true)
            else:
                true_unmatched_one_image.append(p_true)
        pred_matched_one_image = []
        pred_unmatched_one_image = []
        for pred_idx, p_pred in enumerate(p_preds):
            if is_pred_matched[pred_idx]:
                pred_matched_one_image.append(p_pred)
            else:
                pred_unmatched_one_image.append(p_pred)

        true_matched.append(true_matched_one_image)
        true_unmatched.append(true_unmatched_one_image)
        pred_matched.append(pred_matched_one_image)
        pred_unmatched.append(pred_unmatched_one_image)

    return dict(
        true_matched=true_matched,
        true_unmatched=true_unmatched,
        pred_matched=pred_matched,
        pred_unmatched=pred_unmatched,
    )

def localization_metrics(list_of_true_points,
                         list_of_predicted_points,
                         correctness_threshold):
    """
    given 2 lists of lists of tuples (points), the former as the true points
    and the latter as predicted points, returns number of matched and unmatched preds
    and true points
    """
    matched = match_true_and_predicted(list_of_true_points,
                                       list_of_predicted_points,
                                       correctness_threshold)

    return {k: sum(map(len,matched[k])) for k in matched.keys()}
