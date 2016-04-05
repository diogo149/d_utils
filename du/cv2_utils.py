"""
This is a (hopefully) saner wrapper around OpenCV (cv2). The main differences
are:
- all functions return and expect float64 arrays with values in [0, 1]
  - some will work for ranges beyond [0, 1] though
- mutable APIs are wrapped to be immutable (ie. return new arrays)
- the order of coordinates are flipped to match python array syntax
  - eg. (x, y) = x-th row and y-th column

These interface changes do have a performance penalty though.
"""

import contextlib
import types
import cv2
import numpy as np
import functools


def str_to_interpolation(s):
    """
    http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html
    INTER_NEAREST - a nearest-neighbor interpolation
    INTER_LINEAR - a bilinear interpolation (used by default)
    INTER_AREA - resampling using pixel area relation. It may be a preferred
                 method for image decimation, as it gives moire-free results.
                 But when the image is zoomed, it is similar to the
                 INTER_NEAREST method.
    INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
    INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood

    To shrink an image, it will generally look best with CV_INTER_AREA
    interpolation, whereas to enlarge an image, it will generally look best
    with CV_INTER_CUBIC (slow) or CV_INTER_LINEAR (faster but still looks OK)
    """
    if s == "linear":
        return cv2.INTER_LINEAR
    elif s == "cubic":
        return cv2.INTER_CUBIC
    elif s == "area":
        return cv2.INTER_AREA
    else:
        raise ValueError


def str_to_border_mode(s):
    """
    http://docs.opencv.org/modules/imgproc/doc/filtering.html

    Various border types, image boundaries are denoted with '|'

    BORDER_REPLICATE:     aaaaaa|abcdefgh|hhhhhhh
    BORDER_REFLECT:       fedcba|abcdefgh|hgfedcb
    BORDER_REFLECT_101:   gfedcb|abcdefgh|gfedcba
    BORDER_WRAP:          cdefgh|abcdefgh|abcdefg
    BORDER_CONSTANT:      iiiiii|abcdefgh|iiiiiii  with some specified 'i'
    """
    if s == "nearest":
        return cv2.BORDER_REPLICATE
    elif s == "reflect":
        return cv2.BORDER_REFLECT
    elif s == "reflect101":
        return cv2.BORDER_REFLECT_101
    elif s == "wrap":
        return cv2.BORDER_WRAP
    elif s == "constant":
        return cv2.BORDER_CONSTANT
    else:
        raise ValueError


def to_01(img):
    if img.max() - img.min() > 1e-6:
        img = img.astype(np.float)
        return (img - img.min()) / (img.max() - img.min())
    else:
        return np.zeros(img.shape, dtype=np.uint8)


def img_01_to_uint8(img):
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def img_uint8_to_01(img):
    assert img.dtype == np.uint8
    img = np.clip(img, 0, 255)
    return img.astype(np.float) / 255


def wrap_01_to_uint8_conversion(f):
    """Decorator function that can be applied to opencv2 utilities to
    automatically convert the first arg to a uint8 representation and
    convert the return value to an 01 image representation.
    The underlying function will always work on a copy of the image."""

    @functools.wraps(f)
    def wrapped_func(img, *args, **kwargs):
        assert img.dtype == np.float32 or img.dtype == np.float64

        img_conv = img_01_to_uint8(img)
        res = f(img_conv, *args, **kwargs)
        return img_uint8_to_01(res)

    wrapped_func.without_conversion = f
    return wrapped_func


def imread_01(filename):
    """
    Reads an image file and returns it as a numpy float array with values
    scaled to [0.0, 1.0]
    """
    as_uint8 = cv2.imread(filename)
    assert as_uint8 is not None, "Error reading image: %s" % filename
    return img_uint8_to_01(as_uint8)


def imshow(img, keep_window=False):
    """
    wrapper around OpenCV's imshow that allows properly closing the image
    window
    """
    cv2.imshow("tmp", img)
    if keep_window:
        # this pauses and waits until the a key is pressed, to resume code,
        # but keeps the window opened after the key is pressed
        closeWindow = -1
        while closeWindow < 0:
            closeWindow = cv2.waitKey(1)
        closeWindow = -1
        cv2.destroyAllWindows()
    else:
        # this pauses and waits until the a key is pressed, to resume code,
        # and closes the window after the key is pressed
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # for some reason, have to do this a few times...
        for i in range(1, 5):
            cv2.waitKey(1)


def imwrite_01(filename, img, params=None):
    """
    NOTE: this function may not do ANYTHING if it fails (eg. when the process
    doesn't have permission to write a file)
    """
    tmp = img_01_to_uint8(img)
    cv2.imwrite(filename, tmp, params)


def imwrite_01_jpeg(filename, img, quality=95):
    imwrite_01(filename, img, [cv2.IMWRITE_JPEG_QUALITY, quality])


@wrap_01_to_uint8_conversion
def _convert_color_space(as_uint8, converter):
    return cv2.cvtColor(as_uint8, converter)


def _convert_color_space_fn(in_space, out_space):
    converter = getattr(cv2, "COLOR_{}2{}".format(in_space, out_space))
    return functools.partial(_convert_color_space, converter=converter)


def convert_color_space(img, in_space, out_space):
    """
    example:
    convert_color_space(lena, "GRAY", "BGR")
    """
    return _convert_color_space_fn(in_space, out_space)(img)

bgr_to_gray = _convert_color_space_fn("BGR", "GRAY")
gray_to_bgr = _convert_color_space_fn("GRAY", "BGR")

bgra_to_rgba = _convert_color_space_fn("BGRA", "RGBA")
rgba_to_bgra = _convert_color_space_fn("RGBA", "BGRA")

gray_to_rgba = _convert_color_space_fn("GRAY", "RGBA")
rgba_to_gray = _convert_color_space_fn("RGBA", "GRAY")

rgb_to_gray = _convert_color_space_fn("RGB", "GRAY")
gray_to_rgb = _convert_color_space_fn("GRAY", "RGB")

bgr_to_lab = _convert_color_space_fn("BGR", "LAB")
lab_to_bgr = _convert_color_space_fn("LAB", "BGR")

bgr_to_yuv = _convert_color_space_fn("BGR", "YUV")
yuv_to_bgr = _convert_color_space_fn("YUV", "BGR")

bgr_to_hsv = _convert_color_space_fn("BGR", "HSV")
hsv_to_bgr = _convert_color_space_fn("HSV", "BGR")

bgr_to_hls = _convert_color_space_fn("BGR", "HLS")
hls_to_bgr = _convert_color_space_fn("HLS", "BGR")

bgr_to_luv = _convert_color_space_fn("BGR", "LUV")
luv_to_bgr = _convert_color_space_fn("LUV", "BGR")

bgr_to_xyz = _convert_color_space_fn("BGR", "XYZ")
xyz_to_bgr = _convert_color_space_fn("XYZ", "BGR")

bgr_to_ycr_cb = _convert_color_space_fn("BGR", "YCR_CB")
ycr_cb_to_bgr = _convert_color_space_fn("YCR_CB", "BGR")


def equalize_hist(img):
    tmp = img_01_to_uint8(img)
    res = cv2.equalizeHist(tmp)
    return img_uint8_to_01(res)


def _clahe(img, **kwargs):
    # Defaults that make this look like the result of skimage version
    if "clipLimit" not in kwargs:
        kwargs["clipLimit"] = 2.0
    clahe = cv2.createCLAHE(**kwargs)
    return clahe.apply(img)


def equalize_hist_local(img, **kwargs):
    """Returns img with adaptive local contrast equalization (CLAHE) applied.

    See:
    http://en.wikipedia.org/wiki/Adaptive_histogram_equalization#Contrast_Limited_AHE
    """

    assert img.ndim == 2, ("Expected single-channel grayscale img, "
                           "try bgr_to_gray")
    tmp = img_01_to_uint8(img)
    res = _clahe(tmp, **kwargs)
    return img_uint8_to_01(res)


def equalize_hist_channels(img,
                           local=False,
                           channels=None,
                           in_place=False,
                           **kwargs):
    """
    local:
    if True, use local histogram equalization instead of global

    channels:
    sequence of channels to normalize over (default: all of them)

    in_place:
    whether or not to edit the existing image
    """
    if not in_place:
        img = img.copy()
    if channels is None:
        channels = range(img.shape[-1])
    for i in channels:
        if local:
            img[..., i] = equalize_hist_local(img[..., i], **kwargs)
        else:
            img[..., i] = equalize_hist(img[..., i], **kwargs)
    return img


def draw_circle(img,
                center,
                radius=2,
                color=(0, 0, 255),
                thickness=1,
                **kwargs):
    """
    NOTE: thickness of -1 fills in the circle
    """
    tmp = img_01_to_uint8(img)
    x, y = center
    cv2.circle(tmp,
               center=(y, x),
               radius=radius,
               color=color,
               thickness=thickness,
               **kwargs)
    return img_uint8_to_01(tmp)


@wrap_01_to_uint8_conversion
def draw_circles(img_copy,
                 centers,
                 radius=2,
                 color=(0, 0, 255),
                 thickness=1,
                 **kwargs):
    """Same as draw_circle, but takes a list of center coords
    and plots one circle for each center coord given.
    MUCH faster than using draw_circle multiple times."""
    for center in centers:
        x, y = center
        cv2.circle(img_copy,
                   center=(y, x),
                   radius=radius,
                   color=color,
                   thickness=thickness,
                   **kwargs)
    return img_copy


def draw_line(img,
              pt1,
              pt2,
              color=(0, 0, 255),
              thickness=1,
              **kwargs):
    tmp = img_01_to_uint8(img)
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(tmp,
             pt1=(y1, x1),
             pt2=(y2, x2),
             color=color,
             thickness=thickness,
             **kwargs)
    return img_uint8_to_01(tmp)


def draw_rectangle(img,
                   corner1,
                   corner2,
                   color=(0, 0, 255),
                   thickness=1,
                   **kwargs):
    tmp = img_01_to_uint8(img)
    x1, y1 = corner1
    x2, y2 = corner2
    cv2.rectangle(tmp,
                  pt1=(y1, x1),
                  pt2=(y2, x2),
                  color=color,
                  thickness=thickness,
                  **kwargs)
    return img_uint8_to_01(tmp)


def laplacian(img):
    """
    returns gradient magnitude of image
    NOTE: should blur first
    """
    if img.dtype != np.float64:
        img = img.astype(np.float64)
    return cv2.Laplacian(img, cv2.CV_64F)


def sobel(img, dx=0, dy=0):
    """
    NOTE: should blur first
    """
    assert dx or dy
    return cv2.Sobel(img, cv2.CV_64F, dx=dx, dy=dy)


def canny_edges(img, *args, **kwargs):
    """
    NOTE: should blur first
    """
    as_uint8 = img_01_to_uint8(img)
    edges = cv2.Canny(as_uint8, *args, **kwargs)
    return img_uint8_to_01(edges)


def average_blur(img, kernel_size):
    return cv2.blur(img, ksize=(kernel_size, kernel_size))


def gaussian_blur(img, sigma, kernel_size=0):
    """
    http://docs.opencv.org/modules/imgproc/doc/filtering.html#gaussianblur
    """
    return cv2.GaussianBlur(img,
                            ksize=(kernel_size, kernel_size),
                            sigmaX=sigma,
                            sigmaY=sigma)


def median_blur(img, kernel_size):
    as_uint8 = img_01_to_uint8(img)
    blurred = cv2.medianBlur(as_uint8, ksize=kernel_size)
    return img_uint8_to_01(blurred)


def bilateral_filter(img,
                     filter_size=9,
                     sigma=None,
                     sigma_color=None,
                     sigma_space=None):
    """
    preserves edges while removing noise, but considerably slower
    http://docs.opencv.org/modules/imgproc/doc/filtering.html#bilateralfilter
    """
    assert (sigma is not None) != ((sigma_color is not None)
                                   and (sigma_space is not None))
    if sigma is not None:
        sigma_color = sigma
        sigma_space = sigma
    img_f32 = img.astype(np.float32)
    res_f32 = cv2.bilateralFilter(img_f32,
                                  d=filter_size,
                                  sigmaColor=sigma_color,
                                  sigmaSpace=sigma_space)
    return res_f32.astype(np.float)


def adaptive_threshold(img,
                       block_size,
                       C,
                       adaptive_method="gaussian",
                       threshold_type="binary"):
    """
    block_size: size of the area to look at when adapting threshold
    C: value to subtract from weighted average over block size
       can be positive or negative - positive makes more values pass the
       threshold
    NOTE: should blur the image before using
    """
    if adaptive_method == "gaussian":
        adaptive_method_ = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    elif adaptive_method == "mean":
        adaptive_method_ = cv2.ADAPTIVE_THRESH_MEAN_C
    else:
        raise ValueError

    if threshold_type == "binary":
        threshold_type_ = cv2.THRESH_BINARY
    elif threshold_type == "otsu":
        threshold_type_ = cv2.THRESH_OTSU
    elif threshold_type == "binary_inverse":
        threshold_type_ = cv2.THRESH_BINARY_INV
    else:
        raise ValueError

    as_uint8 = img_01_to_uint8(img)
    res = cv2.adaptiveThreshold(as_uint8,
                                maxValue=255,
                                adaptiveMethod=adaptive_method_,
                                thresholdType=threshold_type_,
                                blockSize=block_size,
                                C=C)
    return img_uint8_to_01(res)


def otsu_threshold(img):
    """
    NOTE: should blur the image before using
    NOTE: might be better to use mahotas.thresholding.otsu(blurred)
    """
    as_uint8 = img_01_to_uint8(img)
    threshold_value, thresholded = cv2.threshold(
        as_uint8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_uint8_to_01(thresholded)


def resize_01(img, shape, interpolation="linear"):
    """
    resize img in 01 range

    interpolation:
    see documentation for str_to_interpolation
    """
    assert isinstance(shape, tuple)
    interpolation_ = str_to_interpolation(interpolation)

    x_shape, y_shape = shape
    res = cv2.resize(img, (y_shape, x_shape), interpolation=interpolation_)
    assert res.shape[:2] == shape

    # return back to original dimensions
    if res.ndim == 2 and img.ndim == 3 and img.shape[2] == 1:
        res = res[:, :, np.newaxis]
    return res


def resize(img, shape, interpolation="linear"):
    """
    works for images not limited to [0, 1]

    interpolation:
    see documentation for str_to_interpolation
    """
    # create img in [0, 1] range, because resize requires images in
    # this allows resize images that aren't in [0, 1]
    img_min = img.min()
    img_max = img.max()
    img_scale = float(img_max - img_min)
    # avoid nan's for all black images
    if img_scale == 0:
        img_scale = 1
    img_tmp = (img - img_min) / img_scale
    res = resize_01(img_tmp, shape, interpolation)
    # scale back to original range
    res = img_scale * res + img_min
    return res


def scale_contrast(img, scale_factor):
    assert img.min() >= 0
    assert img.max() <= 1
    return np.clip(img * scale_factor, 0, 1)


def match_template(img, template, method="normalized_correlation"):
    """
    http://docs.opencv.org/doc/tutorials/imgproc/histograms/template_matching/template_matching.html
    """
    if method == "normalized_correlation":
        method_ = cv2.TM_CCOEFF_NORMED
    elif method == "squared_difference":
        method_ = cv2.TM_SQDIFF
    else:
        raise ValueError
    img_f32 = img.astype(np.float32)
    template_f32 = template.astype(np.float32)
    res = cv2.matchTemplate(img_f32, template_f32, method_)
    # TODO convert to f64?
    return res


def sharpen(img, sigma, kernel_size=0, sharpen_amount=0.5):
    """
    http://en.wikipedia.org/wiki/Unsharp_masking#Digital_unsharp_masking
    http://stackoverflow.com/questions/4993082/how-to-sharpen-an-image-in-opencv
    """
    smoothed = gaussian_blur(img, sigma=sigma, kernel_size=kernel_size)
    sharpened = (1 + sharpen_amount) * img - sharpen_amount * smoothed
    return sharpened


def stereo_disparity_map(img_L, img_R):
    """
    computing stereo correspondence (disparity map) using the block matching
    algorithm
    http://docs.opencv.org/trunk/modules/cudastereo/doc/stereo.html
    """
    stereo = cv2.StereoBM()
    disparity_map = stereo.compute(img_01_to_uint8(img_L),
                                   img_01_to_uint8(img_R))
    return disparity_map


def find_contours(edge_img,
                  contour_retrieval_mode="list",
                  contour_approximation_method="simple"):
    """
    NOTE: takes in edge image
    returns a list of arrays with shape (n, 1, 2), each representing a
    contour with n points

    for contour retrieval mode:
    http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_contours/py_contours_hierarchy/py_contours_hierarchy.html

    for contour approximation method:
    http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html
    """
    mode = dict(
        list=cv2.RETR_LIST,
        external=cv2.RETR_EXTERNAL,
        tree=cv2.RETR_TREE,
        ccomp=cv2.RETR_CCOMP,
    )[contour_retrieval_mode]
    method = dict(
        none=cv2.CHAIN_APPROX_NONE,
        simple=cv2.CHAIN_APPROX_SIMPLE,
    )[contour_approximation_method]
    as_uint8 = img_01_to_uint8(edge_img)
    # returns contours and hierarchy
    cnts, _ = cv2.findContours(as_uint8,  # NOTE: this updates the array
                               mode=mode,
                               method=method)
    return cnts


def draw_contour(img, contour, color=(0, 0, 255), thickness=1):
    tmp = img_01_to_uint8(img)
    cv2.drawContours(tmp,
                     contours=[contour],
                     contourIdx=0,
                     color=color,
                     thickness=thickness)
    return img_uint8_to_01(tmp)


def keypoint(x,
             y,
             size,
             angle=-1.0,
             response=0.0,
             octave=0,
             class_id=-1):
    """
    size:
    diameter of the meaningful keypoint neighborhood

    angle:
    orientation in degrees [0, 360) -1 if not applicable

    response:
    the response by which the most strong keypoints have been selected

    octave:
    pyramid layer which the keypoint was extracted from

    class_id:
    can be used to cluster keypoints according to which object they are from

    http://docs.opencv.org/modules/features2d/doc/common_interfaces_of_feature_detectors.html
    """
    # NOTE: reversing x and y:
    return cv2.KeyPoint(y, x,
                        _size=size,
                        _angle=angle,
                        _response=response,
                        _octave=octave,
                        _class_id=class_id)


def keypoint_to_dict(keypoint):
    """
    convert cv2.KeyPoint to dict
    """
    y, x = keypoint.pt
    return dict(
        x=int(x),
        y=int(y),
        size=keypoint.size,
        response=keypoint.response,
        octave=keypoint.octave,
        class_id=keypoint.class_id,
        angle=keypoint.angle,
    )


def draw_keypoints(img, keypoints):
    """
    draws keypoints on an image

    keypoints:
    list of cv2.Keypoint
    """
    as_uint8 = img_01_to_uint8(img)
    res = cv2.drawKeypoints(as_uint8, keypoints,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    as_01 = img_uint8_to_01(res)
    return as_01


def sift_descriptors(img, keypoints):
    """
    returns the descriptors for a set of keypoints in an image (a float32
    numpy array with shape (len(keypoints), 128))

    keypoints:
    list of cv2.Keypoint
    """
    sift = cv2.SIFT()
    as_uint8 = img_01_to_uint8(img)
    # NOTE: points should be the same points as keypoints
    points, descriptors = sift.compute(as_uint8, keypoints)
    # TODO assertions that points and keypoints are the same points
    return descriptors


def brute_force_match(descriptors1, descriptors2):
    bfm = cv2.BFMatcher()
    matches = bfm.match(descriptors1, descriptors2)
    return matches


def k_nearest_neighbor_match(descriptors1, descriptors2, k):
    bfm = cv2.BFMatcher()
    matches = bfm.knnMatch(descriptors1, descriptors2, k)
    return matches


def ratio_test_match(descriptors1, descriptors2, ratio=0.75):
    """
    apply ratio test explained by D.Lowe in his paper

    http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html#matcher
    """
    matches = k_nearest_neighbor_match(descriptors1, descriptors2, 2)
    good = []
    for m1, m2 in matches:
        if m1.distance < m2.distance * ratio:
            good.append(m1)
    return good


def match_to_dict(match):
    """
    convert cv2.DMatch to a dict
    """
    # not using match.imgIdx
    return dict(
        idx1=match.queryIdx,
        idx2=match.trainIdx,
        distance=match.distance,
    )


def invert_affine_transform(affine_matrix):
    """
    takes in 2x3 matrix of affine transform
    """
    return cv2.invertAffineTransform(affine_matrix)


def warp_affine(img,
                affine_matrix,
                shape=None,
                interpolation="linear",
                border_mode="constant",
                fill_value=(0, 0, 0, 0),
                is_inverse_map=False):
    """
    cv2.warpAffine seems to be able to take unscaled images of multiple dtypes
    (tested uint8, float32, and float64)

    affine_matrix:
    2x3 matrix representing affine transform

    interpolation:
    see docs of str_to_interpolation

    border_mode:
    see docs of str_to_border_mode

    fill_value:
    value to fill with if border_mode is "constant"
    possible values include:
    - a single value
    - a tuple of length <= 4 (tuples of length < 4 are treated as tuples of
      length 4, but with the remaining values as 0)
      - in this case, the i-th element of the tuple is used for padding in the
        i-th channel of the image

    is_inverse_map:
    whether or not affine_matrix is the inverse transformation
    (setting this to true makes this function behave more like scikit-image's
    warp function)

    http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html#warpaffine
    """
    flags = str_to_interpolation(interpolation)
    border_mode_ = str_to_border_mode(border_mode)
    if is_inverse_map:
        flags |= cv2.WARP_INVERSE_MAP

    if shape is None:
        shape = img.shape[:2]
    x_shape, y_shape = shape[:2]
    res = cv2.warpAffine(img,
                         M=affine_matrix,
                         dsize=(y_shape, x_shape),
                         flags=flags,
                         borderMode=border_mode_,
                         borderValue=fill_value)
    if len(img.shape) == 3 and img.shape[2] == 1:
        # warp affine appears to convert a 3D image with last dimension 1
        # to a 2D image automatically, so convert it back here:
        return res[..., np.newaxis]
    else:
        return res


def remap(img, map_fn, interpolation="linear", output_shape=None):
    """
    applies a generic geometrical transformation to an image

    interpolation:
    see docs of str_to_interpolation

    http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html#remap
    """
    interpolation_ = str_to_interpolation(interpolation)

    if output_shape is None:
        output_shape = img.shape

    m, n = output_shape
    map_x = np.zeros(output_shape, dtype=np.float32)
    map_y = np.zeros(output_shape, dtype=np.float32)
    for x in range(m):
        for y in range(n):
            new_x, new_y = map_fn(x, y)
            map_x[x, y] = new_x
            map_y[x, y] = new_y

    img_uint8 = img_01_to_uint8(img)
    mapped_uint8 = cv2.remap(img_uint8,
                             map_y,
                             map_x,
                             interpolation=interpolation_)
    mapped = img_uint8_to_01(mapped_uint8)
    return mapped


def vidread(filename):
    """
    generates images instead of storing as a 3-tensor because the videos
    can take up a lot of memory

    based on:
    http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
    """
    cap = cv2.VideoCapture(filename)
    assert cap.isOpened()  # TODO also could call cap.open()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            yield img_uint8_to_01(frame)
        else:
            break
    cap.release()


class VideoWriter(object):

    """
    allows writing several images into a video

    example:
    vw = VideoWriter("foo.avi")
    vw.write(img1)
    vw.write(img2)
    vw.close()

    based on:
    http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
    """

    def __init__(self, filename, fps=20.0, codec="XVID"):
        self.filename = filename
        self.fps = fps
        self.codec = codec
        self.writer_ = None

    def write(self, img):
        if self.writer_ is None:
            # fourcc = cv2.VideoWriter_fourcc(*codec)
            fourcc = cv2.cv.CV_FOURCC(*self.codec)
            is_color = len(img.shape) == 3
            (h, w) = img.shape[:2]
            self.writer_ = cv2.VideoWriter(filename=self.filename,
                                           fourcc=fourcc,
                                           fps=self.fps,
                                           frameSize=(w, h),
                                           isColor=is_color)
        self.writer_.write(img_01_to_uint8(img))

    def close(self):
        if self.writer_ is not None:
            self.writer_.release()


@contextlib.contextmanager
def vidwriter(*args, **kwargs):
    """
    context manager that returns and closes a video writer

    example:
    with vidwriter("foo.avi") as vw:
        for img in imgs:
            vw.write(img)
    """
    w = VideoWriter(*args, **kwargs)
    try:
        yield w
    finally:
        w.close()


def vidwrite(filename, imgs, **kwargs):
    """
    writes a sequence of images into a video file
    """
    with vidwriter(filename, **kwargs) as w:
        for img in imgs:
            w.write(img)


def structuring_element(shape_str,
                        kernel_size):
    """
    returns a structuring_element with the specified shape, size, and anchor
    element
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * 2

    assert len(kernel_size) == 2

    if shape_str in ["circle", "square"]:
        assert kernel_size[0] == kernel_size[1]

    if shape_str in ["ellipse", "circle"]:
        morph_shape = cv2.MORPH_ELLIPSE
    elif shape_str in ["rect", "rectangle", "square"]:
        morph_shape = cv2.MORPH_RECT
    elif shape_str == "cross":
        morph_shape = cv2.MORPH_CROSS
    else:
        raise ValueError("Unknown shape string: %s" % shape_str)
    # NOTE: changing the anchor doesn't seem to do anything
    return cv2.getStructuringElement(morph_shape, kernel_size)


def morphological_transform(img,
                            operation,
                            kernel,
                            kernel_size=None):
    """
    applies a morphological transform

    kernel:
    either the kernel as a numpy array or a string representing the shape of
    the structuring element

    kernel_size:
    size of the created kernel (must be specified if kernel is given as a
    string)
    """
    if operation == "open":
        op = cv2.MORPH_OPEN
    elif operation == "close":
        op = cv2.MORPH_CLOSE
    elif operation == "dilate":
        op = cv2.MORPH_DILATE
    elif operation == "erode":
        op = cv2.MORPH_ERODE
    else:
        # TODO add all op's
        raise ValueError("Unknown operation: %s" % operation)

    if isinstance(kernel, types.StringTypes):
        assert kernel_size is not None
        kernel = structuring_element(kernel, kernel_size)

    return cv2.morphologyEx(img, op, kernel)


def moments(img, is_binary_img=False):
    """
    http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html

    is_binary_img:
    if True, treats all non-zero pixels as 1's
    """
    return cv2.moments(img, is_binary_img)
