from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals


import re
import numpy as np
import dicom

try:
    import gdcm
    GDCM_IS_AVAILABLE = True
except ImportError:
    GDCM_IS_AVAILABLE = False

# ################################ simple itk ################################


def sitk_read_dicom_dir_as_volume(dirname):
    """
    https://pyscience.wordpress.com/2014/10/19/image-segmentation-with-python-and-simpleitk/
    """
    import SimpleITK
    reader = SimpleITK.ImageSeriesReader()
    filenamesDICOM = reader.GetGDCMSeriesFileNames(dirname)
    reader.SetFileNames(filenamesDICOM)
    imgOriginal = reader.Execute()
    img = SimpleITK.GetArrayFromImage(imgOriginal).swapaxes(0, 2)
    return img

# ################################# pydicom #################################


def convert_dicom_value(val, pixel_reader="pydicom"):
    """
    Returns 'val' converted from a dicom-specific representation
    to a python primitive type.

    known issues with pydicom:
    - ValueError: invalid literal for int() with base 10
    """
    if isinstance(val, dicom.valuerep.IS):
        return int(val)
    elif isinstance(val, dicom.valuerep.DSfloat):
        return float(val)
    elif isinstance(val, dicom.multival.MultiValue):
        return map(lambda x: convert_dicom_value(x, pixel_reader), val)
    elif isinstance(val, dicom.tag.BaseTag):
        return {"group": val.group,
                "element": val.element}
    elif isinstance(val, dicom.dataset.Dataset):
        res = {}
        for key in val.dir():
            if key == "PixelData":
                if pixel_reader is None:
                    pass
                elif pixel_reader == "pydicom":
                    res[key] = val.pixel_array
                else:
                    raise ValueError("Unknown pixel_reader value: {}"
                                     .format(pixel_reader))
            else:
                try:
                    data_element = val.data_element(key)
                except Exception as e:
                    # handle case of:
                    # "invalid literal for int() with base 10: '5.000000'"
                    if re.match(r"^invalid literal for int\(\) with base 10: '\d+.0*'$",
                                str(e)):
                        # TODO parse float value out
                        continue
                    else:
                        raise e
                if data_element is not None:
                    res[key] = convert_dicom_value(data_element.value,
                                                   pixel_reader)
                else:
                    res[key] = None
        return res
    elif isinstance(val, np.ndarray):
        return val
    # need to check the dicom values before these, because the dicom values
    # are subclasses
    elif isinstance(val, (int, float, str)):
        return val
    elif isinstance(val, list):
        return map(lambda x: convert_dicom_value(x, pixel_reader), val)
    else:
        raise ValueError

# ################################### gdcm ###################################


def gdcm_read_file(filename):
    """
    mostly copied from http://gdcm.sourceforge.net/html/ConvertNumpy_8py-example.html
    """
    def get_gdcm_to_numpy_typemap():
        """Returns the GDCM Pixel Format to numpy array type mapping."""
        # NOTE: 20160121 the link above had uint8 and int8 swapped
        _gdcm_np = {gdcm.PixelFormat.UINT8: np.uint8,
                    gdcm.PixelFormat.INT8: np.int8,
                    # gdcm.PixelFormat.UINT12 :np.uint12,
                    # gdcm.PixelFormat.INT12  :np.int12,
                    gdcm.PixelFormat.UINT16: np.uint16,
                    gdcm.PixelFormat.INT16: np.int16,
                    gdcm.PixelFormat.UINT32: np.uint32,
                    gdcm.PixelFormat.INT32: np.int32,
                    # gdcm.PixelFormat.FLOAT16:np.float16,
                    gdcm.PixelFormat.FLOAT32: np.float32,
                    gdcm.PixelFormat.FLOAT64: np.float64}
        return _gdcm_np

    def get_numpy_array_type(gdcm_pixel_format):
        """Returns a numpy array typecode given a GDCM Pixel Format."""
        return get_gdcm_to_numpy_typemap()[gdcm_pixel_format]

    def gdcm_to_numpy(image):
        """Converts a GDCM image to a numpy array.
        """
        pf = image.GetPixelFormat()

        assert pf.GetScalarType() in get_gdcm_to_numpy_typemap().keys(), \
            "Unsupported array type %s" % pf

        dims = tuple(image.GetDimensions())
        # note if len(dims) == 3, image shape is num_images x height x width
        # and order of dims is width x height x num_images
        # num_images seems to be indicated by the NumberOfFrames field
        assert len(dims) in (2, 3)
        samples_per_pixel = int(pf.GetSamplesPerPixel())
        # matching how pydicom outputs the image
        if samples_per_pixel == 1:
            shape = dims[::-1]
        else:
            shape = (samples_per_pixel,) + dims[::-1]

        # HACK this doesn't work
        # shape = image.GetDimension(
        #     0) * image.GetDimension(1), pf.GetSamplesPerPixel()
        # if image.GetNumberOfDimensions() == 3:
        #     shape = shape[0] * image.GetDimension(2), shape[1]

        dtype = get_numpy_array_type(pf.GetScalarType())
        gdcm_array = image.GetBuffer()
        result = np.frombuffer(gdcm_array, dtype=dtype)
        result = result.reshape(shape)
        return result

    r = gdcm.ImageReader()
    # gdcm doesn't like unicode strings
    filename = str(filename)
    r.SetFileName(filename)
    if not r.Read():
        raise Exception("Something went wrong with gdcm read")

    numpy_array = gdcm_to_numpy(r.GetImage())
    return numpy_array

# ################## utilities for dealing with dicom maps ##################


def _dcm_format_pixels(dcm, pixel_format):
    """
    attempts to convert a dcm file into a uniform format
    """
    pixel_data = dcm["PixelData"]
    if pixel_format == "pydicom":
        # default format is "pydicom"
        # ---
        # - images without (color)channels are 2-dimensional
        # - images with channels are 3-dimensional with the channel as the
        #   first dimension
        pass
    elif pixel_format == "always_2d":
        # throws an exception for 3d pixels
        # FIXME
        # FIXME could filter for NumberOfFrames first, since dicom files
        # with multiple images can take 100s of times more than the average
        # file
        assert False
    elif pixel_format == "make_3d":
        # adds an extra dimension for 2d images
        if pixel_data.ndim == 2:
            pixel_data = pixel_data[np.newaxis]
    elif pixel_format == "make_2d":
        # FIXME
        # convert RGB to gray
        # if img.shape[0] != 3, it might be an image with extra channels
        assert False
    elif pixel_format == "make_rgb":
        # FIXME
        assert False
    else:
        raise ValueError("Unknown pixel_format: {}".format(pixel_format))
    dcm["PixelData"] = pixel_data
    return dcm


def _dcm_convert_pixels(dcm, mode):
    """
    uses dicom information to convert pixel values from dicom-internal
    representation to other ones

    mode:
    - scale_bits_stored : scales based on the bits stored in the representation
    - window
    - rescale
    - rescale+window
    - try_window : uses the window parameters if available, and takes
        the first window parameters if there are multiple, and if not
        available, reverts to scale_bits_stored
    """
    def clip01(x):
        return np.clip(x, 0., 1.)

    def maybe_flip_intensity(x):
        # HACK - how do we flip for non-01 values?
        assert x.min() >= 0.
        assert x.max() <= 1.
        if ("PhotometricInterpretation" in dcm and
                dcm["PhotometricInterpretation"] == "MONOCHROME1"):
            return 1.0 - x
        else:
            return x

    def apply_scale_bits_stored(x):
        x = x / (2.0 ** dcm['BitsStored'] - 1)
        # some dicom files have values outside of the range of BitsStored,
        # so we must clip to ensure the range (as opposed to assert-ing)
        x = clip01(x)
        x = maybe_flip_intensity(x)
        return x

    def apply_rescale(x):
        rs = dcm["RescaleSlope"]
        ri = dcm["RescaleIntercept"]
        return x * rs + ri

    def apply_window(x):
        ww = dcm["WindowWidth"]
        wc = dcm["WindowCenter"]

        if isinstance(ww, list):
            # as a heurstic, take the first
            # see: https://www.dabsoft.ch/dicom/3/C.11.2.1.2/
            # "If multiple values are present, both Attributes shall have
            # the same number of values and shall be considered as pairs.
            # Multiple values indicate that multiple alternative views
            # may be presented."
            assert len(ww) == len(wc)
            # FIXME return multiple views instead
            ww = ww[0]
            wc = wc[0]

        x = (x - float(wc)) / float(ww)
        # recenter to [0, 1]
        x += 0.5
        x = clip01(x)
        return x

    data = dcm['PixelData']
    if mode == "scale_bits_stored":
        data = apply_scale_bits_stored(data)
    elif mode == "window":
        data = apply_window(data)
    elif mode == "rescale":
        data = apply_rescale(data)
    elif mode == "rescale+window":
        data = apply_rescale(data)
        data = apply_window(data)
    elif mode == "try_window":
        if "WindowWidth" in dcm and "WindowCenter" in dcm:
            data = apply_window(data)
        else:
            data = apply_scale_bits_stored(data)
    else:
        raise ValueError("Incorrect mode: %s" % mode)
    return data


def dcm_read_file(filename,
                  pixel_reader="try_gdcm",
                  convert_pixels=None,
                  pixel_format=None,
                  **kwargs):
    """
    returns a tuple of list of dicom maps and an error (or None if no error
    has occurred)
    """
    assert pixel_reader in {"gdcm", "pydicom", "try_gdcm", None}
    if pixel_reader == "try_gdcm":
        pixel_reader = "gdcm" if GDCM_IS_AVAILABLE else "pydicom"

    # ################################ read dicom #############################
    use_gdcm = pixel_reader == "gdcm"

    if use_gdcm:
        pixel_reader = None
        if "stop_before_pixels" not in kwargs:
            kwargs["stop_before_pixels"] = True

    f = dicom.read_file(filename, **kwargs)
    dcm = convert_dicom_value(f, pixel_reader)

    if use_gdcm:
        dcm["PixelData"] = gdcm_read_file(filename)

    # ############################### postprocess ############################
    if pixel_format is not None:
        # post process pixel format
        dcm = _dcm_format_pixels(dcm, pixel_format=pixel_format)
    if convert_pixels is not None:
        # post process pixel output
        dcm["PixelData"] = _dcm_convert_pixels(dcm, mode=convert_pixels)

    return [dcm], None


def dicom_read_file(filename,
                    pixel_reader="pydicom",
                    convert_pixels=None,
                    pixel_format=None,
                    **kwargs):
    """
    TODO a better interface would be one that returns a sequence of dicom maps
    such that this could return multiple images (?)
    this would also allow us to have flags for handling common error cases with
    the list monad
    eg.
    - "invalid literal for int() with base 10"
    - "Something went wrong with gdcm read"
    """
    import warnings
    warnings.warn("dicom_read_file is deprecated for dcm_read_file")
    dcms, err = dcm_read_file(filename=filename,
                              pixel_reader=pixel_reader,
                              convert_pixels=convert_pixels,
                              pixel_format=pixel_format,
                              **kwargs)
    if err is not None:
        raise err
    else:
        assert len(dcms) == 1
        return dcms[0]
