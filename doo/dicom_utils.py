from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals


import re
import dicom
import numpy as np


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


def gdcm_read_file(filename):
    """
    mostly copied from http://gdcm.sourceforge.net/html/ConvertNumpy_8py-example.html
    """
    import gdcm
    import numpy

    def get_gdcm_to_numpy_typemap():
        """Returns the GDCM Pixel Format to numpy array type mapping."""
        # NOTE: 20160121 the link above had uint8 and int8 swapped
        _gdcm_np = {gdcm.PixelFormat.UINT8: numpy.uint8,
                    gdcm.PixelFormat.INT8: numpy.int8,
                    # gdcm.PixelFormat.UINT12 :numpy.uint12,
                    # gdcm.PixelFormat.INT12  :numpy.int12,
                    gdcm.PixelFormat.UINT16: numpy.uint16,
                    gdcm.PixelFormat.INT16: numpy.int16,
                    gdcm.PixelFormat.UINT32: numpy.uint32,
                    gdcm.PixelFormat.INT32: numpy.int32,
                    # gdcm.PixelFormat.FLOAT16:numpy.float16,
                    gdcm.PixelFormat.FLOAT32: numpy.float32,
                    gdcm.PixelFormat.FLOAT64: numpy.float64}
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
        result = numpy.frombuffer(gdcm_array, dtype=dtype)
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


def dicom_read_file(filename, **kwargs):
    """
    TODO a better interface would be one that returns a sequence of dicom maps
    such that this could return multiple images (?)
    this would also allow us to have flags for handling common error cases with
    the list monad
    eg.
    - "invalid literal for int() with base 10"
    - "Something went wrong with gdcm read"
    """

    pixel_reader = kwargs.pop("pixel_reader", "pydicom")
    convert_pixels_mode = kwargs.pop("convert_pixels", None)
    pixel_format = kwargs.pop("pixel_format", "pydicom")

    use_gdcm = pixel_reader == "gdcm"

    if use_gdcm:
        pixel_reader = None
        if "stop_before_pixels" not in kwargs:
            kwargs["stop_before_pixels"] = True

    f = dicom.read_file(filename, **kwargs)
    res = convert_dicom_value(f, pixel_reader)
    if use_gdcm:
        res["PixelData"] = gdcm_read_file(filename)
    if convert_pixels_mode is not None:
        res["PixelData"] = convert_pixels(res, mode=convert_pixels_mode)
    if pixel_reader is not None:
        pixel_data = res["PixelData"]
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
        res["PixelData"] = pixel_data
    return res


def read_dicom_dir_as_volume(dirname):
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


def convert_pixels(dicom_map, mode="default"):
    """
    Attempts to convert pixel values from dicom-internal representation to
    values in a [0.0,1.0] range, with 1 representing white and 0 black.

    mode:
    - default : defaults to try_window
    - try_window : uses the window parameters if available, and takes
        the first window parameters if there are multiple, and if not
        available, reverts to scale_bits_stored
    - scale_bits_stored : scales based on the bits stored in the representation
    """
    if mode == "default":
        mode = "try_window"

    assert mode in {"try_window", "scale_bits_stored"}

    data = dicom_map['PixelData']
    if mode == "try_window" and "WindowWidth" in dicom_map:
        # WindowCenter must be present if WindowWidth is
        assert "WindowCenter" in dicom_map
        ww = dicom_map["WindowWidth"]
        wc = dicom_map["WindowCenter"]

        if isinstance(ww, list):
            # as a heurstic, take the first
            # see: https://www.dabsoft.ch/dicom/3/C.11.2.1.2/
            # "If multiple values are present, both Attributes shall have
            # the same number of values and shall be considered as pairs.
            # Multiple values indicate that multiple alternative views
            # may be presented."
            ww = ww[0]
            wc = wc[0]

        data = (data - float(wc)) / float(ww)
        # recenter to [0, 1]
        data += 0.5
    else:
        # by default, use BitsStored
        data = data / (2.0 ** dicom_map['BitsStored'] - 1)

    # make sure data is in [0, 1]
    # ---
    # NOTE: some dicom files have values outside of the range of BitsStored,
    # so we must clip to ensure the range (as opposed to assert-ing)
    data = np.clip(data, 0., 1.)

    if ('PhotometricInterpretation' in dicom_map
            and dicom_map['PhotometricInterpretation'] == 'MONOCHROME1'):
        data = 1.0 - data

    return data
