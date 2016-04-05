import itertools
import numpy as np
import du


def block_apply_with_border(arr,
                            fn,
                            block_shape,
                            border,
                            batch_size=None,
                            dtype=float,
                            fill_value=0):
    """
    divides an ndarray into smaller optionally overlapping blocks,
    applies the given function to each block, and reassembles the outputs
    into a block of the same size as the input

    use cases:
    - applying fully convolutional networks to large ndarrays

    arr: array to apply fn to

    fn: function that takes in block with shape block_shape + border and
    returns block with same shape as input

    block_shape: size of the block to be passed to fn

    border: adds zeros/additional context around the block whose results
    are not stored

    batch_size: if not None, adds an additional leading dimension with the
    specified size
    """
    assert isinstance(arr, np.ndarray)
    assert isinstance(block_shape, tuple)
    assert isinstance(border, tuple)
    assert len(block_shape) == len(border) == arr.ndim
    assert batch_size is None or isinstance(batch_size, int)

    shape = arr.shape
    input_shape = tuple([bs + 2 * bd for bs, bd in zip(block_shape, border)])

    # iterate over possible corners, while also calculate their index
    # into both the result, and the intermediate block (output of the fn)
    axis_num_choices = []
    axis_c = []
    axis_r = []
    axis_i = []
    for bd, bs, s in zip(border, block_shape, shape):
        c_s = []
        r_s = []
        i_s = []

        # starting index of the result
        # ---
        # begin at 0
        r_start = 0
        # starting index of the intermediate
        # ---
        # this always starts at the end of the border of that block
        i_start = bd
        while r_start < s:
            # ending index of the result
            # ---
            # end at lower of the start + block or the end of result
            r_end = min(r_start + bs, s)
            # ending index of the intermediate
            # ---
            # have the same length as result slice
            i_end = bd + r_end - r_start
            # corner of the orignal array
            # ---
            # always be border from r_start
            c = r_start - bd

            # save calculated values
            c_s.append(c)
            r_s.append(slice(r_start, r_end))
            i_s.append(slice(i_start, i_end))

            # loop
            r_start = r_end

        axis_num_choices.append(len(c_s))
        axis_c.append(c_s)
        axis_r.append(r_s)
        axis_i.append(i_s)

    def data_gen():
        for axis_idxs in itertools.product(
                *[range(n) for n in axis_num_choices]):

            corner = []
            result_slices = []
            intermediate_slices = []
            for axis, idx in enumerate(axis_idxs):
                corner.append(axis_c[axis][idx])
                result_slices.append(axis_r[axis][idx])
                intermediate_slices.append(axis_i[axis][idx])
            corner = tuple(corner)
            result_slices = tuple(result_slices)
            intermediate_slices = tuple(intermediate_slices)

            block = du.preprocessing.image.get_block_with_corner_and_shape(
                data=arr,
                corner=corner,
                shape=input_shape,
                fill_value=fill_value,
            )
            yield block, intermediate_slices, result_slices

    result = np.zeros(shape, dtype=dtype)
    if batch_size is None:
        for block, i_slices, r_slices in data_gen():
            intermediate = fn(block)
            result[r_slices] = intermediate[i_slices]
    else:
        gen = data_gen()
        while True:
            data = list(du.toolz.take(batch_size, gen))
            if not data:
                break

            blocks, iss, rss = zip(*data)
            blocks = np.array(blocks)
            intermediates = fn(blocks)
            for intermediate, i_slices, r_slices in zip(intermediates,
                                                        iss,
                                                        rss):
                result[r_slices] = intermediate[i_slices]

    return result
