import cupy as cp
import numpy as np


def segment_axis(
    x,
    length: int,
    shift: int,
    axis: int = -1,
    end="pad",
    pad_mode="constant",
    pad_value=0,
):
    """!!! WIP !!!

    ToDo: Discuss: Outsource conv_pad?

    Generate a new array that chops the given array along the given axis
    into overlapping frames.

    Note: if end='pad' the return is maybe a copy

    :param x: The array to segment
    :param length: The length of each frame
    :param shift: The number of array elements by which the frames should shift
        Negative values are also allowed.
    :param axis: The axis to operate on
    :param end:
        'pad' -> pad,
            pad the last block with zeros if necessary
        None -> assert,
            assume the length match, ensures a no copy
        'cut' -> cut,
            remove the last block if there are not enough values
        'conv_pad'
            special padding for convolution, assumes shift == 1, see example
            below

    :param pad_mode: see numpy.pad
    :param pad_value: The value to pad
    :return:

    """
    xp = cp.get_array_module(x)

    axis = axis % x.ndim

    # Implement negative shift with a positive shift and a flip
    # stride_tricks does not work correct with negative stride
    if shift > 0:
        do_flip = False
    elif shift < 0:
        do_flip = True
        shift = abs(shift)
    else:
        raise ValueError(shift)

    if pad_mode == "constant":
        pad_kwargs = {"constant_values": pad_value}
    else:
        pad_kwargs = {}

    # Pad
    if end == "pad":
        if x.shape[axis] < length:
            npad = np.zeros([x.ndim, 2], dtype=xp.int)
            npad[axis, 1] = length - x.shape[axis]
            x = xp.pad(x, pad_width=npad, mode=pad_mode, **pad_kwargs)
        elif shift != 1 and (x.shape[axis] + shift - length) % shift != 0:
            npad = np.zeros([x.ndim, 2], dtype=np.int)
            npad[axis, 1] = shift - ((x.shape[axis] + shift - length) % shift)
            x = xp.pad(x, pad_width=npad, mode=pad_mode, **pad_kwargs)

    elif end == "conv_pad":
        assert shift == 1, shift
        npad = np.zeros([x.ndim, 2], dtype=np.int)
        npad[axis, :] = length - shift
        x = xp.pad(x, pad_width=npad, mode=pad_mode, **pad_kwargs)
    elif end is None:
        assert (
            x.shape[axis] + shift - length
        ) % shift == 0, "{} = x.shape[axis]({}) + shift({}) - length({})) % shift({})" "".format(
            (x.shape[axis] + shift - length) % shift,
            x.shape[axis],
            shift,
            length,
            shift,
        )
    elif end == "cut":
        pass
    else:
        raise ValueError(end)

    # Calculate desired shape and strides
    shape = list(x.shape)
    # assert shape[axis] >= length, shape
    del shape[axis]
    shape.insert(axis, (x.shape[axis] + shift - length) // shift)
    shape.insert(axis + 1, length)

    strides = list(x.strides)
    strides.insert(axis, shift * strides[axis])

    try:
        x = xp.lib.stride_tricks.as_strided(x, strides=strides, shape=shape)

    except Exception:
        print("strides:", x.strides, " -> ", strides)
        print("shape:", x.shape, " -> ", shape)
        print("flags:", x.flags)
        print("Parameters:")
        print(
            "shift:",
            shift,
            "Note: negative shift is implemented with a " "following flip",
        )
        print("length:", length, "<- Has to be positive.")
        raise
    if do_flip:
        return xp.flip(x, axis=axis)
    else:
        return x


# http://stackoverflow.com/a/3153267
def roll_zeropad(a, shift, axis=None):
    """
    Roll array elements along a given axis.

    Elements off the end of the array are treated as zeros.

    Parameters
    ----------
    a : array_like
        Input array.
    shift : int
        The number of places by which elements are shifted.
    axis : int, optional
        The axis along which elements are shifted.  By default, the array
        is flattened before shifting, after which the original
        shape is restored.

    Returns
    -------
    res : ndarray
        Output array, with the same shape as `a`.

    See Also
    --------
    roll     : Elements that roll off one end come back on the other.
    rollaxis : Roll the specified axis backwards, until it lies in a
               given position.

    """
    if a.__class__.__module__ == "cupy.core.core":
        import cupy

        xp = cupy
    else:
        xp = np

    if shift == 0:
        return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if xp.abs(shift) > n:
        res = xp.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = xp.zeros_like(a.take(xp.arange(n - shift), axis))
        res = xp.concatenate((a.take(xp.arange(n - shift, n), axis), zeros), axis)
    else:
        zeros = xp.zeros_like(a.take(xp.arange(n - shift, n), axis))
        res = xp.concatenate((zeros, a.take(xp.arange(n - shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res
