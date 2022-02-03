import re
import numpy as np
from numpy.core.einsumfunc import _parse_einsum_input


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

    >>> # import cupy as np
    >>> segment_axis(np.arange(10), 4, 2)  # simple example
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])
    >>> segment_axis(np.arange(10), 4, -2)  # negative shift
    array([[6, 7, 8, 9],
           [4, 5, 6, 7],
           [2, 3, 4, 5],
           [0, 1, 2, 3]])
    >>> segment_axis(np.arange(5).reshape(5), 4, 1, axis=0)
    array([[0, 1, 2, 3],
           [1, 2, 3, 4]])
    >>> segment_axis(np.arange(5).reshape(5), 4, 2, axis=0, end='cut')
    array([[0, 1, 2, 3]])
    >>> segment_axis(np.arange(5).reshape(5), 4, 2, axis=0, end='pad')
    array([[0, 1, 2, 3],
           [2, 3, 4, 0]])
    >>> segment_axis(np.arange(5).reshape(5), 4, 1, axis=0, end='conv_pad')
    array([[0, 0, 0, 0],
           [0, 0, 0, 1],
           [0, 0, 1, 2],
           [0, 1, 2, 3],
           [1, 2, 3, 4],
           [2, 3, 4, 0],
           [3, 4, 0, 0],
           [4, 0, 0, 0]])
    >>> segment_axis(np.arange(6).reshape(6), 4, 2, axis=0, end='pad')
    array([[0, 1, 2, 3],
           [2, 3, 4, 5]])
    >>> segment_axis(np.arange(10).reshape(2, 5), 4, 1, axis=-1)
    array([[[0, 1, 2, 3],
            [1, 2, 3, 4]],
    <BLANKLINE>
           [[5, 6, 7, 8],
            [6, 7, 8, 9]]])
    >>> segment_axis(np.arange(10).reshape(5, 2).T, 4, 1, axis=1)
    array([[[0, 2, 4, 6],
            [2, 4, 6, 8]],
    <BLANKLINE>
           [[1, 3, 5, 7],
            [3, 5, 7, 9]]])
    >>> segment_axis(np.asfortranarray(np.arange(10).reshape(2, 5)),
    ...                 4, 1, axis=1)
    array([[[0, 1, 2, 3],
            [1, 2, 3, 4]],
    <BLANKLINE>
           [[5, 6, 7, 8],
            [6, 7, 8, 9]]])
    >>> segment_axis(np.arange(8).reshape(2, 2, 2).transpose(1, 2, 0),
    ...                 2, 1, axis=0, end='cut')
    array([[[[0, 4],
             [1, 5]],
    <BLANKLINE>
            [[2, 6],
             [3, 7]]]])
    >>> a = np.arange(7).reshape(7)
    >>> b = segment_axis(a, 4, -2, axis=0, end='cut')
    >>> a += 1  # a and b point to the same memory
    >>> b
    array([[3, 4, 5, 6],
           [1, 2, 3, 4]])

    >>> segment_axis(np.arange(7), 8, 1, axis=0, end='pad').shape
    (1, 8)
    >>> segment_axis(np.arange(8), 8, 1, axis=0, end='pad').shape
    (1, 8)
    >>> segment_axis(np.arange(9), 8, 1, axis=0, end='pad').shape
    (2, 8)
    >>> segment_axis(np.arange(7), 8, 2, axis=0, end='cut').shape
    (0, 8)
    >>> segment_axis(np.arange(8), 8, 2, axis=0, end='cut').shape
    (1, 8)
    >>> segment_axis(np.arange(9), 8, 2, axis=0, end='cut').shape
    (1, 8)

    >>> x = np.arange(1, 10)
    >>> filter_ = np.array([1, 2, 3])
    >>> np.convolve(x, filter_)
    array([ 1,  4, 10, 16, 22, 28, 34, 40, 46, 42, 27])
    >>> x_ = segment_axis(x, len(filter_), 1, end='conv_pad')
    >>> x_
    array([[0, 0, 1],
           [0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5],
           [4, 5, 6],
           [5, 6, 7],
           [6, 7, 8],
           [7, 8, 9],
           [8, 9, 0],
           [9, 0, 0]])
    >>> x_ @ filter_[::-1]  # Equal to convolution
    array([ 1,  4, 10, 16, 22, 28, 34, 40, 46, 42, 27])
    """

    if x.__class__.__module__ == "cupy.core.core":
        import cupy

        xp = cupy
    else:
        xp = np

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
            npad = np.zeros([x.ndim, 2], dtype=np.int)
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

    # Alternative to np.ndarray.__new__
    # I am not sure if np.lib.stride_tricks.as_strided is better.
    # return np.lib.stride_tricks.as_strided(
    #     x, shape=shape, strides=strides)
    try:
        if xp == np:
            x = np.lib.stride_tricks.as_strided(x, strides=strides, shape=shape)
        else:
            x = x.view()
            x._set_shape_and_strides(strides=strides, shape=shape)
        # return np.ndarray.__new__(np.ndarray, strides=strides,
        #                           shape=shape, buffer=x, dtype=x.dtype)
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


def _normalize(op):
    op = op.replace(",", "")
    op = op.replace(" ", "")
    op = " ".join(c for c in op)
    op = op.replace(" * ", "*")
    op = op.replace("- >", "->")
    op = op.replace(". . .", "...")
    return op


def _shrinking_reshape(array, source, target):
    source, target = source.split(), target.replace(" * ", "*").split()

    if "..." in source:
        assert "..." in target, (source, target)
        independent_dims = array.ndim - len(source) + 1
        import string

        ascii_letters = [
            s for s in string.ascii_letters if s not in source and s not in target
        ]
        index = source.index("...")
        source[index : index + 1] = ascii_letters[:independent_dims]
        index = target.index("...")
        target[index : index + 1] = ascii_letters[:independent_dims]

    input_shape = {key: array.shape[index] for index, key in enumerate(source)}

    output_shape = []
    for t in target:
        product = 1
        if not t == "1":
            t = t.split("*")
            for t_ in t:
                product *= input_shape[t_]
        output_shape.append(product)

    return array.reshape(output_shape)


def _expanding_reshape(array, source, target, **shape_hints):

    try:  # Check number of inputs for unflatten operations
        assert len(re.sub(r".\*", "", source.replace(" ", ""))) == array.ndim, (
            array.shape,
            source,
            target,
        )
    except AssertionError:  # Check number of inputs for ellipses operations
        assert (
            len(re.sub(r"(\.\.\.)|(.\*)", "", source.replace(" ", ""))) <= array.ndim
        ), (array.shape, source, target)

    def _get_source_grouping(source):
        """
        Gets axis as alphanumeric.
        """

        source = " ".join(source)
        source = source.replace(" * ", "*")
        groups = source.split()
        groups = [group.split("*") for group in groups]
        return groups

    if "*" not in source:
        return array

    source, target = source.split(), target.replace(" * ", "*").split()

    if "..." in source:
        assert "..." in target, (source, target)
        independent_dims = array.ndim - len(source) + 1
        import string

        ascii_letters = [
            s for s in string.ascii_letters if s not in source and s not in target
        ]
        index = source.index("...")
        source[index : index + 1] = ascii_letters[:independent_dims]
        index = target.index("...")
        target[index : index + 1] = ascii_letters[:independent_dims]

    target_shape = []

    for axis, group in enumerate(_get_source_grouping(source)):
        if len(group) == 1:
            target_shape.append(array.shape[axis : axis + 1])
        else:
            shape_wildcard_remaining = True
            for member in group:
                if member in shape_hints:
                    target_shape.append([shape_hints[member]])
                else:
                    if shape_wildcard_remaining:
                        shape_wildcard_remaining = False
                        target_shape.append([-1])
                    else:
                        raise ValueError("Not enough shape hints provided.")

    target_shape = np.concatenate(target_shape, 0)
    array = array.reshape(target_shape)
    return array


def morph(operation, array, reduce=None, **shape_hints):
    """This is an experimental version of a generalized reshape.
    See test cases for examples.
    """
    operation = _normalize(operation)
    source, target = operation.split("->")

    # Expanding reshape
    array = _expanding_reshape(array, source, target, **shape_hints)

    # Initial squeeze
    squeeze_operation = operation.split("->")[0].split()
    for axis, op in reversed(list(enumerate(squeeze_operation))):
        if op == "1":
            array = np.squeeze(array, axis=axis)

    # Transpose
    transposition_operation = operation.replace("1", " ").replace("*", " ")
    try:
        in_shape, out_shape, (array,) = _parse_einsum_input(
            [transposition_operation.replace(" ", ""), array]
        )

        if len(set(in_shape) - set(out_shape)) > 0:
            assert reduce is not None, (
                "Missing reduce function",
                reduce,
                transposition_operation,
            )

            reduce_axis = tuple(
                [i for i, s in enumerate(in_shape) if s not in out_shape]
            )
            array = reduce(array, axis=reduce_axis)
            in_shape = "".join([s for s in in_shape if s in out_shape])

        array = np.einsum(f"{in_shape}->{out_shape}", array)
    except ValueError as e:
        msg = (
            f"op: {transposition_operation} ({in_shape}->{out_shape}), "
            f"shape: {np.shape(array)}"
        )

        if len(e.args) == 1:
            e.args = (e.args[0] + "\n\n" + msg,)
        else:
            print(msg)
        raise

    # Final reshape
    source = transposition_operation.split("->")[-1]
    target = operation.split("->")[-1]

    return _shrinking_reshape(array, source, target)
