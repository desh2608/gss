# The functions in this file are modified from:
# https://github.com/fgnt/pb_chime5/blob/master/pb_chime5/utils/numpy_utils.py

import re

import cupy as cp
from numpy.core.einsumfunc import _parse_einsum_input


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

    target_shape = cp.concatenate(target_shape, 0)
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
            array = cp.squeeze(array, axis=axis)

    # Transpose
    transposition_operation = operation.replace("1", " ").replace("*", " ")
    try:
        array = cp.asnumpy(array)
        in_shape, out_shape, (array,) = _parse_einsum_input(
            [transposition_operation.replace(" ", ""), array]
        )
        array = cp.asarray(array)

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

        array = cp.einsum(f"{in_shape}->{out_shape}", array)
    except ValueError as e:
        msg = (
            f"op: {transposition_operation} ({in_shape}->{out_shape}), "
            f"shape: {cp.shape(array)}"
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
