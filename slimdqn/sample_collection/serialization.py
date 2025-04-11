# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/dopamine/jax/serialization.py
"""MessagePack serialization hooks.

This is necesarry as Orbax doesn't support serializing Numpy structured arrays.
TensorStore (the underlying backend for Orbax) does support structured arrays.
If Orbax ever adds support for structured arrays, we can remove this altogether
and use Orbax to serialize things like the replay buffer without requiring
a custom checkpoint handler (jax/checkpointers.py).
"""

import functools
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from typing_extensions import Literal, NotRequired, TypedDict

Shape = Tuple[int, ...]


class NumpyEncoding(TypedDict, total=False):
    """Numpy encoding dictionary."""

    dtype: str
    shape: Shape
    data: bytes
    order: NotRequired[Literal["C", "F"]]


class LongIntegerEncoding(TypedDict):
    """Encoding for ints longer than 32-bits which MessagePack doesn't support."""

    integer: str


@functools.singledispatch
def encode(obj: Any, chain: Optional[Callable[[Any], Any]] = None) -> Any:
    """Encode object. Encoders will register with `@encode.register`."""
    return obj if chain is None else chain(obj)


@encode.register(np.ndarray)
@encode.register(np.bool_)
@encode.register(np.number)
def _(
    array: Union[np.ndarray, np.bool_, np.number],
    chain: Optional[Callable[[Any], Any]] = None,
) -> NumpyEncoding:
    """Encode numpy array."""
    del chain

    encoded: NumpyEncoding = {"shape": array.shape, "data": array.tobytes()}

    # Encode the dtype using repr and parse it on decode using `ast.literal_eval`
    if array.dtype.kind == "V":
        encoded["dtype"] = repr(array.dtype.descr)
    else:
        encoded["dtype"] = repr(array.dtype.str)

    # Store the order if we have one
    if array.flags["C_CONTIGUOUS"] and not array.flags["F_CONTIGUOUS"]:
        encoded["order"] = "C"
    elif not array.flags["C_CONTIGUOUS"] and array.flags["F_CONTIGUOUS"]:
        encoded["order"] = "F"

    return encoded


@encode.register
def _(
    integer: int,
    chain: Optional[Callable[[Any], Any]] = None,
) -> Union[int, LongIntegerEncoding]:
    """Encode integers longer than 32 bit as a string."""
    del chain

    if integer.bit_length() > 32:
        return {"integer": str(integer)}

    return integer
