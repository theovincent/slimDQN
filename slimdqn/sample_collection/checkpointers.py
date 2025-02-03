# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/dopamine/jax/checkpointers.py
"""Checkpointer using Orbax and MessagePack.

Usage:
  1) Implement `to_state_dict` and `from_state_dict` on your class.
  2) Create an `orbax.CheckpointManager` and use the `CheckpointHandler`.

  To save: call `save` on your checkpoint manager passing the class instance.
  To restore: call `restore` on your checkpoint manager passing the class
              instance.

"""

import copy
import functools
import typing
from typing import Any, Dict, Generic, Optional, Protocol, TypeVar, Union

from slimdqn.sample_collection import serialization
from etils import epath
import msgpack
from orbax import checkpoint


@typing.runtime_checkable
class Checkpointable(Protocol):
    """Checkpointable protocol. Must implement to_state_dict, from_state_dict."""

    def to_state_dict(self) -> Dict[str, Any]:
        ...

    def from_state_dict(self, state_dict: Dict[str, Any]) -> None:
        ...


CheckpointableT = TypeVar("CheckpointableT", bound=Checkpointable)


class CheckpointHandler(checkpoint.CheckpointHandler, Generic[CheckpointableT]):
    """Checkpointable protocol checkpoint handler."""

    def __init__(self, filename: str = "checkpoint.msgpack") -> None:
        self._filename = filename

    def save(self, directory: epath.Path, item: CheckpointableT) -> None:
        if not isinstance(item, Checkpointable):
            raise NotImplementedError(f"Item {item!r} must implement Checkpointable")
        directory.mkdir(exist_ok=True, parents=True)
        filename = directory / self._filename

        # Get bytes using MsgPack
        packed = msgpack.packb(
            item.to_state_dict(),
            default=serialization.encode,
            strict_types=False,
            use_bin_type=True,
        )
        filename.write_bytes(packed)

    def restore(
        self, directory: epath.Path, item: Optional[CheckpointableT] = None
    ) -> Union[CheckpointableT, Dict[str, Any]]:
        filename = directory / self._filename
        state_dict = msgpack.unpackb(
            filename.read_bytes(),
            object_hook=serialization.decode,
            raw=False,
            strict_map_key=False,
        )

        if item is None:
            return state_dict

        item = copy.deepcopy(item)
        item.from_state_dict(state_dict)
        return item

    def structure(self, directory: epath.Path) -> None:
        return None


# pylint: disable=g-long-lambda
# Orbax requires a `Checkpointer` object. This wraps a `CheckpointHandler`.
# To make it easier for end-users we'll supply a `Checkpointer` that already
# performs this instantiation.
Checkpointer = functools.partial(checkpoint.Checkpointer, CheckpointHandler())
# pylint: enable=g-long-lambda
