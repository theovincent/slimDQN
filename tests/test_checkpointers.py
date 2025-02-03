# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/tests/dopamine/jax/checkpointers_test.py

"""Checkpointer test."""

from typing import Any, Dict

from absl.testing import absltest
from absl.testing import parameterized
from slimdqn.sample_collection import checkpointers
from etils import epath
import numpy as np
from orbax import checkpoint as orbax

from absl import flags

flags.FLAGS(["--test_tmpdir", "/tmpdir"])


class CheckpointTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.tmpdir = epath.Path(self.create_tempdir("checkpoint").full_path)

    def make_checkpoint_manager(self) -> orbax.CheckpointManager:
        return orbax.CheckpointManager(
            self.tmpdir,
            checkpointers=checkpointers.Checkpointer(),
        )

    def testInvalidCheckpointProtocol(self):
        """Test an invalid checkpointable."""

        class A:
            ...

        checkpoint_manager = self.make_checkpoint_manager()
        with self.assertRaisesRegex(NotImplementedError, "must implement Checkpointable$"):
            checkpoint_manager.save(0, A())

    @parameterized.parameters(
        (1,),
        (1.0,),
        (True,),
        (123456789101112131415161718,),
        ("string",),
        ({"key1": 1, "key2": True, "key3": "string"},),
        ([1, 1.0, True, "string"],),
        ([[1, 1.0, True, "string"], [2, 2.0, True, "string"]],),
        # Numpy array with different orders
        (np.array([1, 2, 3], order="F"),),
        (np.array([1, 2, 3], order="C"),),
        (np.array([1, 2, 3]),),
        (np.zeros((4, 4, 4), order="F"),),
        (np.zeros((4, 4, 4), order="C"),),
        (np.zeros((4, 4, 4)),),
        # Structured array
        (
            np.array(
                [("A", 1, 1.0), ("B", 2, 2.0)],
                dtype=[("string", "U1"), ("int", "i4"), ("float", "f4")],
            ),
        ),
    )
    def testCheckpointable(self, data: Any):
        """Test various data types with checkpointable."""

        class A:
            def to_state_dict(self) -> Dict[str, Any]:
                return {"data": data}

            def from_state_dict(self, state_dict: Dict[str, Any]) -> None:
                self.data = state_dict["data"]

        checkpoint_manager = self.make_checkpoint_manager()

        a = A()
        checkpoint_manager.save(0, a)

        # Restore to state dict
        restored = checkpoint_manager.restore(0)
        self.assertIsInstance(restored, dict)
        self.assertIn("data", restored)
        if isinstance(data, np.ndarray):
            np.testing.assert_array_equal(restored["data"], data)
        else:
            self.assertEqual(restored["data"], data)

        restored = checkpoint_manager.restore(0, a)
        self.assertIsInstance(restored, A)
        # Object should have been deep copied
        self.assertNotEqual(id(a), id(restored))
        assert hasattr(restored, "data")
        if isinstance(data, np.ndarray):
            np.testing.assert_array_equal(restored.data, data)
        else:
            self.assertEqual(restored.data, data)
