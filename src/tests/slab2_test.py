import unittest
import src.data.slab2 as slab2
from src.data.slab2 import Slab
import numpy as np


class TestSlab(unittest.TestCase):
    """A set of unit tests for the Slab class."""

    def setUp(self) -> None:
        return super().setUp()

    def test_all_slabs(self):
        """Test that all slab names are valid."""
        for key, values in slab2.ALL_SLABS.items():
            print("Testing: ", key)
            slab = Slab(key)
            self.assertEqual(slab.name, key)
            self.assertTrue(slab.file.exists())
            self.assertTrue(slab.raw_xyz is not None)
            self.assertTrue(slab.raw_xyz.shape[1] == 3)
            self.assertTrue(len(slab.longitude) == len(slab.latitude))

    def test_distance(self):
        slab = Slab("van")
        self.assertTrue(slab.distance([0, 0, 0]) > 0)
        self.assertTrue(slab.distance([0, 0, 0]) < 3 * 10e6)
        self.assertTrue(np.all(slab.distance([[0, 0, 0], [0, 0, 0]]) > 0))
        self.assertTrue(np.all(slab.distance([[0, 0, 0], [0, 0, 0]]) < 3 * 10e6))
        self.assertTrue(
            slab.distance([-1, 1, 10], from_latlon=True, depth_unit="km") > 0
        )


# %%
if __name__ == "__main__":
    unittest.main()
