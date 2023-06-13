import unittest
import src.data.slab2 as slab2
from src.data.slab2 import Slab
import numpy as np


class TestSlab(unittest.TestCase):
    """A set of unit tests for the Slab class."""

    def test_all_slabs(self):
        """Test that all slab names are valid."""
        for key, values in slab2.ALL_SLABS.items():
            print("Testing: ", key)
            slab = Slab(key)
            self.assertEqual(slab.name, key)
            self.assertTrue(slab.file.exists())
            self.assertTrue(slab.raw_xyz is not None)
            self.assertTrue(slab.raw_xyz.shape[1] == 3)
            self.assertTrue(slab.utm_geometry.shape[1] == 3)

    def test_distance(self):
        slab = Slab("van")
        self.assertTrue(slab.distance([0, 0, 0]) > 0)
        self.assertTrue(slab.distance([0, 0, 0]) < 3 * 10e6)
        self.assertTrue(np.all(slab.distance([[0, 0, 0], [0, 0, 0]]) > 0))
        self.assertTrue(np.all(slab.distance([[0, 0, 0], [0, 0, 0]]) < 3 * 10e6))
        self.assertTrue(
            slab.distance([-1, 1, 10], from_latlon=True, depth_unit="km") > 0
        )

    def test_properties(self):
        """Test that all slab propetries exist for"""
        print("Testing all properties...")
        for region in slab2.ALL_SLABS.keys():
            for property_key, property in slab2.SLAB_PROPERTIES.items():
                slab = Slab(region, property=property_key)
                self.assertEqual(slab.property, property)


# %%
if __name__ == "__main__":
    unittest.main()
