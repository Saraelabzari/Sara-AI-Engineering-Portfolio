import unittest

import numpy as np
from PIL import Image

from app import preprocess_image


class PreprocessingTests(unittest.TestCase):
    def test_preprocess_returns_grayscale_array(self):
        image = Image.new("RGB", (100, 60), color="white")
        result = preprocess_image(image)
        self.assertEqual(result.shape, (60, 100))
        self.assertEqual(result.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
