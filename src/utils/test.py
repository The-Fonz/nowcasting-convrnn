import unittest

import torch

from . import *


class OnehotTest(unittest.TestCase):

    def test_simple(self):
        # 1x2x2
        t = torch.FloatTensor([[[1,2],[0,1]]])
        o = onehot.onehot(t, 3)
        self.assertEqual(o.data.tolist(), [[[0.0, 0.0],
                                            [1.0, 0.0]],
                                           [[1.0, 0.0],
                                            [0.0, 1.0]],
                                           [[0.0, 1.0],
                                            [0.0, 0.0]]])
