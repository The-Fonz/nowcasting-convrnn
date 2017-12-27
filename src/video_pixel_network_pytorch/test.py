import unittest

import numpy as np
import torch
from torch.autograd import Variable

from .layers import *
from .convlstm import *
from .vpn import *


class MUtest(unittest.TestCase):

    def test_simple_nomask(self):
        mu = MultiplicativeUnit(5, kernel_size=3, dilation=1, mask=False)
        h = Variable(torch.zeros(1,5,1,1))
        x = mu(h)
        self.assertEqual(x.size(), h.size())

    def test_mask(self):
        mu = MultiplicativeUnit(1, kernel_size=3, dilation=1, mask=True)
        h = Variable(torch.eye(3)[np.newaxis, np.newaxis, :, :], volatile=True)
        x = mu(h)
        self.assertEqual(x.size(), h.size())


# noinspection PyCallingNonCallable
class RMBtest(unittest.TestCase):

    def test_simple_nomask(self):
        rmb = ResidualMultiplicativeBlock(
            input_channels = 4,
            output_channels = 4,
            internal_channels = 2, n_mu=2, kernel_size=3,
            dilation=1, additive_skip=True,
            integrate_frame_channels=0, mask=False)
        h = Variable(torch.zeros(1,4,3,3), volatile=True)
        x = rmb(h)
        self.assertEqual(x.size(), h.size())

    def test_simple_mask(self):
        rmb = ResidualMultiplicativeBlock(
            input_channels=4,
            output_channels=4,
            internal_channels=2, n_mu=2, kernel_size=3,
            dilation=1, additive_skip=True,
            integrate_frame_channels=0, mask=True)
        h = Variable(torch.zeros(1, 4, 3, 3), volatile=True)
        x = rmb(h)
        self.assertEqual(x.size(), h.size())

    def test_integrate_frame(self):
        rmb = ResidualMultiplicativeBlock(
            input_channels=4,
            output_channels=7,
            internal_channels=4, n_mu=2, kernel_size=3,
            dilation=1, additive_skip=False,
            integrate_frame_channels=3, mask=True)
        h = Variable(torch.zeros(1, 4, 3, 3), volatile=True)
        frame = Variable(torch.zeros(1, 3, 3, 3), volatile=True)
        x = rmb(h, frame=frame)
        self.assertEqual(list(x.size()), [1,7,3,3])

    def test_dilation(self):
        rmb = ResidualMultiplicativeBlock(
            input_channels=4,
            output_channels=7,
            internal_channels=4, n_mu=2, kernel_size=3,
            dilation=2, additive_skip=False,
            integrate_frame_channels=3, mask=True)
        h = Variable(torch.zeros(1, 4, 3, 3), volatile=True)
        frame = Variable(torch.zeros(1, 3, 3, 3), volatile=True)
        x = rmb(h, frame=frame)
        self.assertEqual(list(x.size()), [1, 7, 3, 3])

    def test_pixel(self):
        rmb = ResidualMultiplicativeBlock(
            input_channels=4,
            output_channels=4,
            internal_channels=4, n_mu=2, kernel_size=3,
            dilation=1, additive_skip=True,
            integrate_frame_channels=1, mask=True)
        h = Variable(torch.zeros(2, 4, 3, 3), volatile=True)
        frame = Variable(torch.zeros(2, 1, 12, 12), volatile=True)
        x = rmb(h, frame=frame, pixel=(1,2))
        self.assertEqual(list(x.size()), [2, 4, 1, 1])
