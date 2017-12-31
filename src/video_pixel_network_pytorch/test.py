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
        h = Variable(torch.zeros(1,5,1,1), volatile=True)
        x = mu(h)
        self.assertEqual(x.size(), h.size())

    def test_mask(self):
        mu = MultiplicativeUnit(1, kernel_size=3, dilation=1, mask=True)
        h = Variable(torch.eye(3)[np.newaxis, np.newaxis, :, :], volatile=True)
        x = mu(h)
        self.assertEqual(x.size(), h.size())


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


class ConvLSTMtest(unittest.TestCase):

    def test_cell(self):
        cell = ConvLSTMCell(input_dim=3, hidden_dim=32, kernel_size=[3,3],
                            bias=True, peepholes=False)
        h = cell.init_hidden(batch_size=4, height=16, width=9)
        x = Variable(torch.randn(4, 3, 16, 9), volatile=True)
        h_next, c_next = cell(x, h)
        self.assertEqual(list(h_next.size()), [4, 32, 16, 9])

    def test_cell_peepholes(self):
        cell = ConvLSTMCell(input_dim=3, hidden_dim=32, kernel_size=[5,5],
                            bias=True, peepholes=True)
        h = cell.init_hidden(batch_size=4, height=16, width=9)
        x = Variable(torch.randn(4, 3, 16, 9), volatile=True)
        h_next, c_next = cell(x, h)
        self.assertEqual(list(h_next.size()), [4, 32, 16, 9])

    def test_lstm(self):
        lstm = ConvLSTM(input_dim=2, hidden_dim=16, kernel_size=[3,3],
                        num_layers=2, bias=True, peepholes=False)
        # (b,t,c,h,w)
        x = Variable(torch.randn(4, 5, 2, 16, 9), volatile=True)
        layer_output_list, last_state_list = lstm(x)
        # Two layers
        self.assertEqual(len(layer_output_list), 2)
        # Output has shape (b,t,hidden_dim,h,w)
        self.assertEqual(list(layer_output_list[0].size()), [4, 5, 16, 16, 9])
        self.assertEqual(len(last_state_list), 2)
        self.assertEqual(list(last_state_list[0][0].size()), [4, 16, 16, 9])

    def test_lstm_peepholes(self):
        lstm = ConvLSTM(input_dim=2, hidden_dim=16, kernel_size=[3,3],
                        num_layers=2, bias=True, peepholes=True)
        x = Variable(torch.randn(4, 5, 2, 16, 9), volatile=True)
        layer_output_list, last_state_list = lstm(x)
        self.assertEqual(len(layer_output_list), 2)
        self.assertEqual(list(layer_output_list[0].size()), [4, 5, 16, 16, 9])
        self.assertEqual(len(last_state_list), 2)
        self.assertEqual(list(last_state_list[0][0].size()), [4, 16, 16, 9])


class EncoderTest(unittest.TestCase):

    def test_simple(self):
        enc = Encoder(input_channels=1, output_channels=16, internal_channels=8,
                      n_rmb=3, dilation=None, kernel_size=3,
                      lstm_layers=1, use_lstm_peepholes=True)
        x = Variable(torch.randn(4, 5, 1, 8, 10), volatile=True)
        outputs, lstm_state = enc(x)
        self.assertEqual(list(outputs.size()), [4, 5, 16, 8, 10])
        # h
        self.assertEqual(list(lstm_state[0][0].size()), [4, 16, 8, 10])
        # c
        self.assertEqual(list(lstm_state[0][1].size()), [4, 16, 8, 10])

    def test_dilation(self):
        enc = Encoder(input_channels=1, output_channels=16, internal_channels=8,
                      n_rmb=4, dilation=[1,2], kernel_size=3,
                      lstm_layers=1, use_lstm_peepholes=True)
        x = Variable(torch.randn(4, 5, 1, 8, 10), volatile=True)
        outputs, lstm_state = enc(x)
        self.assertEqual(list(outputs.size()), [4, 5, 16, 8, 10])
        # h
        self.assertEqual(list(lstm_state[0][0].size()), [4, 16, 8, 10])
        # c
        self.assertEqual(list(lstm_state[0][1].size()), [4, 16, 8, 10])


class DecoderTest(unittest.TestCase):

    def test_train(self):
        dec = Decoder(n_rmb=4, input_channels=16, image_channels=1,
                      output_channels=128,
                      internal_channels=8, kernel_size=3)
        dec.train()
        h = Variable(torch.randn(4, 5, 16, 8, 10), volatile=True)
        targets = Variable(torch.randn(4, 5, 1, 8, 10))
        logits = dec(h, targets=targets)
        self.assertEqual(list(logits.size()), [4, 5, 128, 8, 10])

    def test_inference(self):
        dec = Decoder(n_rmb=4, input_channels=16, image_channels=1,
                      output_channels=1,
                      internal_channels=8, kernel_size=3)
        # Set to inference mode
        dec.eval()
        h = Variable(torch.randn(4, 5, 16, 8, 10), volatile=True)
        img = dec(h)
        self.assertEqual(list(img.size()), [4, 5, 1, 8, 10])
