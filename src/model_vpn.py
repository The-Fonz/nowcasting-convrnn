import torch
import torch.nn as nn

from video_pixel_network_pytorch.vpn import Encoder, Decoder
from torch.autograd.variable import Variable


class VPN(nn.Module):
    def __init__(self, img_channels, c, n_rmb_encoder, n_rmb_decoder,
                 n_pixvals = 255,
                 enc_dilation = None,
                 enc_kernel_size = 3,
                 lstm_layers = 1,
                 use_lstm_peepholes=True, use_cuda=True):
        """
        Video Pixel Network model
        """
        super(VPN, self).__init__()

        self.use_cuda = use_cuda

        self.encoder = Encoder(input_channels=img_channels, output_channels=c, internal_channels=c//2,
                      n_rmb=n_rmb_encoder, dilation=enc_dilation, kernel_size=enc_kernel_size,
                      lstm_layers=lstm_layers, use_lstm_peepholes=use_lstm_peepholes)

        self.decoder = Decoder(n_rmb=n_rmb_decoder, input_channels=c, image_channels=img_channels,
                      output_channels=n_pixvals,
                      internal_channels=c//2, kernel_size=3)

    def forward(self, inputs_var, targets=None):

        if self.training:

            # Forward pass of encoder for all timesteps
            context, lstm_state = self.encoder(inputs_var)

            # Forward pass of decoder for all timesteps
            output = self.decoder(context, targets=targets)

            # Return decoder outputs
            return output

        else:
            # Forward pass of encoder for all timesteps in input_var

            # Forward pass of decoder with last context (encoder output) and
            # no target img to condition on (all is generated)

            # Feed output of decoder to encoder. Do one forward pass for encoder.
            # Pass context to decoder. Do one forward pass for decoder. Repeat this for `n_predict` times.

            # Output outputs of decoder layer

            raise NotImplementedError("Evaluation mode not implemented yet.")