from itertools import cycle

import torch
import torch.nn as nn

from .layers import ResidualMultiplicativeBlock
from .convlstm import ConvLSTM


class Encoder(nn.Module):
    def __init__(self, input_channels, output_channels, internal_channels, hidden_dim,
                 n_rmb, dilation=None, kernel_size=3, use_lstm_peepholes=False):
        """
        Video Pixel Network encoder

        Args:
            input_channels (int): Number of channels of input (e.g. 1 for grayscale or 3 for RGB)
            internal_channels (int): Number of channels used internally by RMB (*c* in paper)
            output_channels (int): Number of channels used between RMBs, in LSTM, and as output (*2c* in paper)
            n_rmb (int): Number of residual multiplicative blocks (*k* in paper)
            dilation (list, optional): List of dilations per RMB (for dilated convolutions). Default 1. dilation repeats so
                       `len(dilation)` can be an even fraction of `n_rmb`. e.g. [1,2,3,4] with 8 RMBs.
            kernel_size (int): Kernel size
            use_lstm_peepholes (bool): Use peepholes in LSTM so that gates see state C
        """
        super(Encoder, self).__init__()
        
        # Repeat if n_rmb is integer multiple of list length
        if dilation and len(dilation) != n_rmb:
            if len(dilation) % n_rmb:
                raise Warning("Length {} of dilation list is not even fraction of n_rmb {}".format(len(dilation), n_rmb))
            dilation = cycle(dilation)

        # Don't use mask as this is encoder
        # First RMB block needs to (up)convert the number of input channels to internal_channels
        # TODO: For input_channels=1 the residual skip connection works because of broadcasting.
        #       But for multiple channels broadcasting doesn't work any more, so we need to either
        #       upconvert to internal_channels before the RMB or set residual_skip=False. Not sure
        #       which way is better.
        self.rmbs = nn.ModuleList([ResidualMultiplicativeBlock(
            input_channels = input_channels,
            output_channels = output_channels,
            internal_channels = internal_channels,
            n_mu=2,
            kernel_size=kernel_size,
            dilation=dilation[0] if dilation else 1,
            mask=False)])

        for i in range(1, n_rmb):
            # Last RMB block outputs output_channels to feed into LSTM
            self.rmbs.append(ResidualMultiplicativeBlock(
                input_channels = output_channels,
                output_channels = output_channels,
                internal_channels = internal_channels,
                n_mu=2,
                kernel_size=kernel_size,
                dilation=dilation[i] if dilation else 1,
                mask=False))

        # Input dim and hidden dim are the same
        # Use kernel size of 3x3 by default. In the paper it's not clear what they use or
        # what sane defaults are. Just like for number of layers, it seems they use only 1 layer.
        self.lstm = ConvLSTM(output_channels, output_channels, kernel_size=[3,3], num_layers=1,
                             bias=True, peepholes=use_lstm_peepholes)

    def forward(self, input, lstm_state=None):
        """
        Input must have shape (b,t,c,h,w). Don't call this method but __call__ the class, it takes
        care of hooks.
        """
        outputs = []
        # We can theoretically compute the timesteps in parallel by treating
        # every timestep as separate batch, not sure if that'd be useful as we
        # need to compute LSTM serially anyway
        for timestep in input.size(1):
            # Has shape (b,c,h,w) because we index the timestep dimension
            x = input[:,timestep]
            for rmb in self.rmbs:
                x = rmb(x)
                # x has shape (b,c,h,w)
            layer_output_list, lstm_state = self.lstm(x, hidden_state=lstm_state)
            # All outputs are returned so take last one
            x = layer_output_list[-1]
            # Is list of tensor per timestep [(b,c,h,w), (b,c,h,w)]
            outputs.append(x)
        # Stack to (b,t,c,h,w) by adding t dim
        outputs = torch.stack(outputs, dim=1)

        return outputs, lstm_state


class Decoder(nn.Module):
    def __init__(self, n_rmb, input_channels, image_channels, output_channels, internal_channels,
                 kernel_size=3):
        """
        Video Pixel Network decoder

        Args:
            n_rmb (int): Number of residual multiplicative blocks (*l* in paper)
            input_channels (int): Number of channels of input and between RMBs
            image_channels (int): Number of channels of image, e.g. 1 for grayscale and 3 for RGB
            output_channels (int): Number of channels of logit outputs
            internal_channels (int): Number of channels inside RMBs
            kernel_size (int): Size of square kernel. Currently only one kernel size for all RMBs is supported.
        """
        super(Decoder, self).__init__()

        if image_channels != 1:
            raise NotImplementedError("Any other number of channels than 1 is not implemented yet.")

        # Use masking
        # First RMB needs to integrate target frame
        self.rmbs = nn.ModuleList([ResidualMultiplicativeBlock(
            input_channels=input_channels,
            output_channels=input_channels,
            internal_channels=internal_channels,
            n_mu=2,
            kernel_size=kernel_size,
            dilation=1,
            integrate_frame_channels=image_channels,
            mask=True)])

        # Make all RMBs except first and last
        self.rmbs.extend([ResidualMultiplicativeBlock(
            input_channels=input_channels,
            output_channels=input_channels,
            internal_channels=internal_channels,
            n_mu=2,
            kernel_size=kernel_size,
            dilation=1,
            mask=True) for i in range(1, n_rmb-1)])

        # Last RMB outputs logits
        self.rmbs.append(ResidualMultiplicativeBlock(
            input_channels=input_channels,
            output_channels=output_channels,
            internal_channels=internal_channels,
            n_mu=2,
            kernel_size=kernel_size,
            dilation=1,
            mask=True))

    def forward(self, inputs, targets=None):
        """
        Don't call this method but __call__ the class.

        Args:
            inputs: Encoder inputs shaped (b,t,c,h,w)
            targets: Targets used as context for masked convolutions
        """
        if self.training:
            logits = []
            for timestep in inputs.size(1):
                x = inputs[:,timestep]
                # Calc all rmbs
                for rmb in self.rmbs:
                    x = rmb(x, frame=targets[:,timestep])
                # Don't use softmax as we'll use it with loss func for numerical stability
                logits.append(x)

            return logits

        else:
            # Mark input variable as volatile to avoid excessive memory use
            # TODO: Figure out if this is a good thing or if it should be left to caller
            inputs.volatile = True

            # TODO: Implement method to only process center pixel in RMB class. Should
            #       not matter which pixel it is
            #       Then cycle over all pixels here
            raise NotImplementedError()
