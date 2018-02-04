from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .layers import ResidualMultiplicativeBlock
from .convlstm import ConvLSTM


class Encoder(nn.Module):
    def __init__(self, input_channels, output_channels, internal_channels,
                 n_rmb, n_mu_per_rmb=2, dilation=None, kernel_size=3,
                 lstm_layers=1, use_lstm_peepholes=False, lstm_kernel_size=[3,3]):
        """
        Video Pixel Network encoder

        Args:
            input_channels (int): Number of channels of input (e.g. 1 for grayscale or 3 for RGB)
            internal_channels (int): Number of channels used internally by RMB (*c* in paper)
            output_channels (int): Number of channels used between RMBs, in LSTM, and as output (*2c* in paper)
            n_rmb (int): Number of residual multiplicative blocks (*k* in paper)
            n_mu_per_rmb (int): Number of MUs per RMB (2 in paper)
            dilation (list, optional): List of dilations per RMB (for dilated convolutions). Default 1. dilation repeats so
                       `len(dilation)` can be an even fraction of `n_rmb`. e.g. [1,2,3,4] with 8 RMBs.
            kernel_size (int): Kernel size
            use_lstm_peepholes (bool): Use peepholes in LSTM so that gates see state C
            lstm_kernel_size ([int,int]): Kernel size used by ConvLSTM ([3,3] in paper)
        """
        super(Encoder, self).__init__()
        
        # Repeat if n_rmb is integer multiple of list length
        if dilation and len(dilation) != n_rmb:
            if n_rmb % len(dilation):
                raise Warning("Length {} of dilation list is not even fraction of n_rmb {}".format(len(dilation), n_rmb))
            dilation = cycle(dilation)
            # Put into list to make subscriptable
            dilation = [next(dilation) for i in range(n_rmb)]

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
            n_mu=n_mu_per_rmb,
            kernel_size=kernel_size,
            dilation=dilation[0] if dilation else 1)])

        for i in range(1, n_rmb):
            # Last RMB block outputs output_channels to feed into LSTM
            self.rmbs.append(ResidualMultiplicativeBlock(
                input_channels = output_channels,
                output_channels = output_channels,
                internal_channels = internal_channels,
                n_mu=n_mu_per_rmb,
                kernel_size=kernel_size,
                dilation=dilation[i] if dilation else 1))

        # Input dim and hidden dim are the same
        # Use kernel size of 3x3 by default. In the paper it's not clear what they use or
        # what sane defaults are. Just like for number of layers, it seems they use only 1 layer.
        self.lstm = ConvLSTM(output_channels, output_channels, kernel_size=lstm_kernel_size, num_layers=lstm_layers,
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
        for i_timestep in range(input.size(1)):
            # Has shape (b,c,h,w) because we index the timestep dimension
            # Slice to retain timestep dim
            x = input[:,i_timestep]
            for rmb in self.rmbs:
                x = rmb(x)
                # x has shape (b,c,h,w)
            # Add time axis as ConvLSTM expects (b,t,c,h,w)
            layer_output_list, lstm_state = self.lstm(x[:,np.newaxis], hidden_state=lstm_state)
            # All outputs are returned so take last one (highest layer)
            x = layer_output_list[-1]
            # Is list of tensor per timestep [(b,t,c,h,w), (b,t,c,h,w)]
            outputs.append(x)

        # Concatenate in t dim
        outputs = torch.cat(outputs, dim=1)

        return outputs, lstm_state


class Decoder(nn.Module):
    def __init__(self, n_rmb, input_channels, output_channels, internal_channels, image_channels=None, n_context=None,
                 kernel_size=3, mask=False):
        """
        Video Pixel Network decoder.

        NOTE: Need to call .mask() for masking!

        Args:
            n_rmb (int): Number of residual multiplicative blocks (*l* in paper)
            input_channels (int): Number of channels of input and between RMBs
            image_channels (int): Number of channels of image, e.g. 1 for grayscale and 3 for RGB
            output_channels (int): Number of channels of logit outputs
            internal_channels (int): Number of channels inside RMBs
            kernel_size (int): Size of square kernel. Currently only one kernel size for all RMBs is supported.
        """
        super(Decoder, self).__init__()

        self.masking = mask
        self.image_channels = image_channels

        # With current implementation we're creating first and last blocks at least.
        # Supporting n_rmb < 2 is trivial but would need implementation and is probably not effective anyway.
        if n_rmb < 2:
            raise NotImplementedError("Fewer RMBs than 2 are not supported.")
        if image_channels > 1:
            raise NotImplementedError("More than 1 image channel is not implemented yet.")

        # Use masking
        # First RMB needs to integrate target frame
        self.rmbs = nn.ModuleList([ResidualMultiplicativeBlock(
            input_channels=input_channels,
            output_channels=input_channels,
            internal_channels=internal_channels,
            image_channels=image_channels,
            n_context=n_context,
            n_mu=2,
            kernel_size=kernel_size,
            dilation=1,
            integrate_frame_channels=image_channels if mask else 0,
            additive_skip=True)])

        # Make all RMBs except first and last
        self.rmbs.extend([ResidualMultiplicativeBlock(
            input_channels=input_channels,
            output_channels=input_channels,
            internal_channels=internal_channels,
            image_channels=image_channels,
            n_context=n_context,
            n_mu=2,
            kernel_size=kernel_size,
            dilation=1,
            additive_skip=True) for i in range(1, n_rmb-1)])

        # Last RMB outputs logits
        # Turn off additive skip
        self.rmbs.append(ResidualMultiplicativeBlock(
            input_channels=input_channels,
            output_channels=output_channels,
            internal_channels=internal_channels,
            image_channels=image_channels,
            n_context=n_context,
            n_mu=2,
            kernel_size=kernel_size,
            dilation=1,
            additive_skip=False))

    def mask(self):
        """
        Mask MUs of RMBs.
        """
        for i, rmb in enumerate(self.rmbs):
            rmb.mask(last = i==(len(self.rmbs)-1))

    def forward(self, inputs, targets=None, argmax=False):
        """
        Don't call this method but __call__ the class.

        Uses masked convolutions and targets if self.training=True, and sequential
        prediction of all pixels otherwise.

        self.training can be set/unset by calling .train() or
        its inverse .eval() on this module or any parent.

        Args:
            inputs: Encoder inputs shaped (b,t,c,h,w)
            targets: Targets used as context for masked convolutions
        """
        if self.training:
            if self.masking and targets is None:
                raise AssertionError("In training mode, self.mask=True and targets=None, please supply the targets to train on")
            return self._forward_train(inputs, targets)

        else:
            if self.masking:
                return self._forward_inference_autoregressive(inputs, argmax=argmax)
            else:
                return self._forward_inference_vanilla(inputs, argmax=argmax)

    def _forward_train(self, inputs, targets):
        "Training directly on one-hot targets"
        logits = []
        for i_timestep in range(inputs.size(1)):
            x = inputs[:, i_timestep]
            # First RMB needs current context
            if targets is not None:
                x = self.rmbs[0](x, frame=targets[:, i_timestep])

            for i_rmb in range(1 if targets is not None else 0, len(self.rmbs)):
                x = self.rmbs[i_rmb](x)
            # Don't use softmax as we'll use it with loss func for numerical stability
            logits.append(x)

        # Add timestep dim
        return torch.stack(logits, dim=1)

    def _forward_inference_vanilla(self, inputs, argmax=False):
        "Inference without conditioning on generated output"
        logits = []
        for i_timestep in range(inputs.size(1)):
            x = inputs[:, i_timestep]
            # Calc all rmbs
            for rmb in self.rmbs:
                x = rmb(x)
            # Don't use softmax as we'll use it with loss func for numerical stability
            logits.append(x)

        # Add timestep dim
        logits = torch.stack(logits, dim=1)

        soft = F.softmax(logits, dim=2)

        # TODO: Support multi-channel images, where we have to chunk the logits

        # Put channel axis last (b,t,h,w,c)
        mult = soft.permute(0,1,3,4,2)
        size = list(mult.size())
        # Need contiguous array for .view()
        mult = mult.contiguous()
        # Flatten batch, timestep, h, w dimensions so we have list of discrete prob dists
        mult = mult.view(-1, mult.size(-1))
        # Then take one sample for each
        mult = torch.multinomial(mult, 1)

        size[-1] = 1
        mult = mult.view(*size)
        # Put channel axis back in place (b,t,c,h,w)
        pix_vals = mult.permute(0,1,4,2,3)

        # Make sure to cast to proper type
        return pix_vals.type_as(inputs)

    def _forward_inference_autoregressive(self, inputs, argmax=False):
        "Pixel-by-pixel inference, conditioning on current output"

        # Inputs have same b,t,h,w as predictions
        b, t, c, h, w = inputs.size()

        preds = Variable(torch.zeros(b, t, self.image_channels, h, w))

        if inputs.data.is_cuda:
            preds = preds.cuda()

        for i_timestep in range(inputs.size(1)):
            history = inputs[:, i_timestep]

            for i_h in range(h):
                for i_w in range(w):
                    for i_c in range(self.image_channels):
                        # TODO: Set up patch that we're predicting?

                        # TODO: Run through first RMB. Then all others
                        for rmb in self.rmbs:
                            # TODO: Multichannel
                            pix_logits = rmb(history, frame=preds, pixel=(i_c, i_h, i_w))
                        # (b, c)
                        pix_soft = F.softmax(pix_logits, dim=1)
                        # (b, c)
                        pix_sampled = torch.multinomial(pix_soft, 1)
                        # (b, n_samples=1)

                        preds[:, i_c, i_h, i_w] = pix_sampled[:, 0]

        return preds
