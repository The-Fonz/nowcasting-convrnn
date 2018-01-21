import torch
import torch.nn as nn
import torch.nn.functional as F

from .masking import mask


class MultiplicativeUnit(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        """
        Multiplicative Unit

        Args:
            channels (int): Number of channels, input and output are equal size
            kernel_size (int): Size of square kernel
            dilation (int): dilation of convolutions
            mask (bool): Use mask that masks the pixel to predict,
                     any value in the same row,
                     and any value in any row below it.
        """
        super(MultiplicativeUnit, self).__init__()
        # Use SAME padding to retain input resolution, taking dilation into account
        pad = dilation * (kernel_size - 1) // 2
        self.default_padding = (pad, pad)
        # They do use bias in these convolution operators in the paper, but omitted them from the formulas for clarity
        # Use one convolution for all 4 gates, split into 4 after calculation
        self.conv = nn.Conv2d(channels, channels*4, kernel_size, dilation=dilation, padding=self.default_padding)

    def get_weight(self):
        """
        Get number of chunks and convolution weight.

        Returns:
            n_chunks, weight (`torch.autograd.Variable`)
        """
        return 4, self.conv.weight

    def mask(self, maskfunc, *args, **kwargs):
        """
        Mask convolution with `maskfunc(weights: )`
        """
        maskfunc(self.conv.weight)

    def forward(self, h, nopad=False):
        """
        Forward step. Call class instead of this method directly.

        Args:
            h: Variable with shape (b,c,h,w)

        Returns:
            output: Variable with shape (b,c,h,w)
        """

        if nopad:
            self.conv.padding = (0,0)

        # Split along channel axis (b,c,h,w)
        c1, c2, c3, c4 = torch.chunk(self.conv(h), chunks=4, dim=1)

        g1 = torch.sigmoid(c1)
        g2 = torch.sigmoid(c2)
        g3 = torch.sigmoid(c3)
        u  = torch.tanh(c4)

        if nopad:
            p = self.default_padding[0]
            # h was chopped down in the conv
            h = h[:, :, p:-p, p:-p]

        output = g1 * torch.tanh(g2 * h + g3 * u)

        self.conv.padding = self.default_padding

        return output


class ResidualMultiplicativeBlock(nn.Module):
    def __init__(self, input_channels, output_channels, internal_channels, image_channels=None, n_context=None,
                 n_mu=2, kernel_size=3, dilation=1, additive_skip=True, integrate_frame_channels=0,
                 mu_class=MultiplicativeUnit):
        """
        Residual Multiplicative Block consisting of stacked Multiplicative Units

        Args:
            input_channels (int): Number of input channels
            output_channels (int): Number of output channels
            internal_channels (int): Use 1x1 convolutions to have this number of channels internally for the MUs
            n_mu (int): Number of Multiplicative Units
            kernel_size (int): Square kernel size inside MU
            dilation (int): dilation inside MU (for dilations)
            additive_skip (bool): Use additive skip connection from input to output (they
                              need to have the same number of channels)
            integrate_frame_channels (int): Number of channels that we need to put on top of layer stack after input
                                            convolution
            mask (bool): Use masked convolutions
            mu_class: MultiplicativeUnit class. Used as kwa        self.n_rg so we can use other classes if we want.
        """
        super(ResidualMultiplicativeBlock, self).__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.integrate_frame_channels = integrate_frame_channels

        self.image_channels = image_channels
        self.n_context = n_context
        self.output_channels = output_channels

        self.additive_skip = additive_skip
        # Use a ModuleList to make sure modules are discoverable by e.g. .parameters()
        self.mus = nn.ModuleList([
                        mu_class(internal_channels, kernel_size=kernel_size, dilation=dilation)
                        for i in range(n_mu)])

        # These are used to convert to a lower number of channels internally
        # Kernel size and dilation should always be 1, and padding 0
        # Note that we subtract number of channels that we need to add after first convolution
        self.conv_input  = nn.Conv2d(input_channels, internal_channels - self.integrate_frame_channels, 1)
        self.conv_output = nn.Conv2d(internal_channels, output_channels, 1)

    def mask(self, last=False):
        """
        Mask MU's. We have all information needed from the arguments in the constructor
        """
        # First mask input convolution, if needed
        if not self.integrate_frame_channels:
            mask(self.conv_input.weight, self.image_channels, self.n_context, mask_center_pixel_current=False,
                 input_repeats=True)

        for i in range(len(self.mus)):
            n_chunks, weight = self.mus[i].get_weight()
            # We need to stop information flow from current pixel
            # Also, first conv needs to mix in frame channels so input_repeats=False
            first = self.integrate_frame_channels and i==0
            mask(weight, self.image_channels, self.n_context, mask_center_pixel_current=first,
                 input_repeats=not first, n_chunks=n_chunks)

        mask(self.conv_output.weight, self.image_channels, self.n_context, mask_center_pixel_current=False,
             input_repeats=True, n_logits_per_channel=[self.output_channels])

    def forward(self, h, frame=None, pixel=None):
        """
        Make forward step through network. Call class instead of this method directly,
        as nn.Module.__call__ will take care of hooks.

        Args:
            h (Variable): Tensor with shape (b,c,h,w). Full (h,w) required even if kwarg pixel is given
            frame (Variable, optional): Current frame to be conditioned on with shape (b,c,h,w)
            pixel (tuple of (h,w), optional):
                If given, will only process the pixel at this (h,w) location (for inference)

        Returns:
             output: Tensor with shape (b,c,h,w) or single pixel value if kwarg pixel is given
        """

        # Pixel coordinate is given and we need to calculate just that one pixel
        if pixel:
            return self._forward_pixel(h, frame, pixel)

        # Masked convolution
        else:
            return self._forward_masked(h, frame)

    def _forward_pixel(self, h, frame, pixel):
        padding = self.dilation * (self.kernel_size - 1) // 2
        # TODO: Move this repetitive padding, Decoder.forward() maybe
        # Pad with 0's, padding tuple ordered (left, right, top, bottom)
        x = F.pad(h, [padding] * 4)
        # Take just the patch size necessary for our kernel size and dilation
        half_width = ((self.kernel_size - 1) * self.dilation + 1) // 2
        # Add 1 to h_end as end is non-inclusive
        h_start, h_end = pixel[0] + padding - half_width, pixel[0] + padding + half_width + 1
        w_start, w_end = pixel[1] + padding - half_width, pixel[1] + padding + half_width + 1
        # Slice out square with the right size
        x = x[:, :, h_start:h_end, w_start:w_end]
        # Run through 1x1 conv
        x = self.conv_input(x)

        if self.integrate_frame_channels:
            # Need to first pad of course
            frame = F.pad(frame, [padding] * 4)
            frame = frame[:, :, h_start:h_end, w_start:w_end]
            # Concatenate in channel dimension
            x = torch.cat((x, frame), dim=1)

        for mu in self.mus:
            # nopad helps in calculating only the center pixel
            p = mu(x, nopad=True)
            x[:, :, half_width, half_width] = p

        x = self.conv_output(x)

        if self.additive_skip:
            h_pad = F.pad(h, [padding] * 4)
            x = h_pad[:, :, h_start:h_end, w_start:w_end] + x
        # Slice to retain dims
        return x[:, :, half_width:half_width + 1, half_width:half_width + 1]

    def _forward_masked(self, h, frame):
        x = self.conv_input(h)

        if self.integrate_frame_channels:
            x = torch.cat((x, frame), dim=1)

        for mu in self.mus:
            x = mu(x)
        x = self.conv_output(x)

        if self.additive_skip:
            # Residual (additive skip) connection from input to output
            x = h + x
        return x
