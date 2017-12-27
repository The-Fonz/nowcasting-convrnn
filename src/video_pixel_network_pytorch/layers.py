import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMultiplicativeBlock(nn.Module):
    def __init__(self, input_channels, output_channels, internal_channels, n_mu=2, kernel_size=3,
                 dilation=1, additive_skip=True, integrate_frame_channels=0, mask=False):
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
            integrate_frame_channels (int): Number of channels that we need to add after first convolution
            mask (bool): Use masked convolutions
        """
        super(ResidualMultiplicativeBlock, self).__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.integrate_frame_channels = integrate_frame_channels
        
        self.additive_skip = additive_skip
        # Use a ModuleList to make sure modules are discoverable by e.g. .parameters()
        self.mus = nn.ModuleList([
                        MultiplicativeUnit(internal_channels, kernel_size=kernel_size, dilation=dilation, mask=mask)
                        for i in range(n_mu)])

        # These are used to convert to a lower number of channels internally
        # Kernel size and dilation should always be 1, and padding 0
        # Note that we subtract number of channels that we need to add after first convolution
        self.conv_input  = nn.Conv2d(input_channels, internal_channels - self.integrate_frame_channels, 1)
        self.conv_output = nn.Conv2d(internal_channels, output_channels, 1)

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
                frame = frame[:, :, h_start:h_end, w_start:w_end]
                # Concatenate in channel dimension
                x = torch.cat((x, frame), dim=1)

            for mu in self.mus:
                # nopad helps in calculating only the center pixel
                p = mu(x, nopad=True)
                x[:,:,half_width, half_width] = p

            x = self.conv_output(x)

            if self.additive_skip:
                x = h + x
            # Slice to retain dims
            return x[:,:,half_width:half_width+1, half_width:half_width+1]

        # Masked convolution
        else:
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


class MultiplicativeUnit(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, mask=False):
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
        # TODO: Implement masking for multichannel image
        self.mask = mask
        # Use SAME padding to retain input resolution, taking dilation into account
        pad = dilation * (kernel_size - 1) // 2
        self.default_padding = (pad, pad)
        # They do use bias in these convolution operators in the paper, but omitted them from the formulas for clarity
        # Use one convolution for all 4 gates, split into 4 after calculation
        self.conv = nn.Conv2d(channels, channels*4, kernel_size, dilation=dilation, padding=self.default_padding)

    def forward(self, h, nopad=False):
        """
        Forward step. Call class instead of this method directly. Will apply mask if
        self.training=True and self.mask=True. self.training can be set/unset by calling .train() or
        its inverse .eval() on this module or any parent.

        Args:
            h: Variable with shape (b,c,h,w)

        Returns:
            output: Variable with shape (b,c,h,w)
        """
        # TODO: We're now always doing this, as even after training the last step might be
        #       a gradient update. So we need to mask again. We might be able to avoid it during inference though.
        if self.mask:
            # When training masked decoder we need to mask the convolutions, they were
            # probably changed by the last optimizer update step
            # ones_like should cast to input tensor type so type_as is just to make sure
            mask = torch.ones_like(self.conv.weight.data).type_as(self.conv.weight.data)
            # Assume square kernel. Dilations don't make a difference
            idx_middle = self.conv.weight.data.size()[-1] // 2
            # Zero itself and any other value in same row
            mask[:, :,idx_middle, idx_middle:] = 0
            # Zero all values below
            mask[:, :, idx_middle+1:, :] = 0
            # Apply mask. Uses broadcasting as shape of mask is (h,w) and shape of
            # weights is (out_channels, in_channels, h, w)
            self.conv.weight.data *= mask

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
