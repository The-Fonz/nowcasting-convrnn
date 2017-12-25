import torch


class ResidualMultiplicativeBlock(torch.nn.Module):
    def __init__(self, channels, mu_channels_divideby=2, n_mu=2, kernel_size=3,
                 stride=1, mask=False):
        """
        Residual Multiplicative Block consisting of stacked Multiplicative Units

        :param channels: Number of input/output channels
                         (must be divisible by *mu_channels_divideby*)
        :kwarg mu_channels_divideby: Divide input channels by this number to get number
                                     of channels to work with inside of block (achieved
                                     with 1x1 convolutions). We do this for more
                                     computational efficiency.
        :kwarg n_mu: Number of Multiplicative Units
        :kwarg kernel_size: Square kernel size inside MU
        :kwarg stride: Stride inside MU
        """
        if channels % mu_channels_divideby:
            raise Warning("Number of channels {} is not divisible by {}".format(
                channels, mu_channels_divideby))

        mu_channels = channels//mu_channels_divideby
        # Use a ModuleList to make sure modules are discoverable by e.g. .parameters()
        self.mus = torch.nn.ModuleList([
                        MultiplicativeUnit(mu_channels, kernel_size=kernel_size, stride=stride, mask=mask)
                        for i in range(n_mu)])
        # Use SAME padding to retain input resolution
        padding = kernel_size // 2
        # These are used to convert to a lower number of channels internally
        # Kernel size and stride should always be 1
        self.conv_input  = torch.nn.Conv2d(channels, mu_channels, 1, padding=padding)
        self.conv_output = torch.nn.Conv2d(mu_channels, channels, 1, padding=padding)

    def forward(self, h):
        """
        Make forward step through network. Call class instead of this method directly,
        as nn.Module.__call__ will take care of hooks.
        """
        x = self.conv_input(h)
        for mu in self.mus:
            x = mu(x)
        x = self.conv_output(x)
        # Residual (additive skip) connection from input to output
        return h + x


class MultiplicativeUnit(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, mask=False):
        """
        Multiplicative Unit

        :param channels: Number of channels, input and output are equal size
        :kwarg kernel_size: Size of square kernel
        :kwarg stride: Stride of convolutions
        :kwarg mask: Use mask that masks the pixel to predict,
                     any value in the same row,
                     and any value in any row below it.
                     TODO: Implement mask for multichannel image
        """
        self.mask = mask
        # Use SAME padding to retain input resolution
        padding = kernel_size // 2
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size, stride=stride, padding=padding)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size, stride=stride, padding=padding)
        self.conv3 = torch.nn.Conv2d(channels, channels, kernel_size, stride=stride, padding=padding)
        self.conv4 = torch.nn.Conv2d(channels, channels, kernel_size, stride=stride, padding=padding)

    def forward(self, h):
        # self.training can be set/unset by self.train() and self.eval()
        if self.training:
            # When training decoder we need to mask the convolutions, they were
            # probably changed by the last optimizer update step
            if self.mask:
                for conv in (self.conv1, self.conv2, self.conv3, self.conv4):
                    # TODO: Check if ones_like already casts to tensor type
                    mask = torch.ones_like(conv.weight.data).type_as(conv.weight.data)
                    # Assume square kernel
                    idx_middle = conv.weight.data.size()[0] // 2
                    # Zero itself and any other value in same row
                    mask[idx_middle, idx_middle:] = 0
                    # Zero all values below
                    mask[idx_middle+1:, :] = 0
                    # Apply mask
                    conv.weight.data *= mask
            g1 = torch.sigmoid(self.conv1(h))
            g2 = torch.sigmoid(self.conv2(h))
            g3 = torch.sigmoid(self.conv3(h))
            u  = torch.tanh(self.conv4(h))
            return g1 * torch.tanh(g2 * h + g3 * u)
        # When not training decoder we need to generate samples sequentially
        else:
            raise NotImplementedError()

