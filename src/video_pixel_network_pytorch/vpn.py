import torch

from .layers import ResidualMultiplicativeBlock
from .convlstm import ConvLSTM


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_rmb, stride=None, kernel_size=3):
        """
        Video Pixel Network encoder

        :param n_rmb: Number of residual multiplicative blocks (*k* in paper)
        :kwarg stride: List of strides per RMB (for dilated convolutions). Default 1
        :kwarg kernel_size: Kernel size
        :kwarg use_cuda: Whether or not to use CUDA
        """
        # Don't use mask as this is encoder
        # TODO: Find out how to convert img to RMB channel size
        self.rmbs = torch.nn.ModuleList([ResidualMultiplicativeBlock(
            channels=n_channels,
            mu_channels_divideby=2, n_mu=2, kernel_size=kernel_size,
            stride=stride[i] if stride else 1,
            mask=False) for i in range(n_rmb)])

        # TODO: what is rmb output dim
        self.lstm = ConvLSTM(rmb_output_dim, hidden_dim, kernel_size, num_layers,
                             batch_first=True, bias=True, return_all_layers=False)

    def forward(self, x):
        "x must have shape (b,t,c,h,w)"
        if self.training:
            # We can theoretically compute the timesteps in parallel by treating
            # every timestep as separate batch
            for timestep in x.size(1):
                for rmb in self.rmbs:
                    x = rmb(x[:,timestep])
                # TODO: Properly implement
                self.lstm(x)


class Decoder(torch.nn.Module):
    def __init__(self, n_rmb):
        """
        Video Pixel Network decoder

        :param n_rmb: Number of residual multiplicative blocks (*l* in paper)
        """
        # Use masking
        self.rmbs = torch.nn.ModuleList([ResidualMultiplicativeBlock(
                mu_channels_divideby=2, n_mu=2, kernel_size=3,
                stride=1, mask=True) for i in range(n_rmb)])

    def forward(self, input):
        if self.training:
            for timestep in input.size(1):
                pass

        else:
            raise NotImplementedError()
