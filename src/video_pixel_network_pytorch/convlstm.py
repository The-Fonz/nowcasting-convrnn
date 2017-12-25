#
# Rework of ConvLSTM to make it simpler to use
#

import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias, peepholes):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        peepholes: bool
            Use peepholes (state c_cur is dependent on i, f, o (for the latter elementwise))
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.use_bias = bias
        self.use_peepholes = peepholes

        # Don't use built-in bias of conv layers
        # self.conv.weights have shape (hidden_dim*4, input_dim+hidden_dim, kernel_w, kernel_h)
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=False)
        # Set weights to something sane
        self.conv.weight.data.normal_(0, .01)

        if self.use_peepholes:
            # i, f gates are dependent on c_cur
            self.conv_peep_i_f = nn.Conv2d(in_channels=self.hidden_dim,
                                           out_channels=2 * self.hidden_dim,
                                           kernel_size=self.kernel_size,
                                           padding=self.padding,
                                           bias=False)
            # o is dependent on c_cur elementwise
            self.conv_peep_o = nn.Conv2d(in_channels=self.hidden_dim,
                                         out_channels=self.hidden_dim,
                                         kernel_size=[1, 1],
                                         padding=0,
                                         bias=False)
            # Set weights to something sane
            self.conv_peep_i_f.weight.data.normal_(0, .01)
            self.conv_peep_o.weight.data.normal_(0, .01)

        # Use our own separate bias Variables
        # Order is (i, f, o, g)
        # Make sure most is *i*nput to the state, little is *f*orgotten,
        # most is *o*utput, and no bias for candidate C
        # TODO: These are put through a sigmoid so 1 does not have special value. Get more sane defaults?
        self.bias = Variable(torch.FloatTensor([1, 1, 1, 0]), requires_grad=True)

    def forward(self, input_tensor, cur_state):

        h_cur, c_cur = cur_state

        try:
            combined = torch.cat((input_tensor, h_cur), dim=1)  # concatenate along channel axis
        # More useful notification for this common error
        except TypeError as e:
            raise Warning("TypeError when concatenating, you've probably given an incorrect tensor type. "
                          "Tried to concatenate input_tensor {} and h_cur {}"
                          .format(type(input_tensor.data), type(h_cur.data)))

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        if self.use_peepholes:
            # Optional peepholes where i, f are dependent on c_cur...
            cc_i_peep, cc_f_peep = torch.split(self.conv_peep_i_f(c_cur), self.hidden_dim, dim=1)
            # ...and o is dependent on c_cur but elementwise (1x1 convolution)
            cc_o_peep = self.conv_peep_o(c_cur)
            i = torch.sigmoid(cc_i + cc_i_peep + self.bias[0])
            f = torch.sigmoid(cc_f + cc_f_peep + self.bias[1])
            o = torch.sigmoid(cc_o + cc_o_peep + self.bias[2])
            g = torch.tanh(cc_g + self.bias[3])
        else:
            # Standard calculations without peepholes
            i = torch.sigmoid(cc_i + self.bias[0])
            f = torch.sigmoid(cc_f + self.bias[1])
            o = torch.sigmoid(cc_o + self.bias[2])
            g = torch.tanh(cc_g + self.bias[3])

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, height, width):
        hidden_states = (Variable(torch.zeros(batch_size, self.hidden_dim, height, width)),
                         Variable(torch.zeros(batch_size, self.hidden_dim, height, width)))
        # Check if convolution layer is on GPU, if so, put hidden states there too
        # (this avoids having to set a self.is_cuda flag, instead we only have to do nn.Module.cuda() once)
        return [hs.cuda() for hs in hidden_states] if self.conv.weight.data.is_cuda else hidden_states


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 bias=True, return_all_layers=False, peepholes=False, use_cuda=False):
        """
        Multi-layer unrolled Convolutional LSTM implementation.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: int or (int, int) or ((int, int), ...) where len == num_layers
            Size of the convolutional kernel. Can be a single int for square kernel size equal for all layers,
            (int, int) for rectangular kernel equal for all layers, or a fully-specified list of tuples with first
            dimension equal to num_layers.
        bias: bool
            Whether or not to add the bias.
        use_cuda: bool
            Whether or not to put tensors on GPU (using nn.Module.cuda() does not work here, we are initializing
            hidden states during .forward(), which is nice because it gives us a flexible sequence length).
        """
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer_kernel(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer_dim(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias
        self.return_all_layers = return_all_layers
        # Use a ModuleList to make sure modules are discoverable by e.g. .parameters()
        self.cell_list = nn.ModuleList()
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            self.cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias,
                                          peepholes=peepholes))

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor of shape (b, t, c, h, w) where t is sequence length
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """

        # hidden_state kwarg corresponds directly to last_state_list output
        if hidden_state is not None:
            hidden_state = hidden_state
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0),
                                             height=input_tensor.size(-2),
                                             width=input_tensor.size(-1))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, height, width):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, height, width))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer_dim(param, num_layers):
        try:
            len(param)
            # If sequence, return it
            return list(param)
        except TypeError:
            # Extend it to a sequence with num_layers length
            return [param] * num_layers

    @staticmethod
    def _extend_for_multilayer_kernel(param, num_layers):
        try:
            # Verify if it's a sequence
            len(param)
            # It already is a list of tuples per layer
            if np.array(param).ndim == 2:
                return param
            # If not, we need to copy it num_layers times
            return [param] * num_layers
        except TypeError:
            # One value was given for a square kernel size equal for each layer
            return [[param] * 2 for i in range(num_layers)]
