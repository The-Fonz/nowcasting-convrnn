import torch
import torch.nn as nn

from convlstm_pytorch import ConvLSTM, ConvLSTMCell
from torch.autograd.variable import Variable


class ConvSeq2Seq(nn.Module):
    
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 peepholes=False, fullstack_output_conv=False, use_cuda=True):
        """
        Sequence-to-sequence convolutional LSTM model.
        
        :param input_size: w*h of input
        :param input_dim: Number of channels of input
        :param hidden_dim: List of channels of hidden dimension for each layer
        :param kernel_size: Size of kernel in one tuple [w,h] or list of tuples (per layer)
        :param num_layers: Number of layers
        :kwarg peepholes: Whether to use peepholes (i,f dependent on c_cur with convolution,
                                                    and o dependent on c_cur elementwise)
        :fullstack_output_conv: Whether to convolute the entire stack of states to make the output
                                or just the top hidden state
        :use_cuda: We must know whether to use CUDA when making state tensors
        """
        super(ConvSeq2Seq, self).__init__()
        
        self.fullstack_output_conv = fullstack_output_conv
        self.use_cuda = use_cuda
        
        self.encoder = ConvLSTM(input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=True, peepholes=peepholes, use_cuda=use_cuda)
        self.decoder = ConvLSTM(input_size, input_dim, hidden_dim, kernel_size, num_layers,
                         batch_first=True, bias=True, return_all_layers=True, peepholes=peepholes, use_cuda=use_cuda)
        
        if self.fullstack_output_conv:
            # Takes all hidden states h
            in_channels = sum(hidden_dim)
        else:
            in_channels = hidden_dim[-1]
        # Default weights should be ok
        self.decoder_output_conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=(1,1),
                                             padding=0, bias=True)

    def forward(self, inputs_var, n_targets=3):
        # Compute encoded state
        # We don't care about outputs other than the last one
        enc_layer_output_list, enc_last_state_list = self.encoder.forward(inputs_var)

        # Use list to make Size (subclass of tuple) mutable
        s = list(inputs_var.size())
        # Adjust number of steps to take
        s[1] = n_targets
        dummy_target_inputs = Variable(torch.zeros(s))

        if self.use_cuda:
            dummy_target_inputs = dummy_target_inputs.cuda()

        # Compute time series using encoded state (not that it's passed to decoder)
        # No conditioning on own outputs
        dec_layer_output_list, dec_last_state_list = self.decoder.forward(dummy_target_inputs, hidden_state=enc_last_state_list)

        # Get highest layer h
        last_layer_h = dec_layer_output_list[-1]
        # Map to output using 1x1 convolution
        # preds is ordered list of predictions
        if self.fullstack_output_conv:
            # Use all layer outputs h per timestep
            preds = [self.decoder_output_conv(dec_layer_output_list[:,timestep]) for timestep in range(dummy_target_inputs.size()[1])]
        else:
            # Only use output h of last layer
            preds = [self.decoder_output_conv(last_layer_h[:,timestep]) for timestep in range(dummy_target_inputs.size()[1])]
        # Run through sigmoid to restrict to range [0,1], and make it possible to use cross-entropy loss
        #preds = [torch.nn.functional.hardtanh(p)*.5+.5 for p in preds]
        preds = [torch.sigmoid(p) for p in preds]
        return preds
