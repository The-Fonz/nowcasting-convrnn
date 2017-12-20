import torch
import torch.nn as nn

from convlstm_pytorch import ConvLSTM, ConvLSTMCell
from torch.autograd.variable import Variable


class ConvSeq2Seq(nn.Module):
    
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, peepholes=False, use_cuda=True):
        """
        Sequence-to-sequence convolutional LSTM model.
        """
        super(ConvSeq2Seq, self).__init__()
        
        self.use_cuda = use_cuda
        
        self.encoder = ConvLSTM(input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=True, peepholes=peepholes, use_cuda=use_cuda)
        self.decoder = ConvLSTM(input_size, input_dim, hidden_dim, kernel_size, num_layers,
                         batch_first=True, bias=True, return_all_layers=True, peepholes=peepholes, use_cuda=use_cuda)
        # Default weights should be ok
        self.decoder_output_conv = torch.nn.Conv2d(in_channels=hidden_dim[-1], out_channels=1, kernel_size=(1,1),
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
        # TODO: Test if convolution from all h,c at the same time works better
        preds = [self.decoder_output_conv(last_layer_h[:,timestep]) for timestep in range(dummy_target_inputs.size()[1])]
        # Run through sigmoid to restrict to range [0,1], and make it possible to use cross-entropy loss
        #preds = [torch.nn.functional.hardtanh(p)*.5+.5 for p in preds]
        preds = [torch.sigmoid(p) for p in preds]
        return preds
