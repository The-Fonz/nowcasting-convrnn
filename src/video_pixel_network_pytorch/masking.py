from itertools import accumulate

import torch


def mask(weights, n_channels, n_context, mask_center_pixel_current=False, input_repeats=True,
         n_logits_per_channel=None, n_chunks=None):
    """
    Mask used for all intermediate convolutions. Assumes that input/output stacks are ordered
    channel_1, channel_2, channel_3, n * context_channels, ... (repeat)
    Does not block own channel center pixel. In this example, channel_2 is not masked for all context channels,
    is depedent on all pixels left/top *and* center pixel of channel_3 and channel_2, and only pixels left/top of
    channel_1.

    Args:
        weights: torch.autograd.Variable of weights in shape [out_channels, in_channels, h, w]
        n_channels: Number of channels
        n_context: Number of layers of context
        mask_center_pixel_current: If True, mask the center weight value of the convolutions that connect a layer with
                                   the next timestep version of itself. Used for first convolution.
        input_repeats: If True, it is assumed that input is ordered like R,G,B,context,R,G,(etc)... while if False
                       it is assumed that the channels only occur once at the start and rest is context layers
                       like R,G,B,context... This is useful for the first convolution when channels need to be mixed
                       into the stack, and we can simply put them on top.
        n_logits_per_channel (optional): List of logits to output per channel. If given, needs to have same length
                                         as first dimension of weights. Used for converting stack of layers to logits
        n_chunks: Reset the channel categorization for the output layers after every chunk of layers.
                  Useful when using one big chunked convolution, e.g. for multiple gates of LSTM (which usually has
                  4 chunks, one for each gate).
    """

    n_output_layers = weights.size(0)
    n_input_layers = weights.size(1)

    if n_logits_per_channel and n_channels != len(n_logits_per_channel):
        raise AssertionError("Please specify a list of logits, one value per channel")
    if n_logits_per_channel and n_output_layers != sum(n_logits_per_channel):
        raise AssertionError("n_logits_per_channel={} do not sum to number of output channels={}"
                             .format(n_logits_per_channel, n_output_layers))
    if n_chunks and n_output_layers / n_chunks % 1:
        raise AssertionError("n_chunks={} is not an integer fraction of n_output_layers={}"
                             .format(n_chunks, n_output_layers))

    def is_channel(i, repeat=True, n_logits_per_channel=None, reset_every=None):
        "Returns None if context, else number of channel"
        if reset_every:
            i = i % reset_every
        if repeat and not n_logits_per_channel:
            i = i % (n_channels + n_context)

        if n_logits_per_channel:
            # Find first index where i < a, which is the group it belongs to
            return [i < a for a in accumulate(n_logits_per_channel)].index(True)
        elif i < n_channels:
            return i
        else:
            return None

    # Divide output layers in chunks if needed
    reset_every = n_output_layers // n_chunks if n_chunks else None

    kernel_size = (weights.size(2), weights.size(3))
    h_half = kernel_size[0] // 2
    w_half = kernel_size[1] // 2
    # Mask context and center pixel
    mask_center = torch.zeros(kernel_size)
    if h_half > 0:
        mask_center[:h_half, :] = 1
    if w_half > 0:
        mask_center[h_half, :w_half] = 1
    # Mask just context and don't mask center pixel
    mask_context = mask_center.clone()
    mask_context[h_half, w_half] = 1

    # Tensors have a flag indicating if they're on gpu or not
    if weights.data.is_cuda:
        mask_center = mask_center.cuda()
        mask_context = mask_context.cuda()

    for i_output in range(n_output_layers):
        for i_input in range(n_input_layers):
            i_input_channel = is_channel(i_input, repeat=input_repeats)
            i_output_channel = is_channel(i_output, n_logits_per_channel=n_logits_per_channel,
                                          reset_every=reset_every)
            # If input layer is context, no mask
            if i_input_channel is None:
                pass
            # If output layer is context and input layer is channel, full mask except if we divide into logits
            elif i_output_channel is None:
                if not n_logits_per_channel:
                    weights.data[i_output, i_input] *= 0
            # Allow center pixel to be masked
            elif i_output_channel == i_input_channel and mask_center_pixel_current:
                weights.data[i_output, i_input] *= mask_center
            # If input layer is channel and output layer is same or later channel, mask but keep center pixel
            elif i_output_channel <= i_input_channel:
                weights.data[i_output, i_input] *= mask_context
            # If input layer is channel and output layer is earlier channel
            else:
                weights.data[i_output, i_input] *= mask_center
