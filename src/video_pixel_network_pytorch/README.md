# Video Pixel Network in PyTorch

This is my implementation of the fascinating [Video Pixel Networks](https://arxiv.org/abs/1610.00527), which improves upon earlier sequence prediction methods by conditioning on the pixels in the current frame that it is predicting, so any pixel will depend on all the pixels in the past but also on the already predicted pixels in the current frame. The ordering can be arbitrary but they choose left-right top-bottom.

They achieve state of the art on Moving MNIST, nearing the theoretical bound, and interesting results on video sequence prediction.

## Getting started

Some simple unittests are in `./test.py`, run it with something like `python -m unittests video_pixel_network_pytorch.test`.

TODO: Test Encoder+Decoder, train on ball dataset.

## Implementation details

The original paper omitted many implementation details. The next subheadings contain some decisions I had to make and some open problems. I looked at the [Tensorflow implementation](https://github.com/3ammor/Video-Pixel-Networks) which is not perfect but helped me by seeing what decisions that author made.

### Decoder

The *first block of the decoder* needs to somehow change the number of channels of the input image. It can be done by first using a 1x1 convolution to map the image to the number of channels used between the RMBs, and then input it into the RMB and treat it like any other. The other method is to treat the first RMB as a special block, and use the first 1x1 convolution of the RMB to map the image to the number of channels used internally in the RMB. The skip connection does not work with multiple channels for the latter method. I chose the latter method.

It is not clear to me whether the LSTM uses *peephole connections* (state C is observable by the gates). Need to test both.

The *number of layers* of the LSTM that integrates encoder RMB outputs in time is not clearly stated, I assume it is 1, which is unintuitive as I'd assume that multiple layers would perform better with the time integration.

*Conditioning* the decoder on the current frame can be done in many ways. I chose the same way as the Tensorflow implementation, by condensing the input tensor in the RMB to `n_rmb_internal_channels - n_img_channel` channels with the 1x1 convolution, then concatenating the "image" channels. I can think of other ways but this at least makes sure that the current image is present as a strong signal in the first RMB. This might have to change when I figure out how to do multi-channel prediction.

### Multi-channel prediction

I'm not sure how to do multi-channel prediction. The current implementation only does single-channel prediction. With multi-channel prediction part of the stack of channels in the decoder would be assigned to each channel, where the first channel would not have any knowledge of other channels, and the last channel would know of the chosen pixel values of all the other channels.

The [PixelCNN paper](https://arxiv.org/abs/1606.05328) says *"For each pixel the three colour channels (R, G, B) are modelled successively, with B conditioned on (R, G), and G conditioned on R. This is achieved by splitting the feature maps at every layer of the network into three and adjusting the centre values of the mask tensors"*. I don't fully understand this because in principle the last channel can have knowledge of all the other channels in all the convolutions. Dividing the feature maps in three (in the channel dimension I guess) might not give equal representational power to each layer. Also, the three groups of feature maps will have to be kept separate through the entire pipeline to avoid mixing of lower image channels to higher image channels, where the higher image channels can know about the *current pixel value* of lower image channels in the convolutions.

### Blind spot problem

The [PixelCNN paper](https://arxiv.org/abs/1606.05328) shows that there is a blind spot problem with stacked layers of convolutions. They solve this by using one vertical and one horizontal stack, but there is no mention of this problem or a solution in the VPN paper. 

### Conditioning on things like state/action vectors or one-hot labels

This is interesting as it might let one condition on things like label. I'm not sure how to do this though. In the original paper they note (section 6.1) that they condition on state and action vectors by applying a 1x1 convolution to the action and state vector (presumably to make them have the same number of channels as the MUs) and broadcast these to all 64x64 positions for all layers in the encoders and decoders. In the [PixelCNN paper](https://arxiv.org/abs/1606.05328) they elaborate on this in section 2.3 where they propose to add similar non-location-dependent bias-like terms to the layers before the activation functions. They also describe how to add location dependent features, basically using a deconvolutional network to make a feature map with the same dimensions and use a 1x1 convolution on it and add it to the terms before the nonlinearities.
