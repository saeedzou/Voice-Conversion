# Vocoders

This directory contains the implementations of the following vocoders:

- [Hifi-GAN](https://github.com/jik876/hifi-gan)
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)

## Hifi-GAN

The Hifi-GAN vocoder consists of a generator and two discriminators: a multi-scale and multi-period discriminator.

The generator is a fully convolutional neural network.

`weight_norm` is used in the generator to stabilize training. However, it is removed after training because it is not needed for inference.

The Multi-Receptive Field (MRF) module is used in the generator to capture long-range dependencies. It consists of a stack of residual dilated convolutions with different dilation rates.