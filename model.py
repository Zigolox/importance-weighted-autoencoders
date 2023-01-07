import jax.numpy as jnp

# Froms
from equinox.nn import Sequential, Linear, Lambda, Conv2d, ConvTranspose2d
from equinox import Module
from jax.nn import celu
from jax.random import split, normal
from jax import vmap
from einops import rearrange
from functools import partial

# Types
from jax.random import PRNGKeyArray
from jax import Array
from typing import Tuple

# Create layers from activation functions

Celu = Lambda(celu)
Tanh = Lambda(jnp.tanh)

Flatten = Lambda(partial(rearrange, pattern='c h w -> (c h w)'))
Split = Lambda(partial(rearrange, pattern='(p c) -> p c', p=2))
CreateGrid = Lambda(partial(rearrange, pattern='(c h w) -> c h w', c=512, h=2, w=2))
Unflatten = Lambda(partial(rearrange, pattern='(c h w) -> c h w', h=28, w=28))
# Variational autoencoder

# Encoder
class ConvolutionalEncoder(Sequential):
    def __init__(self, key: PRNGKeyArray):
        keys = split(key, 6)
        super().__init__(
            [
                Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), key=keys[0]),
                Celu,
                Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), key=keys[1]),
                Celu,
                Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), key=keys[2]),
                Celu,
                Conv2d(128, 256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), key=keys[3]),
                Celu,
                Conv2d(256, 512, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), key=keys[4]),
                Celu,
                Flatten,
                Linear(2048, 4096, key=keys[5]),
                Split,
            ]
        )


# Decoder
class ConvolutionalDecoder(Sequential):
    def __init__(self, key: PRNGKeyArray):
        keys = split(key, 3)
        super().__init__(
            [
                CreateGrid,
                ConvTranspose2d(512, 256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1), key=keys[0]),
                Celu,
                ConvTranspose2d(256, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1), key=keys[1]),
                Celu,
                ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1), key=keys[2]),
                Celu,
                ConvTranspose2d(64, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1), key=keys[3]),
                Celu,
                ConvTranspose2d(32, 1, kernel_size=(5, 5), stride=(1, 1), padding=(4, 4), key=keys[4]),
            ]
        )


# VAE
class IWAE(Module):
    encoder: ConvolutionalEncoder
    decoder: ConvolutionalDecoder

    def __init__(self, key: PRNGKeyArray):
        enc_key, dec_key = split(key)
        self.encoder = ConvolutionalEncoder(enc_key)
        self.decoder = ConvolutionalDecoder(dec_key)

    def __call__(self, x, K: int, key: PRNGKeyArray):

        # Encode
        mean, logvar = self.encoder(x)

        # Sample
        z = self.sample(mean, logvar, (K, *mean.shape), key)

        # Decode
        x = vmap(self.decoder)(z)

        return x, z, mean, logvar

    def sample(self, mean: Array, logvar: Array, shape: Tuple, key: PRNGKeyArray):
        std = jnp.exp(0.5 * logvar)
        eps = normal(key, shape=shape)
        # Reparameterization trick
        return eps * std + mean


class LinearEncoder(Sequential):
    def __init__(self, key: PRNGKeyArray):
        keys = split(key, 3)
        super().__init__(
            [
                Flatten,
                Linear(784, 200, key=keys[0]),
                Tanh,
                Linear(200, 200, key=keys[1]),
                Tanh,
                Linear(200, 100, key=keys[2]),
                Split,
            ]
        )


class LinearDecoder(Sequential):
    def __init__(self, key: PRNGKeyArray):
        keys = split(key, 2)
        super().__init__(
            [
                Linear(50, 200, key=keys[0]),
                Tanh,
                Linear(200, 200, key=keys[1]),
                Tanh,
                Linear(200, 784, key=keys[2]),
                Unflatten,
            ]
        )


class LinearIWAE(IWAE):
    encoder: LinearEncoder
    decoder: LinearDecoder

    def __init__(self, key: PRNGKeyArray):
        enc_key, dec_key = split(key)
        self.encoder = LinearEncoder(enc_key)
        self.decoder = LinearDecoder(dec_key)


class ConvolutionalIWAE(IWAE):
    encoder: ConvolutionalEncoder
    decoder: ConvolutionalDecoder

    def __init__(self, key: PRNGKeyArray):
        enc_key, dec_key = split(key)
        self.encoder = ConvolutionalEncoder(enc_key)
        self.decoder = ConvolutionalDecoder(dec_key)
