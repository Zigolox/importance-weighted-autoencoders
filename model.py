import jax.numpy as jnp

# Froms
from equinox.nn import Sequential, Linear, Lambda
from equinox import Module
from jax.nn import celu
from jax.random import split, normal
from jax import vmap
from einops import rearrange

# Types
from jax.random import PRNGKeyArray
from jax import Array
from typing import Tuple

# Create layers from activation functions

celu = Lambda(celu)


# Variational autoencoder

# Encoder
class Encoder(Sequential):
    def __init__(self, key: PRNGKeyArray):
        keys = split(key, 3)
        super().__init__(
            [
                Linear(784, 784, key=keys[0]),
                celu,
                Linear(784, 784, key=keys[1]),
                celu,
                Linear(784, 784 * 2, key=keys[2]),
                Lambda(lambda x: jnp.split(x, 2, axis=-1)),
            ]
        )


# Decoder
class Decoder(Sequential):
    def __init__(self, key: PRNGKeyArray):
        keys = split(key, 3)
        super().__init__(
            [
                Linear(784, 784, key=keys[0]),
                celu,
                Linear(784, 784, key=keys[1]),
                celu,
                Linear(784, 784, key=keys[2]),
                celu,
            ]
        )


# VAE
class VAE(Module):
    encoder: Encoder
    decoder: Decoder

    def __init__(self, key: PRNGKeyArray):
        enc_key, dec_key = split(key)
        self.encoder = Encoder(enc_key)
        self.decoder = Decoder(dec_key)

    def __call__(self, x, K: int, key: PRNGKeyArray):

        # Encode
        mean, logvar = self.encoder(x)

        # Sample
        z = self.sample(mean, logvar, (K, *mean.shape), key)

        # Decode
        x = vmap(self.decoder)(z)

        return x, mean, logvar

    def sample(self, mean: Array, logvar: Array, shape: Tuple, key: PRNGKeyArray):
        std = jnp.exp(0.5 * logvar)
        eps = normal(key, shape=shape)
        # Reparameterization trick
        return eps * std + mean
