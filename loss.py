import jax.numpy as jnp

from model import VAE
from jax.random import split
from jax import vmap
from jax.scipy.special import logsumexp
from jax.nn import sigmoid
from einops import rearrange

# Typing
from jax.random import PRNGKeyArray
from jax import Array


def kl_divergence(mu, logvar):
    """The KL divergence between a normal distribution and a standard normal distribution."""
    return -0.5 * jnp.sum(1 + logvar - mu ** 2 - jnp.exp(logvar))


def log_prob_bernoulli(x, x_rec):
    """Log probability of a Bernoulli distribution for one sample."""
    return jnp.sum(x * jnp.log(x_rec) + (1 - x) * jnp.log(1 - x_rec), axis=(1, 2, 3))


# Importance weighted loss
def iwae_loss(model: VAE, x: Array, K: int, key: PRNGKeyArray):
    def loss_fn(x: Array, key: PRNGKeyArray):
        x_rec, mean, logvar = model(x, K, key=key)
        log_p_x_z = log_prob_bernoulli(x, sigmoid(x_rec))
        log_q_z_x = -kl_divergence(mean, logvar)
        # log p(x|z) + log p(z) - log q(z|x)
        log_w = log_p_x_z + log_q_z_x
        return -logsumexp(log_w, axis=0) - jnp.log(K)

    keys = split(key, x.shape[0])
    return jnp.mean(vmap(loss_fn)(x, keys))
