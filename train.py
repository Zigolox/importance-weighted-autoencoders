from jax import numpy as jnp
from jax import random, value_and_grad, vmap
from optax import adam, apply_updates
from einops import rearrange
import equinox as eqx
from time import time

from model import LinearIWAE, ConvolutionalIWAE
from loss import vae_loss, iwae_loss, old_iwae_loss
import tensorflow_datasets as tfds

MODEL_KEY = random.PRNGKey(0)

# Load the dataset
dataset = tfds.load("binarized_mnist", split="train", data_dir="data", shuffle_files=True)
ds = dataset.batch(20, drop_remainder=True).cache().shuffle(50_000, seed=0, reshuffle_each_iteration=True)
# dataset = dataset.batch(128).prefetch(1)

# Initialize the model
model = LinearIWAE(key=MODEL_KEY)

# Initialize the optimizer
optim = adam(1e-4)
opt_state = optim.init(eqx.filter(model, eqx.is_array))

# Step function
@eqx.filter_jit
def step(model, x, opt_state, key):
    loss, grad = eqx.filter_value_and_grad(vae_loss)(model, x, 10, key=key)
    updates, opt_state = optim.update(grad, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


# Train the model
for epoch in range(100):
    for x, key in zip(ds.as_numpy_iterator(), random.split(MODEL_KEY, 100)):
        x = jnp.array(x["image"], dtype=jnp.float32)
        x = rearrange(x, "b h w c -> b c h w")
        # Calculate the loss
        # loss = iwae_loss(model, x, 2, key)
        model, opt_state, loss = step(model, x, opt_state, key=key)
        print(loss)
    # print(time())

# Plot the model
import matplotlib.pyplot as plt

x = jnp.array(next(iter(ds.as_numpy_iterator()))["image"], dtype=jnp.float32) / 255
x = rearrange(x, "b h w c -> b c h w")
x = x[:10]

fig, axs = plt.subplots(1, 10, figsize=(10, 1))
for i in range(10):
    axs[i].imshow(x[i, 0], cmap="gray")
    axs[i].axis("off")
plt.show()
