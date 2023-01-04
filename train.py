from jax import numpy as jnp
from jax import random, value_and_grad, vmap
from optax import adam, apply_updates
from einops import rearrange
import equinox as eqx

from model import LinearIWAE, ConvoluationalIWAE
from loss import iwae_loss
import tensorflow_datasets as tfds

MODEL_KEY = random.PRNGKey(0)

# Load the dataset
dataset = tfds.load("mnist", split="train", data_dir="data", shuffle_files=True)
ds = dataset.shuffle(1024).batch(128).prefetch(1)
# dataset = dataset.batch(128).prefetch(1)

# Initialize the model
model = ConvoluationalIWAE(key=MODEL_KEY)

# Initialize the optimizer
optim = adam(1e-4)
opt_state = optim.init(eqx.filter(model, eqx.is_array))

# Step function
def step(model, x, opt_state, key):
    loss, grad = eqx.filter_value_and_grad(iwae_loss)(model, x, 1, key=key)
    updates, opt_state = optim.update(grad, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


# Train the model
for x, key in zip(ds.take(100).as_numpy_iterator(), random.split(MODEL_KEY, 100)):
    x = jnp.array(x["image"], dtype=jnp.float32) / 255
    x = rearrange(x, "b h w c -> b c h w")
    print(x.shape)
    model, opt_state, loss = step(model, x, opt_state, key=key)
    print(loss)
