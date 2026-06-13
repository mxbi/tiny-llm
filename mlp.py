from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import jax.numpy as jnp
import jax
import optax
import numpy as np

from utils import batcher

N_CLASSES = 4
# Make data
X, y = make_classification(n_samples=100000, n_features=20, n_classes=N_CLASSES, random_state=4, n_informative=10)
x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=4)

x_train = jnp.array(x_train)
x_valid = jnp.array(x_valid)
y_train = jnp.array(y_train)
y_valid = jnp.array(y_valid)

# MLP: 20 -> 100 -> 4

def make_mlp(sizes, key):
    params = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        key, w_key = jax.random.split(key)
        params.append({
            "W": jax.random.normal(w_key, shape=(fan_in, fan_out)) * jnp.sqrt(2/fan_in),
            "b": jnp.zeros(shape=(fan_out,)),
        })
    
    def forward(params, x):
        for i, layer in enumerate(params):
            x = x @ layer["W"] + layer["b"]
            if i < len(params) - 1:
                x = jax.nn.gelu(x)
        return x
    return params, forward

# Define params
key = jax.random.PRNGKey(4)
params, forward = make_mlp([20, 100, 100, N_CLASSES], key)

batched_forward = jax.vmap(forward, in_axes=(None, 0))

@jax.jit
def mce_loss(params, X, y):
    p = batched_forward(params, X)
    log_p = jax.nn.log_softmax(p, axis=1)
    loss = -jnp.sum(jax.nn.one_hot(y, N_CLASSES) * log_p) / X.shape[0]

    acc = jnp.mean(jnp.argmax(p, axis=1) == y)
    return loss, acc

optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(params)

@jax.jit
def step(params, X, Y, opt_state):
    (loss, acc), grads = jax.value_and_grad(mce_loss, has_aux=True)(params, X, Y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, acc

for epoch in range(10):
    print(f"Epoch {epoch}", end=" ")
    losses = []
    accs = []
    for X_batch, y_batch in batcher(x_train, y_train):
        params, opt_state, loss, batch_acc = step(params, X_batch, y_batch, opt_state)
        losses.append(loss)
        accs.append(batch_acc)

    print(f"- train-loss: {np.mean(losses):.4f} - train-acc: {np.mean(accs):.4f}", end=" ")
    val_loss, val_acc = mce_loss(params, x_valid, y_valid)
    print(f"- val-loss: {val_loss:.4f} - val-acc: {val_acc:.4f}")