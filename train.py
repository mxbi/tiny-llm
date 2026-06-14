import jax.numpy as jnp
import jax
import optax
import numpy as np

from transformer import make_llm
from tokenization import CharTokenizer
from data import data_iterator, load_tinyshakespeare

import wandb
run = wandb.init(entity="mxbi", project="teeny-llm")

# Crash when producing a NaN
jax.config.update("jax_debug_nans", True)

dataset = load_tinyshakespeare()
tokenizer = CharTokenizer(dataset[0])
n_vocab = len(tokenizer.tokens)

key = jax.random.PRNGKey(4)
params, forward = make_llm(key, n_vocab, 32, 16, 8)
def get_size(params):
    if isinstance(params, dict):
        return sum(get_size(v) for v in params.values())# {k: get_size(v) for k,v in params.items()}
    return params.size
print(f'Model size: {get_size(params)} params, data size: {len(dataset[0])} tokens')

batched_forward = jax.vmap(forward, in_axes=(None, 0))

@jax.jit
def mce_loss(params, X):
    p = batched_forward(params, X[:, :-1])
    log_p = jax.nn.log_softmax(p, axis=-1)
    loss = -jnp.sum(jax.nn.one_hot(X[:, 1:], n_vocab) * log_p) / X.shape[0] / (X.shape[1] - 1)

    acc = jnp.mean(jnp.argmax(p, axis=-1) == X[:, 1:])
    return loss, acc

schedule = 0.001#optax.linear_schedule(0.0, 3e-4, transition_steps=200)
optimizer = optax.chain(
    optax.zero_nans(),
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=schedule),
)
opt_state = optimizer.init(params)

@jax.jit
def step(params, X, opt_state):
    (loss, acc), grads = jax.value_and_grad(mce_loss, has_aux=True)(params, X)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, acc

    # print(f"Epoch {epoch}", end=" ")
losses = []
accs = []
tok = 0
for i, x_batch in enumerate(data_iterator([tokenizer.tokenize(dataset[0])], batch_size=32, context_size=32)):
    tok += x_batch.shape[0] * (x_batch.shape[1] - 1)
    params, opt_state, loss, batch_acc = step(params, x_batch, opt_state)
    # print(loss, batch_acc)
    print(f'Batch {i} - loss: {loss:.4f} - acc: {batch_acc:.4f} - tok: {tok}')
    wandb.log({
        "loss": loss,
        "acc": batch_acc,
        "tok": tok,
    })

    losses.append(loss)
    accs.append(batch_acc)