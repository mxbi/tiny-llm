import jax.numpy as jnp
import jax
import optax
from orbax.checkpoint import v1 as ocp
import os

from transformer import make_llm
from tokenization import CharTokenizer
from data import data_iterator, load_tinyshakespeare

N_HEADS = 16
HEAD_SIZE = 32
N_BLOCKS = 8
BATCH_SIZE = 32
CONTEXT_SIZE = 32

VALIDATION_BATCH_SIZE = 256
VALIDATION_STEPS = 100

CHECKPOINT_STEPS = 1000

import wandb
run = wandb.init(entity="mxbi", project="teeny-llm")

# Crash when producing a NaN
jax.config.update("jax_debug_nans", True)

dataset = load_tinyshakespeare()
tokenizer = CharTokenizer(dataset[0])
n_vocab = len(tokenizer.tokens)
tokenized_data = [tokenizer.tokenize(data) for data in dataset]

total_len = len(tokenized_data[0])
train_data = [tokenized_data[0][:(total_len*9//10)]]
val_data = [tokenized_data[0][(total_len*9//10):]]

key = jax.random.PRNGKey(4)
params, forward = make_llm(key, n_vocab, HEAD_SIZE, N_HEADS, N_BLOCKS)
def get_size(params):
    if isinstance(params, dict):
        return sum(get_size(v) for v in params.values())# {k: get_size(v) for k,v in params.items()}
    return params.size
print(f'Model size: {get_size(params)} params, data size: {len(dataset[0])} tokens, {n_vocab} vocab')

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

train_iter = data_iterator(train_data, batch_size=BATCH_SIZE, context_size=CONTEXT_SIZE)
val_iter = data_iterator(val_data, batch_size=VALIDATION_BATCH_SIZE, context_size=CONTEXT_SIZE)

tok = 0
for i, x_batch in enumerate(train_iter):
    tok += x_batch.shape[0] * (x_batch.shape[1] - 1)
    params, opt_state, loss, batch_acc = step(params, x_batch, opt_state)
    print(f'Batch {i} - loss: {loss:.4f} - acc: {batch_acc:.4f} - tok: {tok}')
    wandb.log({
        "loss": loss,
        "acc": batch_acc,
        "tok": tok,
    })

    if i % VALIDATION_STEPS == 0:
        val_loss, val_acc = mce_loss(params, next(val_iter))
        print(f'Valid - loss: {val_loss:.4f} - acc: {val_acc:.4f}')
        wandb.log({
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

    if i % CHECKPOINT_STEPS == 0:
        print(f'Checkpoint {i}')
        ocp.save(os.path.abspath(f'models/{wandb.run.id}_{i}'), {'params': params, 'opt_state': opt_state})
        wandb.log({"checkpoint": i})