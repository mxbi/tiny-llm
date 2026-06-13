import jax.numpy as jnp
import jax
import optax
import numpy as np

from transformer import make_llm
from tokenization import CharTokenizer
from data import data_iterator, load_tinyshakespeare

dataset = load_tinyshakespeare()
tokenizer = CharTokenizer(dataset[0])
n_vocab = len(tokenizer.tokens)

key = jax.random.PRNGKey(4)
params, forward = make_llm(key, n_vocab, 32, 16, 8)
def get_size(params):
    if isinstance(params, dict):
        return {k: get_size(v) for k,v in params.items()}
    return params.size
print(get_size(params))

batched_forward = jax.vmap(forward, in_axes=(None, 0))

@jax.jit
def mce_loss(params, X):
    p = batched_forward(params, X[:, :-1])
    log_p = jax.nn.log_softmax(p, axis=-1)
    loss = -jnp.sum(jax.nn.one_hot(X[:, 1:], n_vocab) * log_p) / X.shape[0] / (X.shape[1] - 1)

    acc = jnp.mean(jnp.argmax(p, axis=-1) == X[:, 1:])
    return loss, acc

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=0.001),
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
for i, x_batch in enumerate(data_iterator([tokenizer.tokenize(dataset[0])], batch_size=64, context_size=32)):
    print(f'Batch {i}')
    params, opt_state, loss, batch_acc = step(params, x_batch, opt_state)
    print(loss, batch_acc)

    losses.append(loss)
    accs.append(batch_acc)