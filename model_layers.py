# Tokenization
# Embedding -> sequence of fixed-width mapping from tokens (single NxW matrix with lookup table). Let's say W is 4096. This forms the initial stream
# Residual block A:

#     RMSNorm
#     MHA layer:
#         For example, let's say there are 32 attention heads. Each head has three 4096x128 matrices which linearly(?) map the input embedding to Q,K,V. Apply RoPE when calculating Q/K.
#         The output of each head is a weighted sum of all previous V_i (for that head) weighted by softmax(Q_current dot K_i) - why divide by sqrt(size)?.
#         The output of the MHA is the concatenation of each head back to 4096 dim
#         add back to residual 
#     Residual Block B:
#       Linear layer to 16384 with GELU nonlinearity (I want to avoid the complexity of trainable activations to start)
#       Linear layer back to 4096, no activation
#     Add to residual

# These layers get stacked an arbitrary number of times
# Final RMSNorm - important because everything else is operating on rmsnorm input - we don't control the scale of the stream
# A final linear layer to the token count (softmax applied by the caller)


# Tokenization

import jax.numpy as jnp
import jax
from jax.random import PRNGKey

def rmsnorm_weights(width):
    return {
        'eps': jnp.array(0.0001),
        'gamma': jnp.ones((width, ))
    }

def rmsnorm(params, x):
    return x / jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + params['eps']) * params['gamma']

def mha_block_weights(key, n_heads, head_size):
    params = {}
    width = n_heads * head_size
    
    # Q: could Q=K
    # Q: Could the embedding dimension be different than the whole width
    # Q: I think that q/k/v shouldn't need biases but maybe I'm wrong
    keys = jax.random.split(keys, 4)
    params['q'] = jax.random.normal(keys[0], shape=(width, width)) / jnp.sqrt(2/width)
    params['k'] = jax.random.normal(keys[1], shape=(width, width)) / jnp.sqrt(2/width)
    params['v'] = jax.random.normal(keys[2], shape=(width, width)) / jnp.sqrt(2/width)
    params['rmsnorm'] = rmsnorm_weights(width)

    # TODO: Rope
    return params

def mha_block_forward(params, x, n_heads, head_size):
    # (n_seq, width) -> (n_seq, width)

    # 1. RMSNorm input
    x = rmsnorm(params['rmsnorm'], x)
    # 2. Compute Q,K,V, and reshape to the individual heads
    S = x.shape[0]
    q = jnp.reshape(x @ params['q'], (S, n_heads, head_size)) # ihd
    k = jnp.reshape(x @ params['k'], (S, n_heads, head_size)) # jhd
    v = jnp.reshape(x @ params['v'], (S, n_heads, head_size)) # jhd
    # (n_seq, heads, head_size)

    # hij means: for each head, an (i,j) matrix of attention scores
    # i is the current token and j is the token being attended to
    qk = jnp.einsum('ihd,jhd->hij', q, k) / jnp.sqrt(head_size)
    # Mask to avoid attending to future tokens (attending to current token seems ok?
    qk = jnp.where(jnp.tril(jnp.ones((S, S))), qk, 10**-20)
    # Get weighted-sum weights with softmax
    qk = jax.nn.softmax(qk, axis=-1)
    output = jnp.einsum('hij,jhd->ihd', qk, v)

    # Now we can flatten the heads back
    output = jnp.reshape(output, (S, n_heads*head_size))

    return output

def ffa_block_weights(key: PRNGKey, in_size, hidden_size):
    keys = jax.random.split(key, 2)

    params = {}
    params['rmsnorm'] = rmsnorm_weights(in_size)
    params['w1'] = jax.random.normal(keys[0], shape=(in_size, hidden_size)) * jnp.sqrt(2/in_size)
    params['b1'] = jnp.zeros(shape=(hidden_size, ))
    params['w2'] = jax.random.normal(keys[1], shape=(hidden_size, in_size)) * jnp.sqrt(2/hidden_size)
    params['b2'] = jnp.zeros(shape=(in_size, ))
    
    return params

def ffa_block_forward(params, x):
    x = rmsnorm(params['rmsnorm'], x)

    hidden = jnp.einsum('sd,dh->sh', x, params['w1']) + params['b1']
    hidden = jax.nn.gelu(hidden)

    out = jnp.einsum('sd,dh->sh', x, params['w2']) + params['b2']
    return out

def make_llm(key, n_tokens, head_size, n_heads, n_blocks):
    width = head_size * n_heads
    params = {}
    # Embedding weights
    key, embed_key = jax.random.split(key)
    params['rmsnorm'] = rmsnorm_weights(width)
    params['embed'] = jax.random.normal(embed_key, shape=(n_tokens, width)) * 0.02

    for i in range(n_blocks):
        key, mha_key, ffa_key = jax.random.split(key, 3)
        params[f'mha_{i}'] = mha_block_weights(mha_key, n_heads, head_size)
        params[f'ffa_{i}'] = ffa_block_weights(ffa_key, width, width * 2)

    def forward(params, x):
        # Embedding layer
        # (n_seq, ) -> (n_seq, width)
        # Our stream starts with the embedded tokens
        stream = jnp.take(params['embed'], x, axis=0)

        for i in range(n_blocks):
            stream += mha_block_forward(params[f'mha_{i}'], stream, n_heads, head_size)
            stream += ffa_block_forward(params[f'ffa_{i}'], stream)

        stream = rmsnorm(params['rmsnorm'], stream)
        
        # unembed layer
        # (n_seq, width) -> (n_seq, vocab)
        output = jnp.einsum('sd,vd->sv', stream, params['embed'])

        return output
        
    return params, forward