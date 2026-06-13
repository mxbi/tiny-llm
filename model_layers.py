# Tokenization
# Embedding -> sequence of fixed-width mapping from tokens (single NxW matrix with lookup table). Let's say W is 4096. This forms the initial stream
# Residual block A:

#     RMSNorm
#     MHA layer:
#         For example, let's say there are 32 attention heads. Each head has three 4096x128 matrices which linearly(?) map the input embedding to Q,K,V. Apply RoPE when calculating Q/K.
#         The output of each head is a weighted sum of all previous V_i (for that head) weighted by softmax(Q_current dot K_i) - why divide by sqrt(size)?.
#         The output of the MHA is the concatenation of each head back to 4096 dim
#         add back to residual Residual Block B:
#                 Linear layer to 16384 with GELU nonlinearity (I want to avoid the complexity of trainable activations to start)
#     Linear layer back to 4096, no activation
#     Add to residual

# These layers get stacked an arbitrary number of times
# Final RMSNorm - why is this important?
# A final linear layer to the token count (softmax applied by the caller)


# Tokenization

import jax.numpy as jnp
import jax

param_key = 0

def mha_block_weights(n_heads, head_size):
    params = {}
    width = n_heads * head_size
    
    # Q: could Q=K
    # Q: Could the embedding dimension be different than the whole width
    # Q: I think that q/k/v shouldn't need biases but maybe I'm wrong
    params['q'] = jnp.random.normal(shape=(width, width)) # TODO: He initialisation
    params['k'] = jnp.random.normal(shape=(width, width))
    params['v'] = jnp.random.normal(shape=(width, width))
    params['rmsnorm_eps'] = jnp.array(0.0001)
    params['rmsnorm_gamma'] = jnp.random.normal(shape=(width))#jnp.array(1.0)

    # TODO: Rope
    return params

def mha_block_forward(params, x, n_heads, head_size):
    # (n_seq, width) -> (n_seq, width)

    # 1. RMSNorm input
    x = x / jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + params['rmsnorm_eps']) * params['rmsnorm_gamma']

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
    qk = jnp.where(qk, jnp.tril(jnp.ones((S, S))), 10**-20)
    # Get weighted-sum weights with softmax
    qk = jax.nn.softmax(qk, axis=-1)
    output = jnp.einsum('hij,jhd->ihd', qk, v)

    # Now we can flatten the heads back
    output = jnp.reshape(output, (S, n_heads*head_size))

    return output




def make_llm(key, n_tokens, width):
    params = {}
    # Embedding weights
    params['embed'] = jnp.random.normal(key, shape=(n_tokens, width))

    def forward(params, x):
        # Embedding layer
        # (n_seq, ) -> (n_seq, width)
        # Our stream starts with the embedded tokens
        stream = jnp.take(params['embed'], x, axis=0)




        # -> (n_seq, n_tokens) logits. At each position i, we're predicting logits for i+1




# def make_embedding(key, n_tokens, width):
#     w = jnp.random.normal(key, shape=(n_tokens, width))
#     def embedding_forward(w, x):
#         # (n_tokens, ) -> (n_tokens, width)
#         return jnp.take(w, x, axis=0)
#     return w, 

# def make_mha_block(key, width):
