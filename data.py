import jax.numpy as jnp
import random

def load_tinyshakespeare():
    # One very long sample here
    return [open('tinyshakespeare.txt','r').read().strip()]

def data_iterator(tokenized, batch_size=64, context_size=512):

    while True: # infinite batch generator
        # sample samples
        batch = []

        for _ in range(batch_size):
            sample = random.choice(tokenized)
            max_idx = len(sample) - context_size # TODO: make this work nice with short sequences
            start_idx = random.randint(0, max(0, max_idx))
            batch.append(sample[start_idx:start_idx+context_size])

        yield jnp.array(batch)

if __name__ == '__main__':
    from tokenization import CharTokenizer
    data = load_tinyshakespeare()

    tokenizer = CharTokenizer(data[0])
    data_t = [tokenizer.tokenize(data[0])]

    it = data_iterator(data_t)

    print(next(it))