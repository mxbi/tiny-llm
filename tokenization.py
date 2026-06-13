class CharTokenizer():
    def __init__(self, sample: str):
        self.tokens = {
            '<bos>': 0, # Beginning of sequence
            '<eos>': 1, # End of sequence
            '<unk>': 2  # Unknown token
        }
        self.next_token = len(self.tokens) + 1

        for char in sample:
            if char not in self.tokens:
                self.tokens[char] = self.next_token
                self.next_token += 1

        self.detokens = {t: c for c, t in self.tokens.items()}

    def tokenize(self, string, add_bos=True, add_eos=True):
        tokens = []
        if add_bos:
            tokens.append(self.tokens['<bos>'])
        for char in string:
            if char in self.tokens:
                tokens.append(self.tokens[char])
            else:
                tokens.append(self.tokens['<unk>'])
        if add_eos:
            tokens.append(self.tokens['<eos>'])
        return tokens

    def untokenize(self, tokens):
        output = ''
        for token in tokens:
            output += self.detokens[token]
        return output

if __name__ == '__main__':
    import string
    tokenizer = CharTokenizer(string.ascii_letters + string.digits + ' !')
    
    my_string = 'Hello there!'
    tokens = tokenizer.tokenize(my_string)
    print(tokens)
    print(tokenizer.untokenize(tokens))