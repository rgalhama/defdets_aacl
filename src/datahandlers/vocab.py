from torchtext.vocab import Vocab
from collections import Counter


def build_vocab_from_text(sentence_reader, tokenizer):
    counter = Counter()
    for line in sentence_reader:
        counter.update(tokenizer(line))
    return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

def build_vocab_from_tokenized(tokenized_sentences):
    counter = Counter()
    for s in tokenized_sentences:
        counter.update(s)
    return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
