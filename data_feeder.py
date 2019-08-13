from hparams import hp
from torch.utils import data
import numpy as np
import torch

def load_vocab():
    g2idx = {g: idx for idx, g in enumerate(hp.graphemes)}
    idx2g = {idx: g for idx, g in enumerate(hp.graphemes)}

    p2idx = {p: idx for idx, p in enumerate(hp.phonemes)}
    idx2p = {idx: p for idx, p in enumerate(hp.phonemes)}
    return g2idx, idx2g, p2idx, idx2p  # note that g and p mean grapheme and phoneme, respectively.

def prepare_data(dictionary):
    # Tihu persian dictionary
    lexicon_dic = {}
    with open(dictionary) as f:
        for line in f:
            (key, val) = line.strip('\n').split('\t')
            lexicon_dic[key] = val

    words = [" ".join(list(word)) for word, prons in lexicon_dic.items()]
    prons = [" ".join(prons) for word, prons in lexicon_dic.items()]
    indices = list(range(len(words)))
    from random import shuffle
    shuffle(indices)
    words = [words[idx] for idx in indices]
    prons = [prons[idx] for idx in indices]
    num_train, num_test = int(len(words) * .8), int(len(words) * .1)
    train_words, eval_words, test_words = words[:num_train], \
                                          words[num_train:-num_test], \
                                          words[-num_test:]
    train_prons, eval_prons, test_prons = prons[:num_train], \
                                          prons[num_train:-num_test], \
                                          prons[-num_test:]
    return train_words, eval_words, test_words, train_prons, eval_prons, test_prons

def drop_lengthy_samples(words, prons, enc_maxlen, dec_maxlen):
    """We only include such samples less than maxlen."""
    _words, _prons = [], []
    for w, p in zip(words, prons):
        if len(w.split()) + 1 > enc_maxlen: continue
        if len(p.split()) + 1 > dec_maxlen: continue  # 1: <EOS>
        _words.append(w)
        _prons.append(p)
    return _words, _prons

def encode(inp, type, dict):
    '''convert string into ids
    type: "x" or "y"
    dict: g2idx for 'x', p2idx for 'y'
    '''
    if type == "x":
        tokens = inp.split() + ["</s>"]
    else:
        tokens = ["<s>"] + inp.split() + ["</s>"]

    x = [dict.get(t, dict["<unk>"]) for t in tokens]
    return x

class G2pDataset(data.Dataset):

    def __init__(self, words, prons, g2idx, p2idx):
        """
        words: list of words. e.g., ["w o r d", ]
        prons: list of prons. e.g., ['W ER1 D',]
        """
        self.words = words
        self.prons = prons
        self.g2idx = g2idx
        self.p2idx = p2idx

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word, pron = self.words[idx], self.prons[idx]
        x = encode(word, "x", self.g2idx)
        y = encode(pron, "y", self.p2idx)
        decoder_input, y = y[:-1], y[1:]

        x_seqlen, y_seqlen = len(x), len(y)
        return x, x_seqlen, word, decoder_input, y, y_seqlen, pron

def pad(batch):
    '''Pads zeros such that the length of all samples in a batch is the same.'''
    f = lambda x: [sample[x] for sample in batch]
    x_seqlens = f(1)
    y_seqlens = f(5)
    words = f(2)
    prons = f(-1)

    x_maxlen = np.array(x_seqlens).max()
    y_maxlen = np.array(y_seqlens).max()

    f = lambda x, maxlen, batch: [sample[x] + [0] * (maxlen - len(sample[x])) for sample in batch]
    x = f(0, x_maxlen, batch)
    decoder_inputs = f(3, y_maxlen, batch)
    y = f(4, y_maxlen, batch)

    f = torch.LongTensor
    return f(x), x_seqlens, words, f(decoder_inputs), f(y), y_seqlens, prons

def convert_ids_to_phonemes(ids, idx2p):
    phonemes = []
    for idx in ids:
        if idx == 3:  # 3: </s>
            break
        p = idx2p[idx]
        phonemes.append(p)
    return phonemes
