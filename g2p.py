# -*- coding: utf-8 -*-
# /usr/bin/python
'''
By kyubyong park(kbpark.linguist@gmail.com) and Jongseok Kim(https://github.com/ozmig77)
https://www.github.com/kyubyong/g2p
'''
import nltk
import argparse
import numpy as np
import codecs
import os
import re
from builtins import str as unicode
import hazm
from expand import normalize_numbers
from hparams import hp
from data_feeder import load_vocab

dirname = os.path.dirname(__file__)

def construct_homograph_dictionary():
    f = os.path.join(dirname,'homographs.en')
    homograph2features = dict()
    for line in codecs.open(f, 'r', 'utf8').read().splitlines():
        if line.startswith("#"): continue # comment
        headword, pron1, pron2, pos1 = line.strip().split("|")
        homograph2features[headword.lower()] = (pron1.split(), pron2.split(), pos1)
    return homograph2features

# def segment(text):
#     '''
#     Splits text into `tokens`.
#     :param text: A string.
#     :return: A list of tokens (string).
#     '''
#     print(text)
#     text = re.sub('([.,?!]( |$))', r' \1', text)
#     print(text)
#     return text.split()

class G2p(object):
    def __init__(self, args):
        super().__init__()
        # self.graphemes = ["<pad>", "<unk>", "</s>"] + list("آئابتثجحخدذرزسشصضطظعغفقلمنهوپچژکگی")
        self.graphemes = hp.graphemes
        self.phonemes = hp.phonemes
        self.g2idx, self.idx2g, self.p2idx, self.idx2p = load_vocab()
        self.checkpoint = args.checkpoint
        # load Tihu dictionary as the Persian lexicon
        tihu = {}
        with open("tihudict.dict") as f:
            for line in f:
                (key, val) = line.strip('\n').split('\t')
                tihu[key] = val
        self.tihu = tihu
        self.load_variables()
        # self.homograph2features = construct_homograph_dictionary()

    def load_variables(self):
        self.variables = np.load(os.path.join(dirname, self.checkpoint), allow_pickle=True)
        self.enc_emb = self.variables.item().get("encoder.emb.weight")  # (29, 64). (len(graphemes), emb)
        self.enc_w_ih = self.variables.item().get("encoder.rnn.weight_ih_l0")  # (3*128, 64)
        self.enc_w_hh = self.variables.item().get("encoder.rnn.weight_hh_l0")  # (3*128, 128)
        self.enc_b_ih = self.variables.item().get("encoder.rnn.bias_ih_l0")  # (3*128,)
        self.enc_b_hh = self.variables.item().get("encoder.rnn.bias_hh_l0")  # (3*128,)

        self.dec_emb = self.variables.item().get("decoder.emb.weight")  # (74, 64). (len(phonemes), emb)
        self.dec_w_ih = self.variables.item().get("decoder.rnn.weight_ih_l0")  # (3*128, 64)
        self.dec_w_hh = self.variables.item().get("decoder.rnn.weight_hh_l0")  # (3*128, 128)
        self.dec_b_ih = self.variables.item().get("decoder.rnn.bias_ih_l0")  # (3*128,)
        self.dec_b_hh = self.variables.item().get("decoder.rnn.bias_hh_l0")  # (3*128,)
        self.fc_w = self.variables.item().get("decoder.fc.weight")  # (74, 128)
        self.fc_b = self.variables.item().get("decoder.fc.bias")  # (74,)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def grucell(self, x, h, w_ih, w_hh, b_ih, b_hh):
        rzn_ih = np.matmul(x, w_ih.T) + b_ih
        rzn_hh = np.matmul(h, w_hh.T) + b_hh

        rz_ih, n_ih = rzn_ih[:, :rzn_ih.shape[-1] * 2 // 3], rzn_ih[:, rzn_ih.shape[-1] * 2 // 3:]
        rz_hh, n_hh = rzn_hh[:, :rzn_hh.shape[-1] * 2 // 3], rzn_hh[:, rzn_hh.shape[-1] * 2 // 3:]

        rz = self.sigmoid(rz_ih + rz_hh)
        r, z = np.split(rz, 2, -1)

        n = np.tanh(n_ih + r * n_hh)
        h = (1 - z) * n + z * h

        return h

    def gru(self, x, steps, w_ih, w_hh, b_ih, b_hh, h0=None):
        if h0 is None:
            h0 = np.zeros((x.shape[0], w_hh.shape[1]), np.float32)
        h = h0  # initial hidden state
        outputs = np.zeros((x.shape[0], steps, w_hh.shape[1]), np.float32)
        for t in range(steps):
            h = self.grucell(x[:, t, :], h, w_ih, w_hh, b_ih, b_hh)  # (b, h)
            outputs[:, t, ::] = h
        return outputs

    def encode(self, word):
        chars = list(word) + ["</s>"]
        x = [self.g2idx.get(char, self.g2idx["<unk>"]) for char in chars]
        x = np.take(self.enc_emb, np.expand_dims(x, 0), axis=0)

        return x

    def predict(self, word):
        # encoder
        enc = self.encode(word)
        enc = self.gru(enc, len(word) + 1, self.enc_w_ih, self.enc_w_hh,
                       self.enc_b_ih, self.enc_b_hh, h0=np.zeros((1, self.enc_w_hh.shape[-1]), np.float32))
        last_hidden = enc[:, -1, :]

        # decoder
        dec = np.take(self.dec_emb, [2], axis=0)  # 2: <s>
        h = last_hidden

        preds = []
        for i in range(20):
            h = self.grucell(dec, h, self.dec_w_ih, self.dec_w_hh, self.dec_b_ih, self.dec_b_hh)  # (b, h)
            logits = np.matmul(h, self.fc_w.T) + self.fc_b
            pred = logits.argmax()
            if pred == 3: break  # 3: </s>
            preds.append(pred)
            dec = np.take(self.dec_emb, [pred], axis=0)

        preds = [self.idx2p.get(idx, "<unk>") for idx in preds]
        return preds

    def __call__(self, text):
        # preprocessing
        text = unicode(text)
        text = normalize_numbers(text)
        # text = ''.join(char for char in unicodedata.normalize('NFD', text)
        #                if unicodedata.category(char) != 'Mn')  # Strip accents
        # text = re.sub("[^ a-z'.,?!\-]", "", text)

        normalizer = hazm.Normalizer()
        text = normalizer.normalize(text)
        # tokenization
        words = hazm.word_tokenize(text)
        # tokens = pos_tag(words)  # tuples of (word, tag)

        # steps
        prons = []
        for word in words:
            if not any(letter in word for letter in self.graphemes):
                pron = [word]

            # elif word in self.homograph2features:  # Check homograph
            #     pron1, pron2, pos1 = self.homograph2features[word]
            #     if pos.startswith(pos1):
            #         pron = pron1
            #     else:
            #         pron = pron2
            elif word in self.tihu:  # lookup tihu dict
                pron = self.tihu[word]
            else: # predict for oov
                pron = self.predict(word)

            prons.extend(pron)
            prons.extend([" "])

        return prons[:-1]

TEXT = "زان یار دلنوازم شکریست با شکایت"
CHECKPOINT = 'logs-01/checkpoint.npy'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default=TEXT, help='the input character sequence')
    parser.add_argument('--checkpoint', default=CHECKPOINT, help='the input character sequence')
    args = parser.parse_args()
    g2p = G2p(args)
    out = g2p(args.text)
    print("".join(symbol for symbol in out))
