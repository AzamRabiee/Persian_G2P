# -*- coding: utf-8 -*-
# /usr/bin/python
'''
By Azam Rabiee (azrabiee@gmail.com)
forked from
https://github.com/Kyubyong/nlp_made_easy/blob/master/PyTorch%20seq2seq%20template%20based%20on%20the%20g2p%20task.ipynb
'''
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from hparams import hp
from data_feeder import drop_lengthy_samples, prepare_data, load_vocab, pad, convert_ids_to_phonemes, G2pDataset
from model import Encoder, Decoder, Net
from distance import levenshtein
import argparse

def train(model, iterator, optimizer, criterion, device):
    model.train()
    for i, batch in enumerate(iterator):
        x, x_seqlens, words, decoder_inputs, y, y_seqlens, prons = batch

        x, decoder_inputs = x.to(device), decoder_inputs.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits, y_hat = model(x, x_seqlens, decoder_inputs)

        # calc loss
        logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
        y = y.view(-1)  # (N*T,)
        loss = criterion(logits, y)
        loss.backward()

        optimizer.step()

        if i and i % 100 == 0:
            print(f"step: {i}, loss: {loss.item()}")

def calc_per(Y_true, Y_pred):
    '''Calc phoneme error rate
    Y_true: list of predicted phoneme sequences. e.g., [["k", "a", "m", "a", "n", "d"], ...]
    Y_pred: list of ground truth phoneme sequences. e.g., [["k", "a", "m", "a", "n", "d"], ...]
    '''
    num_phonemes, num_erros = 0, 0
    for y_true, y_pred in zip(Y_true, Y_pred):
        num_phonemes += len(y_true)
        num_erros += levenshtein(y_true, y_pred)

    per = round(num_erros / num_phonemes, 4)
    return per

def eval(model, iterator, device, dec_maxlen, run_name):
    model.eval()
    Y_true, Y_pred = [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            x, x_seqlens, words, decoder_inputs, y, y_seqlens, prons = batch
            x, decoder_inputs = x.to(device), decoder_inputs.to(device)

            _, y_hat = model(x, x_seqlens, decoder_inputs, dec_maxlen)

            y = y.to('cpu').numpy().tolist()
            y_hat = y_hat.to('cpu').numpy().tolist()
            for yy, yy_hat in zip(y, y_hat):
                y_true = convert_ids_to_phonemes(yy, model.idx2p)
                y_pred = convert_ids_to_phonemes(yy_hat, model.idx2p)
                Y_true.append(y_true)
                Y_pred.append(y_pred)

    # calc per.
    per = calc_per(Y_true, Y_pred)
    print("per: %.4f" % per)

    with open("logs-%s/result.txt" % run_name, "w") as fout:
        for y_true, y_pred in zip(Y_true, Y_pred):
            fout.write(" ".join(y_true) + "\n")
            fout.write(" ".join(y_pred) + "\n\n")

    return per


# Train & Evaluate
def main(args):
    train_words, eval_words, test_words, train_prons, eval_prons, test_prons = prepare_data(args.dictionary)
    train_words, train_prons = drop_lengthy_samples(train_words, train_prons, hp.enc_maxlen, hp.dec_maxlen)

    g2idx, idx2g, p2idx, idx2p = load_vocab()
    train_dataset = G2pDataset(train_words, train_prons, g2idx, p2idx)
    eval_dataset = G2pDataset(eval_words, eval_prons, g2idx, p2idx)

    train_iter = data.DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=pad)
    eval_iter = data.DataLoader(eval_dataset, batch_size=hp.batch_size, shuffle=False, collate_fn=pad)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(g2idx, idx2g, p2idx, idx2p)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(1, hp.num_epochs + 1):
        print(f"\nepoch: {epoch}")
        train(model, train_iter, optimizer, criterion, device)
        eval(model, eval_iter, device, hp.dec_maxlen, args.name)

    # test
    test_dataset = G2pDataset(test_words, test_prons, g2idx, p2idx)
    test_iter = data.DataLoader(test_dataset, batch_size=hp.batch_size, shuffle=False, collate_fn=pad)
    eval(model, test_iter, device, hp.dec_maxlen, args.name)
    print('target and output results for eval and test sets are saved in \'logs-%s/result.txt\' file.' % args.name)
    print()
    print('Here are some samples for the test set:')
    print(open('logs-%s/result.txt' % args.name, 'r').read().splitlines()[-100:])

    print("\n\nModel's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # save model parameters in a numpy dictionary in 'checkpoint.npy'
    params_np_dic = {}
    for param_tensor in model.state_dict():
        params_np_dic[param_tensor] = np.array(model.state_dict()[param_tensor].cpu())
    np.save('logs-%s/checkpoint.npy' % args.name, params_np_dic)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='02', help='the run name; a log folder with logs-<run-name> will be made '
                                                     'for storing checkpoints, results and logs of loss and PERs')
    parser.add_argument('--dictionary', default="tihudict.dict", help='path to the word-pronunciation dictionary')
    args = parser.parse_args()
    os.makedirs('logs-%s' % args.name, exist_ok=True)
    main(args)