import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from hparams import hp

class Encoder(nn.Module):
    def __init__(self, emb_units, hidden_units, g2idx):
        super().__init__()
        self.emb_units = emb_units
        self.hidden_units = hidden_units
        self.emb = nn.Embedding(len(g2idx), emb_units)
        self.rnn = nn.GRU(emb_units, hidden_units, batch_first=True)

    def forward(self, x, seqlens):
        x = self.emb(x)

        # reverse sorting by length
        seqlens = torch.IntTensor(seqlens)
        seqlens_sorted, perm_idx = seqlens.sort(0, descending=True)
        _, unperm_idx = perm_idx.sort(0)  # for recovery
        x = x[perm_idx]

        # packing -> rnn -> unpacking -> position recovery
        packed_input = pack_padded_sequence(x, seqlens_sorted, batch_first=True)
        outputs, last_hidden = self.rnn(packed_input)
        #         outputs, _ = pad_packed_sequence(outputs, batch_first=True, total_length=x.size()[1])
        #         outputs = outputs[unperm_idx]

        # last hidden
        last_hidden = last_hidden.permute(1, 2, 0)
        last_hidden = last_hidden.view(last_hidden.size()[0], -1)
        last_hidden = last_hidden[unperm_idx]

        return last_hidden

class Decoder(nn.Module):
    def __init__(self, emb_units, hidden_units, p2idx):
        super().__init__()

        self.emb_units = emb_units
        self.hidden_units = hidden_units
        self.emb = nn.Embedding(len(p2idx), emb_units)
        self.rnn = nn.GRU(emb_units, hidden_units, batch_first=True)
        self.fc = nn.Linear(hidden_units, len(p2idx))

    def forward(self, decoder_inputs, h0):
        decoder_inputs = self.emb(decoder_inputs)

        outputs, last_hidden = self.rnn(decoder_inputs, h0)
        logits = self.fc(outputs)  # (N, T, V)
        y_hat = logits.argmax(-1)

        return logits, y_hat, last_hidden

class Net(nn.Module):
    def __init__(self, g2idx, idx2g, p2idx, idx2p):
        super().__init__()
        self.encoder = Encoder(hp.emb_units, hp.hidden_units, g2idx)
        self.decoder = Decoder(hp.emb_units, hp.hidden_units, p2idx)
        self.g2idx = g2idx
        self.idx2g = idx2g
        self.p2idx = p2idx
        self.idx2p = idx2p

    def forward(self, x, seqlens, decoder_inputs, teacher_forcing=True, dec_maxlen=None):
        '''
        At training, teacher forcing is True.
        if teacher_forcing is True, dec_maxlen must be given.
        '''
        last_hidden = self.encoder(x, seqlens)
        h0 = last_hidden.unsqueeze(0)

        if teacher_forcing:
            logits, y_hat, h0 = self.decoder(decoder_inputs, h0)
        else:
            logits, y_hat = [], []
            for t in range(dec_maxlen):
                _logits, _y_hat, h0 = self.decoder(decoder_inputs[:, t:t + 1, :],
                                                   h0)  # logits: (N, 1, V), y_hat: (N, 1)
                logits.append(_logits)
                y_hat.append(_y_hat)
                if _y_hat.sum(1) == 0: break  # all samples reach padding.

            logits = torch.cat(logits, 1)
            y_hat = torch.cat(y_hat, 1)

        return logits, y_hat
