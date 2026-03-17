import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.n_layers  = n_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden_dim, n_layers, batch_first=True,
                          dropout=dropout if n_layers>1 else 0, bidirectional=True)
        self.fc      = nn.Linear(hidden_dim*2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = hidden.view(self.n_layers, 2, hidden.size(1), -1)
        hidden = torch.tanh(self.fc(
            torch.cat((hidden[:,0], hidden[:,1]), dim=2)))
        return outputs, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim*3, hidden_dim)
        self.v    = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        B, L, _ = encoder_outputs.shape
        top = hidden[-1].unsqueeze(1).expand(B, L, -1)
        energy = torch.tanh(self.attn(torch.cat((top, encoder_outputs), dim=2)))
        return F.softmax(self.v(energy).squeeze(2), dim=1)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_dim)
        self.rnn = nn.GRU(embed_dim+hidden_dim*2, hidden_dim, n_layers,
                          batch_first=True, dropout=dropout if n_layers>1 else 0)
        self.fc_out  = nn.Linear(hidden_dim*3+embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt_token, hidden, encoder_outputs):
        embedded     = self.dropout(self.embedding(tgt_token.unsqueeze(1)))
        attn_weights = self.attention(hidden, encoder_outputs)
        context      = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        rnn_input    = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        pred = self.fc_out(torch.cat((output, context, embedded), dim=2)).squeeze(1)
        return pred, hidden, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device  = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        B, tgt_len = tgt.shape
        vocab_size  = self.decoder.fc_out.out_features
        outputs     = torch.zeros(B, tgt_len, vocab_size).to(self.device)
        enc_out, hidden = self.encoder(src)
        dec_input = tgt[:,0]
        for t in range(1, tgt_len):
            pred, hidden, _ = self.decoder(dec_input, hidden, enc_out)
            outputs[:,t]    = pred
            dec_input = tgt[:,t] if random.random()<teacher_forcing_ratio else pred.argmax(1)
        return outputs