import copy
import math
import torch
import torch.nn as nn

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    def __init__(self, shape, eps=1e-5):
        super().__init__()
        self.shape = tuple(shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(self.shape))
        self.bias = nn.Parameter(torch.empty(self.shape))
        self._reset_parameters()

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias
    
    def _reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout=.1, norm_first=True):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x, sublayer):
        if self.norm_first:
            x = x + self.dropout(sublayer(self.norm(x)))
        else:
            x = self.norm(x + self.dropout(sublayer(x)))
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout=.1):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = _get_clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, src_mask=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, src_mask))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = norm
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        if self.norm is not None:
            x = self.norm(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout=.1):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = _get_clones(SublayerConnection(d_model, dropout), 3)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, memory_mask))
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.norm = norm

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._reset_parameters()
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        return output

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linears = _get_clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)

        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

def build_transformer(d_model=512, d_ff=2048, h=8, N=6, dropout=.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(d_model, h)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)
    )
    return model