import copy
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
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias

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

    def forward(self, x, mask=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
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

    def forward(self, x, mem, src_mask=None, tgt_mask=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, mem, mem, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.norm = norm

    def forward(self, x, mem, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, mem, src_mask, tgt_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x